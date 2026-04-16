from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, Iterator, Sequence
from urllib.parse import quote

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.memory import InMemorySaver

from src.core.security import sanitize_error_text
from src.core.settings import (
    GRAPH_CHECKPOINT_TTL_SECONDS,
    GRAPH_CHECKPOINTER_BACKEND,
    REDIS_KEY_PREFIX,
    REDIS_URL,
)

try:
    import redis
except ImportError:  # pragma: no cover - exercised indirectly when redis isn't installed
    redis = None  # type: ignore[assignment]


def _key_component(value: str) -> str:
    return quote(str(value or "").strip(), safe="")


class RedisCheckpointSaver(BaseCheckpointSaver[str]):
    def __init__(
        self,
        client: Any,
        *,
        key_prefix: str = REDIS_KEY_PREFIX,
        ttl_seconds: int = GRAPH_CHECKPOINT_TTL_SECONDS,
    ) -> None:
        super().__init__()
        self.client = client
        self.key_prefix = (key_prefix or "internal_support_copilot").strip()
        self.ttl_seconds = max(0, int(ttl_seconds))

    @classmethod
    def from_url(
        cls,
        url: str = REDIS_URL,
        *,
        key_prefix: str = REDIS_KEY_PREFIX,
        ttl_seconds: int = GRAPH_CHECKPOINT_TTL_SECONDS,
    ) -> "RedisCheckpointSaver":
        if redis is None:
            raise RuntimeError(
                "GRAPH_CHECKPOINTER_BACKEND=redis requires the `redis` package to be installed."
            )
        client = redis.from_url(url, decode_responses=False)
        return cls(client, key_prefix=key_prefix, ttl_seconds=ttl_seconds)

    def _threads_key(self) -> str:
        return f"{self.key_prefix}:graph:threads"

    def _namespaces_key(self, thread_id: str) -> str:
        return f"{self.key_prefix}:graph:thread:{_key_component(thread_id)}:namespaces"

    def _checkpoint_index_key(self, thread_id: str, checkpoint_ns: str) -> str:
        return (
            f"{self.key_prefix}:graph:thread:{_key_component(thread_id)}:"
            f"ns:{_key_component(checkpoint_ns)}:checkpoints"
        )

    def _checkpoint_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        return (
            f"{self.key_prefix}:graph:thread:{_key_component(thread_id)}:"
            f"ns:{_key_component(checkpoint_ns)}:checkpoint:{_key_component(checkpoint_id)}"
        )

    def _apply_ttl(self, pipe: Any, *keys: str) -> None:
        if self.ttl_seconds <= 0:
            return
        for key in keys:
            pipe.expire(key, self.ttl_seconds)

    def _serialize_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        payload_type, payload = self.serde.dumps_typed(document)
        return {
            "payload_type": payload_type,
            "payload": payload,
        }

    def _deserialize_document(self, raw: Dict[Any, Any] | None) -> Dict[str, Any] | None:
        if not raw:
            return None

        payload_type = raw.get(b"payload_type") or raw.get("payload_type")
        payload = raw.get(b"payload") or raw.get("payload")
        if payload_type is None or payload is None:
            return None

        if isinstance(payload_type, bytes):
            payload_type = payload_type.decode("utf-8")

        return self.serde.loads_typed((payload_type, payload))

    def _build_tuple(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        document: Dict[str, Any],
    ) -> CheckpointTuple:
        parent_checkpoint_id = document.get("parent_checkpoint_id")
        pending_writes_map = document.get("writes_map", {}) or {}

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=document["checkpoint"],
            metadata=document["metadata"],
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
            pending_writes=[
                (
                    entry["task_id"],
                    entry["channel"],
                    entry["value"],
                )
                for _, entry in sorted(
                    pending_writes_map.items(),
                    key=lambda item: int(str(item[0]).rsplit("|", 1)[-1]),
                )
            ],
        )

    def get_tuple(self, config: Dict[str, Any]) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        if checkpoint_id is None:
            checkpoint_ids = self.client.zrevrange(
                self._checkpoint_index_key(thread_id, checkpoint_ns),
                0,
                0,
            )
            if not checkpoint_ids:
                return None
            checkpoint_id = checkpoint_ids[0]
            if isinstance(checkpoint_id, bytes):
                checkpoint_id = checkpoint_id.decode("utf-8")

        raw = self.client.hgetall(self._checkpoint_key(thread_id, checkpoint_ns, checkpoint_id))
        document = self._deserialize_document(raw)
        if document is None:
            return None

        return self._build_tuple(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            document=document,
        )

    def list(
        self,
        config: Dict[str, Any] | None,
        *,
        filter: dict[str, Any] | None = None,
        before: Dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        thread_ids = (
            [config["configurable"]["thread_id"]]
            if config
            else sorted(
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in self.client.smembers(self._threads_key())
            )
        )

        config_checkpoint_ns = config["configurable"].get("checkpoint_ns") if config else None
        config_checkpoint_id = get_checkpoint_id(config) if config else None
        before_checkpoint_id = get_checkpoint_id(before) if before else None

        for thread_id in thread_ids:
            namespaces = sorted(
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in self.client.smembers(self._namespaces_key(thread_id))
            )
            for checkpoint_ns in namespaces:
                if config_checkpoint_ns is not None and checkpoint_ns != config_checkpoint_ns:
                    continue

                checkpoint_ids = self.client.zrevrange(
                    self._checkpoint_index_key(thread_id, checkpoint_ns),
                    0,
                    -1,
                )
                for raw_checkpoint_id in checkpoint_ids:
                    checkpoint_id = (
                        raw_checkpoint_id.decode("utf-8")
                        if isinstance(raw_checkpoint_id, bytes)
                        else str(raw_checkpoint_id)
                    )

                    if config_checkpoint_id and checkpoint_id != config_checkpoint_id:
                        continue
                    if before_checkpoint_id and checkpoint_id >= before_checkpoint_id:
                        continue

                    raw = self.client.hgetall(
                        self._checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
                    )
                    document = self._deserialize_document(raw)
                    if document is None:
                        continue

                    metadata = document.get("metadata", {})
                    if filter and not all(metadata.get(key) == value for key, value in filter.items()):
                        continue

                    if limit is not None and limit <= 0:
                        return
                    if limit is not None:
                        limit -= 1

                    yield self._build_tuple(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                        document=document,
                    )

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> Dict[str, Any]:
        del new_versions

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        checkpoint_key = self._checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        index_key = self._checkpoint_index_key(thread_id, checkpoint_ns)
        threads_key = self._threads_key()
        namespaces_key = self._namespaces_key(thread_id)

        document = {
            "checkpoint": checkpoint,
            "metadata": get_checkpoint_metadata(config, metadata),
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
            "writes_map": {},
        }

        pipe = self.client.pipeline()
        pipe.hset(checkpoint_key, mapping=self._serialize_document(document))
        pipe.zadd(index_key, {checkpoint_id: float(time.time_ns())})
        pipe.sadd(threads_key, thread_id)
        pipe.sadd(namespaces_key, checkpoint_ns)
        self._apply_ttl(pipe, checkpoint_key, index_key, threads_key, namespaces_key)
        pipe.execute()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        checkpoint_key = self._checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        raw = self.client.hgetall(checkpoint_key)
        document = self._deserialize_document(raw)
        if document is None:
            return

        writes_map = dict(document.get("writes_map", {}) or {})
        for idx, (channel, value) in enumerate(writes):
            write_idx = WRITES_IDX_MAP.get(channel, idx)
            map_key = f"{task_id}|{write_idx}"
            if write_idx >= 0 and map_key in writes_map:
                continue

            writes_map[map_key] = {
                "task_id": task_id,
                "channel": channel,
                "value": value,
                "task_path": task_path,
            }

        document["writes_map"] = writes_map
        pipe = self.client.pipeline()
        pipe.hset(checkpoint_key, mapping=self._serialize_document(document))
        self._apply_ttl(pipe, checkpoint_key)
        pipe.execute()

    def _delete_checkpoint(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> None:
        checkpoint_key = self._checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        index_key = self._checkpoint_index_key(thread_id, checkpoint_ns)
        pipe = self.client.pipeline()
        pipe.delete(checkpoint_key)
        pipe.zrem(index_key, checkpoint_id)
        pipe.execute()

    def delete_thread(self, thread_id: str) -> None:
        namespaces_key = self._namespaces_key(thread_id)
        namespaces = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in self.client.smembers(namespaces_key)
        ]

        pipe = self.client.pipeline()
        for checkpoint_ns in namespaces:
            index_key = self._checkpoint_index_key(thread_id, checkpoint_ns)
            checkpoint_ids = self.client.zrange(index_key, 0, -1)
            for raw_checkpoint_id in checkpoint_ids:
                checkpoint_id = (
                    raw_checkpoint_id.decode("utf-8")
                    if isinstance(raw_checkpoint_id, bytes)
                    else str(raw_checkpoint_id)
                )
                pipe.delete(self._checkpoint_key(thread_id, checkpoint_ns, checkpoint_id))
            pipe.delete(index_key)

        pipe.delete(namespaces_key)
        pipe.srem(self._threads_key(), thread_id)
        pipe.execute()

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        if not run_ids:
            return
        run_id_set = {str(item) for item in run_ids}
        for item in list(self.list(None)):
            run_id = str(item.metadata.get("run_id") or "")
            if run_id in run_id_set:
                thread_id = item.config["configurable"]["thread_id"]
                checkpoint_ns = item.config["configurable"].get("checkpoint_ns", "")
                checkpoint_id = item.config["configurable"]["checkpoint_id"]
                self._delete_checkpoint(thread_id, checkpoint_ns, checkpoint_id)

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        source_namespaces = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in self.client.smembers(self._namespaces_key(source_thread_id))
        ]
        for checkpoint_ns in source_namespaces:
            checkpoint_ids = self.client.zrange(
                self._checkpoint_index_key(source_thread_id, checkpoint_ns),
                0,
                -1,
            )
            for raw_checkpoint_id in checkpoint_ids:
                checkpoint_id = (
                    raw_checkpoint_id.decode("utf-8")
                    if isinstance(raw_checkpoint_id, bytes)
                    else str(raw_checkpoint_id)
                )
                raw = self.client.hgetall(
                    self._checkpoint_key(source_thread_id, checkpoint_ns, checkpoint_id)
                )
                document = self._deserialize_document(raw)
                if document is None:
                    continue

                target_checkpoint_key = self._checkpoint_key(
                    target_thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                )
                target_index_key = self._checkpoint_index_key(target_thread_id, checkpoint_ns)
                target_namespaces_key = self._namespaces_key(target_thread_id)

                pipe = self.client.pipeline()
                pipe.hset(target_checkpoint_key, mapping=self._serialize_document(document))
                pipe.zadd(target_index_key, {checkpoint_id: float(time.time_ns())})
                pipe.sadd(self._threads_key(), target_thread_id)
                pipe.sadd(target_namespaces_key, checkpoint_ns)
                self._apply_ttl(
                    pipe,
                    target_checkpoint_key,
                    target_index_key,
                    target_namespaces_key,
                    self._threads_key(),
                )
                pipe.execute()

    def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        for thread_id in thread_ids:
            if strategy == "delete":
                self.delete_thread(thread_id)
                continue

            namespaces = [
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in self.client.smembers(self._namespaces_key(thread_id))
            ]
            for checkpoint_ns in namespaces:
                checkpoint_ids = self.client.zrevrange(
                    self._checkpoint_index_key(thread_id, checkpoint_ns),
                    0,
                    -1,
                )
                for raw_checkpoint_id in checkpoint_ids[1:]:
                    checkpoint_id = (
                        raw_checkpoint_id.decode("utf-8")
                        if isinstance(raw_checkpoint_id, bytes)
                        else str(raw_checkpoint_id)
                    )
                    self._delete_checkpoint(thread_id, checkpoint_ns, checkpoint_id)

    async def aget_tuple(self, config: Dict[str, Any]) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: Dict[str, Any] | None,
        *,
        filter: dict[str, Any] | None = None,
        before: Dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> Dict[str, Any]:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        self.delete_for_runs(run_ids)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        self.copy_thread(source_thread_id, target_thread_id)

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        self.prune(thread_ids, strategy=strategy)

    def healthcheck(self) -> Dict[str, Any]:
        payload = {
            "backend": "redis",
            "ok": False,
            "key_prefix": self.key_prefix,
        }
        try:
            payload["ok"] = bool(self.client.ping())
        except Exception as exc:
            payload["error"] = sanitize_error_text(exc, max_length=240)
        return payload


_GRAPH_CHECKPOINTER: BaseCheckpointSaver[str] | None = None


def build_graph_checkpointer() -> BaseCheckpointSaver[str]:
    if GRAPH_CHECKPOINTER_BACKEND == "redis":
        return RedisCheckpointSaver.from_url()
    return InMemorySaver()


def get_graph_checkpointer() -> BaseCheckpointSaver[str]:
    global _GRAPH_CHECKPOINTER
    if _GRAPH_CHECKPOINTER is None:
        _GRAPH_CHECKPOINTER = build_graph_checkpointer()
    return _GRAPH_CHECKPOINTER


def configure_graph_checkpointer(checkpointer: BaseCheckpointSaver[str]) -> None:
    global _GRAPH_CHECKPOINTER
    _GRAPH_CHECKPOINTER = checkpointer


def reset_graph_checkpointer() -> None:
    global _GRAPH_CHECKPOINTER
    _GRAPH_CHECKPOINTER = None
