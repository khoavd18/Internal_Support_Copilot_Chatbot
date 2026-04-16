from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from time import perf_counter
from typing import Any, Dict, Iterator, Mapping, Optional
from uuid import uuid4

from src.core.logging_utils import get_log_context
from src.core.settings import OBSERVABILITY_BACKEND, OBSERVABILITY_TRACE_HISTORY_LIMIT


_CURRENT_SPAN: ContextVar["BaseSpan | None"] = ContextVar("current_observability_span", default=None)
_BACKEND: "ObservabilityBackend | None" = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_attribute_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def normalize_attributes(attributes: Mapping[str, Any] | None) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in dict(attributes or {}).items():
        cleaned = _normalize_attribute_value(value)
        if cleaned in (None, ""):
            continue
        normalized[str(key)] = cleaned
    return normalized


def _metric_key(name: str, attributes: Mapping[str, Any] | None) -> tuple[str, tuple[tuple[str, Any], ...]]:
    normalized = normalize_attributes(attributes)
    return str(name), tuple(sorted(normalized.items()))


@dataclass
class HistogramSummary:
    count: int = 0
    sum: float = 0.0
    min: float | None = None
    max: float | None = None
    last: float | None = None

    def observe(self, value: float) -> None:
        numeric = float(value)
        self.count += 1
        self.sum += numeric
        self.last = numeric
        self.min = numeric if self.min is None else min(self.min, numeric)
        self.max = numeric if self.max is None else max(self.max, numeric)


@dataclass
class CompletedSpan:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str = ""
    status: str = "ok"
    started_at: str = ""
    ended_at: str = ""
    duration_ms: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    exception_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "attributes": dict(self.attributes),
            "exception_type": self.exception_type,
        }


class BaseSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def record_exception(self, exc: BaseException) -> None:
        raise NotImplementedError

    def finish(self, *, status: str = "ok", duration_ms: float | None = None) -> CompletedSpan | None:
        raise NotImplementedError


class NoopSpan(BaseSpan):
    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def record_exception(self, exc: BaseException) -> None:
        return None

    def finish(self, *, status: str = "ok", duration_ms: float | None = None) -> CompletedSpan | None:
        return None


class InMemorySpan(BaseSpan):
    def __init__(
        self,
        *,
        backend: "InMemoryObservabilityBackend",
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ):
        parent = _CURRENT_SPAN.get()
        log_context = get_log_context()
        self.backend = backend
        self.name = str(name)
        self.trace_id = parent.trace_id if parent is not None else str(log_context.get("request_id") or uuid4().hex)
        self.parent_span_id = parent.span_id if parent is not None else ""
        self.span_id = uuid4().hex
        self.attributes = normalize_attributes(attributes)
        for field_name in ("request_id", "session_id", "user_id", "action_id", "agent_name"):
            value = log_context.get(field_name)
            if value and field_name not in self.attributes:
                self.attributes[field_name] = value
        self.started_at = _utc_now()
        self._started_perf = perf_counter()
        self._exception_type = ""
        self._finished = False
        self._completed_span: CompletedSpan | None = None
        self._token = _CURRENT_SPAN.set(self)

    def set_attribute(self, key: str, value: Any) -> None:
        cleaned = _normalize_attribute_value(value)
        if cleaned in (None, ""):
            return
        self.attributes[str(key)] = cleaned

    def record_exception(self, exc: BaseException) -> None:
        self._exception_type = exc.__class__.__name__

    def finish(self, *, status: str = "ok", duration_ms: float | None = None) -> CompletedSpan:
        if self._finished:
            if self._completed_span is None:
                raise RuntimeError("Span has already finished without a stored snapshot.")
            return self._completed_span

        resolved_duration = float(duration_ms) if duration_ms is not None else round((perf_counter() - self._started_perf) * 1000, 2)
        span = CompletedSpan(
            name=self.name,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            status=str(status or "ok"),
            started_at=self.started_at,
            ended_at=_utc_now(),
            duration_ms=resolved_duration,
            attributes=dict(self.attributes),
            exception_type=self._exception_type,
        )
        self._completed_span = span
        self.backend._record_completed_span(span)
        self._finished = True
        _CURRENT_SPAN.reset(self._token)
        return span


class ObservabilityBackend:
    def increment_counter(
        self,
        name: str,
        *,
        value: float = 1.0,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def record_histogram(
        self,
        name: str,
        value: float,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> BaseSpan:
        raise NotImplementedError

    def snapshot(self) -> Dict[str, Any]:
        raise NotImplementedError


class NoopObservabilityBackend(ObservabilityBackend):
    def increment_counter(
        self,
        name: str,
        *,
        value: float = 1.0,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        return None

    def record_histogram(
        self,
        name: str,
        value: float,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        return None

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> BaseSpan:
        return NoopSpan()

    def snapshot(self) -> Dict[str, Any]:
        return {"counters": [], "histograms": [], "spans": []}


class InMemoryObservabilityBackend(ObservabilityBackend):
    def __init__(self, *, trace_history_limit: int = 200):
        self.trace_history_limit = max(int(trace_history_limit or 0), 0)
        self._lock = RLock()
        self._counters: Dict[tuple[str, tuple[tuple[str, Any], ...]], float] = {}
        self._histograms: Dict[tuple[str, tuple[tuple[str, Any], ...]], HistogramSummary] = {}
        self.completed_spans: deque[CompletedSpan] = deque(maxlen=self.trace_history_limit or None)

    def increment_counter(
        self,
        name: str,
        *,
        value: float = 1.0,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        key = _metric_key(name, attributes)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + float(value)

    def record_histogram(
        self,
        name: str,
        value: float,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        key = _metric_key(name, attributes)
        with self._lock:
            summary = self._histograms.setdefault(key, HistogramSummary())
            summary.observe(value)

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> BaseSpan:
        return InMemorySpan(backend=self, name=name, attributes=attributes)

    def _record_completed_span(self, span: CompletedSpan) -> None:
        if self.trace_history_limit == 0:
            return
        with self._lock:
            self.completed_spans.append(span)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counters = [
                {
                    "name": name,
                    "attributes": dict(attributes),
                    "value": value,
                }
                for (name, attributes), value in sorted(self._counters.items(), key=lambda item: item[0])
            ]
            histograms = [
                {
                    "name": name,
                    "attributes": dict(attributes),
                    "count": summary.count,
                    "sum": summary.sum,
                    "min": summary.min,
                    "max": summary.max,
                    "last": summary.last,
                }
                for (name, attributes), summary in sorted(self._histograms.items(), key=lambda item: item[0])
            ]
            spans = [span.to_dict() for span in list(self.completed_spans)]
        return {
            "counters": counters,
            "histograms": histograms,
            "spans": spans,
        }


class Observation:
    def __init__(
        self,
        *,
        backend: ObservabilityBackend,
        span_name: str,
        metric_name: str | None,
        metric_attributes: Mapping[str, Any] | None = None,
        span_attributes: Mapping[str, Any] | None = None,
    ):
        self.backend = backend
        self.metric_name = str(metric_name).strip() if metric_name else ""
        self.metric_attributes = normalize_attributes(metric_attributes)
        self.span = backend.start_span(
            span_name,
            attributes=span_attributes or self.metric_attributes,
        )
        self._started_perf = perf_counter()
        self._status = "ok"
        self._finished = False
        self.duration_ms = 0.0

    def set_attribute(self, key: str, value: Any) -> None:
        self.span.set_attribute(key, value)

    def set_metric_attribute(self, key: str, value: Any) -> None:
        cleaned = _normalize_attribute_value(value)
        if cleaned in (None, ""):
            return
        self.metric_attributes[str(key)] = cleaned

    def record_exception(self, exc: BaseException) -> None:
        self._status = "error"
        self.span.record_exception(exc)

    def finish(self, *, status: str | None = None) -> float:
        if self._finished:
            return self.duration_ms

        resolved_status = str(status or self._status or "ok")
        self.duration_ms = round((perf_counter() - self._started_perf) * 1000, 2)
        metric_attributes = dict(self.metric_attributes)
        metric_attributes.setdefault("status", resolved_status)
        if self.metric_name:
            self.backend.record_histogram(
                self.metric_name,
                self.duration_ms,
                attributes=metric_attributes,
            )
        self.span.finish(status=resolved_status, duration_ms=self.duration_ms)
        self._finished = True
        return self.duration_ms


def configure_observability(*, force: bool = False) -> None:
    global _BACKEND
    if _BACKEND is not None and not force:
        return

    backend_name = str(OBSERVABILITY_BACKEND or "memory").strip().lower()
    if backend_name == "noop":
        _BACKEND = NoopObservabilityBackend()
    else:
        _BACKEND = InMemoryObservabilityBackend(
            trace_history_limit=OBSERVABILITY_TRACE_HISTORY_LIMIT,
        )


def get_observability_backend() -> ObservabilityBackend:
    global _BACKEND
    if _BACKEND is None:
        configure_observability()
    return _BACKEND


def set_observability_backend(backend: ObservabilityBackend) -> None:
    global _BACKEND
    _BACKEND = backend


def increment_counter(
    name: str,
    *,
    value: float = 1.0,
    attributes: Mapping[str, Any] | None = None,
) -> None:
    get_observability_backend().increment_counter(
        name,
        value=value,
        attributes=attributes,
    )


def record_histogram(
    name: str,
    value: float,
    *,
    attributes: Mapping[str, Any] | None = None,
) -> None:
    get_observability_backend().record_histogram(
        name,
        value,
        attributes=attributes,
    )


def start_span(
    name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
) -> BaseSpan:
    return get_observability_backend().start_span(name, attributes=attributes)


@contextmanager
def observe_duration(
    span_name: str,
    *,
    metric_name: str | None = None,
    metric_attributes: Mapping[str, Any] | None = None,
    span_attributes: Mapping[str, Any] | None = None,
) -> Iterator[Observation]:
    observation = Observation(
        backend=get_observability_backend(),
        span_name=span_name,
        metric_name=metric_name,
        metric_attributes=metric_attributes,
        span_attributes=span_attributes,
    )
    try:
        yield observation
    except Exception as exc:
        observation.record_exception(exc)
        observation.finish(status="error")
        raise
    else:
        observation.finish(status="ok")


def get_observability_snapshot() -> Dict[str, Any]:
    return get_observability_backend().snapshot()


__all__ = [
    "CompletedSpan",
    "InMemoryObservabilityBackend",
    "NoopObservabilityBackend",
    "Observation",
    "ObservabilityBackend",
    "configure_observability",
    "get_observability_backend",
    "get_observability_snapshot",
    "increment_counter",
    "normalize_attributes",
    "observe_duration",
    "record_histogram",
    "set_observability_backend",
    "start_span",
]
