from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.core.observability import observe_duration
from src.rag.generation.answer_postprocess import (
    build_debug_rows,
    build_safe_fallback_answer,
    build_sources,
    clean_answer,
    extract_llm_text,
)
from src.rag.generation.prompt_builder import build_prompt
from src.rag.retrieval.retriever import retrieve_documents

logger = logging.getLogger(__name__)


class FunctionLLMWrapper:
    """Wrap a generate(prompt) function with an invoke(prompt) interface."""

    def __init__(self, fn):
        self.fn = fn

    def invoke(self, prompt: Any) -> Any:
        return self.fn(prompt)

    def __call__(self, prompt: Any) -> Any:
        return self.fn(prompt)


@lru_cache(maxsize=1)
def load_llm_backend():
    """Discover and load the local LLM implementation from src.llm.llm_model."""
    module = importlib.import_module("src.llm.llm_model")

    object_loader_names = ("get_llm", "build_llm", "load_llm", "create_llm")
    for fn_name in object_loader_names:
        if hasattr(module, fn_name):
            llm = getattr(module, fn_name)()
            logger.info("Loaded LLM via object loader: %s", fn_name)
            return llm

    generator_names = ("generate_answer", "generate", "run_llm", "ask_llm", "infer")
    for fn_name in generator_names:
        if hasattr(module, fn_name):
            fn = getattr(module, fn_name)
            logger.info("Loaded LLM via generator function: %s", fn_name)
            return FunctionLLMWrapper(fn)

    raise AttributeError(
        "No compatible local LLM entrypoint found in src.llm.llm_model.py. "
        "Expected one of: get_llm, build_llm, load_llm, create_llm, "
        "generate_answer, generate, run_llm, ask_llm, infer."
    )


def call_build_prompt(question: str, documents: List[Document]) -> Any:
    attempts = [
        lambda: build_prompt(question=question, documents=documents),
        lambda: build_prompt(query=question, documents=documents),
        lambda: build_prompt(question, documents),
    ]

    last_error = None
    for fn in attempts:
        try:
            return fn()
        except TypeError as exc:
            last_error = exc

    raise TypeError(
        f"Unable to call build_prompt with the current signature. Details: {last_error}"
    )


class LocalRAGPipeline:
    def __init__(
        self,
        top_k: int = 4,
        rebuild: bool = False,
    ):
        self.top_k = top_k
        self.rebuild = rebuild
        self.llm: Optional[Any] = None

        logger.info(
            "Initialized LocalRAGPipeline",
            extra={"top_k": self.top_k, "rebuild": self.rebuild, "llm_lazy_load": True},
        )

    def _get_llm(self):
        if self.llm is None:
            logger.info("Lazy loading the local LLM backend")
            self.llm = load_llm_backend()
        return self.llm

    def _invoke_llm(self, prompt: Any) -> str:
        llm = self._get_llm()
        with observe_duration(
            "llm.call",
            metric_name="llm.call.duration_ms",
            metric_attributes={
                "backend": "local",
            },
            span_attributes={
                "backend": "local",
                "prompt_length": len(str(prompt or "")),
            },
        ) as observation:
            if hasattr(llm, "invoke"):
                answer = extract_llm_text(llm.invoke(prompt))
                observation.set_metric_attribute("entrypoint", "invoke")
                observation.set_attribute("entrypoint", "invoke")
                observation.set_attribute("answer_length", len(answer or ""))
                return answer

            if callable(llm):
                answer = extract_llm_text(llm(prompt))
                observation.set_metric_attribute("entrypoint", "call")
                observation.set_attribute("entrypoint", "call")
                observation.set_attribute("answer_length", len(answer or ""))
                return answer

            raise TypeError(
                "The configured local LLM does not support invoke() and is not callable. "
                "Update src/llm/llm_model.py to expose a supported entrypoint."
            )

    def answer_from_documents(
        self,
        question: str,
        documents: List[Document],
        debug: bool = False,
    ) -> Dict[str, Any]:
        question = (question or "").strip()
        sources = build_sources(documents)

        logger.info(
            "Answer-from-documents started",
            extra={
                "event": "pipeline.answer_from_documents.started",
                "question_length": len(question),
                "documents_count": len(documents),
                "debug_requested": debug,
                "top_k": self.top_k,
            },
        )

        if not question:
            return {
                "answer": "Question is empty.",
                "sources": [],
                "debug": [],
                "stats": {
                    "retrieved_docs": 0,
                    "top_k_requested": self.top_k,
                    "llm_loaded": self.llm is not None,
                    "used_fallback": True,
                    "stage": "empty_question",
                },
            }

        if not documents:
            return {
                "answer": build_safe_fallback_answer(question),
                "sources": [],
                "debug": [],
                "stats": {
                    "retrieved_docs": 0,
                    "top_k_requested": self.top_k,
                    "llm_loaded": self.llm is not None,
                    "used_fallback": True,
                    "stage": "no_documents",
                },
            }

        try:
            prompt = call_build_prompt(question=question, documents=documents)
        except Exception:
            logger.exception(
                "Prompt construction failed",
                extra={
                    "event": "pipeline.answer_from_documents.prompt_failed",
                    "documents_count": len(documents),
                },
            )
            return {
                "answer": (
                    "I retrieved relevant evidence, but I could not build a final answer prompt "
                    "for this request."
                ),
                "sources": sources,
                "debug": build_debug_rows(documents) if debug else [],
                "stats": {
                    "retrieved_docs": len(documents),
                    "top_k_requested": self.top_k,
                    "llm_loaded": self.llm is not None,
                    "used_fallback": True,
                    "stage": "prompt_error",
                },
            }

        try:
            raw_answer = self._invoke_llm(prompt)
            answer = clean_answer(raw_answer)
        except Exception:
            logger.exception(
                "LLM answer generation failed",
                extra={
                    "event": "pipeline.answer_from_documents.llm_failed",
                    "documents_count": len(documents),
                },
            )
            return {
                "answer": "I found relevant evidence, but answer generation failed for this request.",
                "sources": sources,
                "debug": build_debug_rows(documents) if debug else [],
                "stats": {
                    "retrieved_docs": len(documents),
                    "top_k_requested": self.top_k,
                    "llm_loaded": self.llm is not None,
                    "used_fallback": True,
                    "stage": "llm_error",
                },
            }

        result = {
            "answer": answer,
            "sources": sources,
            "debug": build_debug_rows(documents) if debug else [],
            "stats": {
                "retrieved_docs": len(documents),
                "top_k_requested": self.top_k,
                "llm_loaded": self.llm is not None,
                "used_fallback": False,
                "stage": "ok",
            },
        }

        if debug:
            result["prompt"] = str(prompt)

        logger.info(
            "Answer-from-documents completed",
            extra={
                "event": "pipeline.answer_from_documents.completed",
                "documents_count": len(documents),
                "used_fallback": False,
                "answer_length": len(answer),
            },
        )
        return result

    def ask(self, question: str, debug: bool = False) -> Dict[str, Any]:
        question = (question or "").strip()

        logger.info(
            "Pipeline ask started",
            extra={
                "event": "pipeline.ask.started",
                "question_length": len(question),
                "debug_requested": debug,
                "top_k": self.top_k,
                "rebuild": self.rebuild,
            },
        )

        if not question:
            return {
                "answer": "Question is empty.",
                "sources": [],
                "debug": [],
                "stats": {
                    "retrieved_docs": 0,
                    "top_k_requested": self.top_k,
                    "llm_loaded": self.llm is not None,
                    "used_fallback": True,
                    "stage": "empty_question",
                },
            }

        try:
            documents = retrieve_documents(
                query=question,
                top_k=self.top_k,
                rebuild=self.rebuild,
            )
        except Exception:
            logger.exception(
                "Retriever failed while processing pipeline ask",
                extra={"event": "pipeline.ask.retrieval_failed"},
            )
            return {
                "answer": "I could not retrieve relevant documents for this request.",
                "sources": [],
                "debug": [],
                "stats": {
                    "retrieved_docs": 0,
                    "top_k_requested": self.top_k,
                    "llm_loaded": self.llm is not None,
                    "used_fallback": True,
                    "stage": "retrieval_error",
                },
            }

        result = self.answer_from_documents(
            question=question,
            documents=documents,
            debug=debug,
        )
        logger.info(
            "Pipeline ask completed",
            extra={
                "event": "pipeline.ask.completed",
                "retrieved_docs": len(documents),
                "used_fallback": bool(result.get("stats", {}).get("used_fallback")),
                "stage": result.get("stats", {}).get("stage", ""),
            },
        )
        return result


def build_pipeline(
    top_k: int = 4,
    rebuild: bool = False,
) -> LocalRAGPipeline:
    return LocalRAGPipeline(
        top_k=top_k,
        rebuild=rebuild,
    )


@lru_cache(maxsize=1)
def get_default_pipeline() -> LocalRAGPipeline:
    return build_pipeline(
        top_k=4,
        rebuild=False,
    )
