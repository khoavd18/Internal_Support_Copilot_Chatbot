from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.rag.generation.answer_postprocess import (
    build_debug_rows,
    build_safe_fallback_answer,
    build_sources,
    clean_answer,
    extract_llm_text,
)
from src.rag.generation.prompt_builder import build_prompt
from src.rag.retrieval.retriever import retrieve_documents


class FunctionLLMWrapper:
    """
    Bọc một hàm generate(prompt) thành object có invoke(prompt)
    để pipeline dùng thống nhất.
    """

    def __init__(self, fn):
        self.fn = fn

    def invoke(self, prompt: Any) -> Any:
        return self.fn(prompt)

    def __call__(self, prompt: Any) -> Any:
        return self.fn(prompt)


@lru_cache(maxsize=1)
def load_llm_backend():
    """
    Tự dò local LLM trong src.base.llm_model.py

    Hỗ trợ 2 kiểu:
    1) object loader: get_llm / build_llm / load_llm / create_llm
    2) generator function: generate_answer / generate / run_llm / ask_llm / infer
    """
    module = importlib.import_module("src.llm.llm_model")

    object_loader_names = ("get_llm", "build_llm", "load_llm", "create_llm")
    for fn_name in object_loader_names:
        if hasattr(module, fn_name):
            llm = getattr(module, fn_name)()
            print(f"[INFO] Loaded LLM via object loader: {fn_name}")
            return llm

    generator_names = ("generate_answer", "generate", "run_llm", "ask_llm", "infer")
    for fn_name in generator_names:
        if hasattr(module, fn_name):
            fn = getattr(module, fn_name)
            print(f"[INFO] Loaded LLM via generator function: {fn_name}")
            return FunctionLLMWrapper(fn)

    raise AttributeError(
        "Không tìm thấy local LLM phù hợp trong src.llm.llm_model.py. "
        "Hãy có một trong các hàm sau:\n"
        "- object loader: get_llm / build_llm / load_llm / create_llm\n"
        "- hoặc generator: generate_answer / generate / run_llm / ask_llm / infer"
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
        f"Không gọi được build_prompt với signature hiện tại. Chi tiết: {last_error}"
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

        print("=" * 80)
        print("[INFO] LocalRAGPipeline initialized")
        print(f"[INFO] top_k   = {self.top_k}")
        print(f"[INFO] rebuild = {self.rebuild}")
        print("[INFO] LLM lazy load = enabled")
        print("=" * 80)

    def _get_llm(self):
        if self.llm is None:
            print("[INFO] Lazy loading LLM now...")
            self.llm = load_llm_backend()
        return self.llm

    def _invoke_llm(self, prompt: Any) -> str:
        llm = self._get_llm()

        if hasattr(llm, "invoke"):
            return extract_llm_text(llm.invoke(prompt))

        if callable(llm):
            return extract_llm_text(llm(prompt))

        raise TypeError(
            "Local LLM hiện tại không hỗ trợ invoke() và cũng không callable(). "
            "Hãy map lại trong src/llm/llm_model.py"
        )

    def ask(self, question: str, debug: bool = False) -> Dict[str, Any]:
        question = question.strip()

        if not question:
            return {
                "answer": "Câu hỏi đang rỗng.",
                "sources": [],
                "debug": [],
                "stats": {
                    "retrieved_docs": 0,
                    "top_k_requested": self.top_k,
                    "llm_loaded": self.llm is not None,
                    "used_fallback": True,
                },
            }

        try:
            documents = retrieve_documents(
                query=question,
                top_k=self.top_k,
                rebuild=self.rebuild,
            )
        except Exception as exc:
            return {
                "answer": (
                    "Retriever đang lỗi khi lấy tài liệu liên quan.\n"
                    f"Lỗi: {exc}"
                ),
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

        sources = build_sources(documents)

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
        except Exception as exc:
            return {
                "answer": (
                    "Đã retrieve được tài liệu nhưng build prompt bị lỗi.\n"
                    f"Lỗi: {exc}"
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
        except Exception as exc:
            return {
                "answer": (
                    "Tôi đã tìm được tài liệu liên quan nhưng local LLM đang lỗi khi sinh câu trả lời.\n"
                    f"Lỗi: {exc}"
                ),
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