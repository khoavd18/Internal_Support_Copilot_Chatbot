from __future__ import annotations

import traceback
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.agent.service import get_agent
from src.core.schema import (
    AgentAskRequest,
    AgentResponse,
    AskRequest,
    AskResponse,
)
from src.pipeline import get_default_pipeline, build_pipeline

app = FastAPI(
    title="Internal Support Copilot API",
    version="1.2.0",
    description="Local RAG API cho GitHub Docs, GitLab Handbook, GitHub Issues và agent v1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_health_payload() -> Dict[str, Any]:
    pipeline_cached = get_default_pipeline.cache_info().currsize > 0

    payload = {
        "status": "ok",
        "service": "internal-support-copilot-api",
        "pipeline_cached": pipeline_cached,
        "default_top_k": 4,
        "agent_ready": True,
    }

    if pipeline_cached:
        try:
            pipeline = get_default_pipeline()
            payload["llm_loaded"] = pipeline.llm is not None
        except Exception:
            payload["llm_loaded"] = False
    else:
        payload["llm_loaded"] = False

    return payload


@app.get("/health")
def health_check():
    return _build_health_payload()


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question không được rỗng")

    try:
        if payload.top_k is not None:
            pipeline = build_pipeline(
                top_k=payload.top_k,
                rebuild=False,
            )
        else:
            pipeline = get_default_pipeline()

        result = pipeline.ask(question, debug=payload.debug)

        return AskResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            stats=result.get("stats", {}),
            debug=result.get("debug", []) if payload.debug else [],
        )

    except HTTPException:
        raise
    except Exception as exc:
        print("\n" + "=" * 100)
        print("[ERROR] /ask failed")
        traceback.print_exc()
        print("=" * 100 + "\n")

        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý câu hỏi: {exc}",
        )


@app.post("/agent/ask", response_model=AgentResponse)
def ask_agent(payload: AgentAskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question không được rỗng")

    try:
        agent = get_agent()
        result = agent.ask(
        question=question,
        debug=payload.debug,
        top_k=payload.top_k,
        mode=payload.mode,
        session_id=payload.session_id,
    )
        return AgentResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        print("\n" + "=" * 100)
        print("[ERROR] /agent/ask failed")
        traceback.print_exc()
        print("=" * 100 + "\n")

        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý câu hỏi qua agent: {exc}",
        )