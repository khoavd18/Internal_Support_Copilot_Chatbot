from __future__ import annotations

import uuid

import requests
import streamlit as st

DEFAULT_API_BASE = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Internal Support Copilot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Internal Support Copilot")
st.caption("UI riêng cho local/offline RAG chatbot và agent v1")


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def call_health(api_base: str):
    response = requests.get(f"{api_base.rstrip('/')}/health", timeout=15)
    response.raise_for_status()
    return response.json()


def call_backend(
    api_base: str,
    question: str,
    debug: bool,
    top_k: int | None,
    backend_mode: str,
    agent_mode: str,
    session_id: str,
):
    payload = {
        "question": question,
        "debug": debug,
        "session_id": session_id,
    }

    if top_k is not None:
        payload["top_k"] = top_k

    if backend_mode == "agent":
        payload["mode"] = agent_mode
        endpoint = "/agent/ask"
    else:
        endpoint = "/ask"

    response = requests.post(
        f"{api_base.rstrip('/')}{endpoint}",
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def render_sources(sources):
    if not sources:
        st.write("Không có source.")
        return

    for item in sources:
        st.markdown(f"**[{item['index']}] {item['title']}**")
        st.caption(
            f"source={item.get('source', '')} | "
            f"doc_id={item.get('doc_id', '')} | "
            f"rerank_score={item.get('rerank_score', '')}"
        )


def render_debug(debug_rows):
    if not debug_rows:
        st.write("Không có debug retrieval.")
        return

    st.dataframe(debug_rows, use_container_width=True)


def render_agent_meta(agent_meta):
    if not agent_meta:
        st.write("Không có metadata agent.")
        return

    st.json(agent_meta)


with st.sidebar:
    st.subheader("Cấu hình kết nối")

    api_base = st.text_input(
        "API Base URL",
        value=DEFAULT_API_BASE,
    )

    st.caption(f"session_id: {st.session_state.session_id}")

    backend_mode = st.selectbox(
        "Chế độ backend",
        options=["agent", "rag"],
        index=0,
        help="agent = agent v1 có route; rag = pipeline cũ",
    )

    agent_mode = st.selectbox(
        "Agent mode",
        options=["auto", "answer", "search"],
        index=0,
        help="auto = agent tự chọn, answer = ép trả lời, search = ép chỉ lấy sources",
    )

    debug_mode = st.checkbox(
        "Hiện debug retrieval",
        value=True,
    )

    top_k = st.slider(
        "top_k",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
    )

    st.subheader("Kiểm tra API")
    if st.button("Check /health"):
        try:
            health = call_health(api_base)
            st.success("API đang hoạt động")
            st.json(health)
        except Exception as exc:
            st.error(f"Không kết nối được API: {exc}")

    if st.button("Xóa lịch sử chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Chào bạn, hãy hỏi tôi về GitHub Docs, GitLab Handbook hoặc GitHub Issues.",
            "sources": [],
            "debug": [],
            "agent": None,
        }
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            with st.expander("Nguồn đã dùng", expanded=False):
                render_sources(message.get("sources", []))

            if debug_mode:
                with st.expander("Debug retrieval", expanded=False):
                    render_debug(message.get("debug", []))

            if message.get("agent"):
                with st.expander("Agent metadata", expanded=False):
                    render_agent_meta(message.get("agent"))


user_query = st.chat_input("Ví dụ: Làm thế nào để đăng nhập bằng passkey?")

if user_query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
            "sources": [],
            "debug": [],
            "agent": None,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Đang hỏi backend..."):
                result = call_backend(
                    api_base=api_base,
                    question=user_query,
                    debug=debug_mode,
                    top_k=top_k,
                    backend_mode=backend_mode,
                    agent_mode=agent_mode,
                    session_id=st.session_state.session_id,
                )

            answer = result.get("answer", "Không có câu trả lời.")
            sources = result.get("sources", [])
            debug_rows = result.get("debug", [])
            agent_meta = result.get("agent")
            stats = result.get("stats", {})

            st.markdown(answer)

            if backend_mode == "agent":
                effective_question = stats.get("effective_question")
                if effective_question:
                    st.caption(f"effective_question: {effective_question}")

                if stats.get("guardrail_action") and stats.get("guardrail_action") != "allow_answer":
                    st.caption(
                        f"guardrail: {stats.get('guardrail_action')} | reason: {stats.get('guardrail_reason', '')}"
                    )

            with st.expander("Nguồn đã dùng", expanded=False):
                render_sources(sources)

            if debug_mode:
                with st.expander("Debug retrieval", expanded=False):
                    render_debug(debug_rows)

            if agent_meta:
                with st.expander("Agent metadata", expanded=False):
                    render_agent_meta(agent_meta)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "debug": debug_rows,
                    "agent": agent_meta,
                }
            )

        except requests.HTTPError as exc:
            error_text = f"Lỗi HTTP từ backend: {exc}"
            try:
                error_payload = exc.response.json()
                error_text += f"\n\nChi tiết: {error_payload}"
            except Exception:
                pass

            st.error(error_text)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_text,
                    "sources": [],
                    "debug": [],
                    "agent": None,
                }
            )

        except Exception as exc:
            error_text = f"Lỗi khi gọi backend: {exc}"
            st.error(error_text)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_text,
                    "sources": [],
                    "debug": [],
                    "agent": None,
                }
            )