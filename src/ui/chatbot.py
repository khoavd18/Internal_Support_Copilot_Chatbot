from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

DEFAULT_API_BASE = os.getenv(
    "INTERNAL_SUPPORT_API_BASE_URL",
    "http://127.0.0.1:8000",
).strip()
DEFAULT_USER_ID = os.getenv("INTERNAL_SUPPORT_UI_USER_ID", "").strip()
DEFAULT_USER_ROLE = os.getenv("INTERNAL_SUPPORT_UI_USER_ROLE", "viewer").strip().lower()

st.set_page_config(
    page_title="Internal Support Copilot",
    layout="wide",
)

st.title("Internal Support Copilot")
st.caption("Local RAG, agent routing, and multi-agent orchestration in a single operator UI.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def _build_headers(user_id: str, user_role: str) -> dict[str, str]:
    headers: dict[str, str] = {}

    normalized_user_id = str(user_id or "").strip()
    normalized_role = str(user_role or "").strip().lower()

    if normalized_user_id:
        headers["X-User-ID"] = normalized_user_id

    if normalized_role:
        headers["X-User-Role"] = normalized_role

    return headers


def call_health(api_base: str, user_id: str, user_role: str):
    response = requests.get(
        f"{api_base.rstrip('/')}/health",
        headers=_build_headers(user_id, user_role),
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def call_backend(
    api_base: str,
    question: str,
    debug: bool,
    top_k: int | None,
    confirmed: bool,
    backend_mode: str,
    agent_mode: str,
    session_id: str,
    user_id: str,
    user_role: str,
):
    payload = {
        "question": question,
        "debug": debug,
        "session_id": session_id,
        "confirmed": confirmed,
    }

    if top_k is not None:
        payload["top_k"] = top_k

    if backend_mode == "multi_agent":
        payload["mode"] = agent_mode
        endpoint = "/multi-agent/ask"
    elif backend_mode == "agent":
        payload["mode"] = agent_mode
        endpoint = "/agent/ask"
    else:
        endpoint = "/ask"

    response = requests.post(
        f"{api_base.rstrip('/')}{endpoint}",
        json=payload,
        headers=_build_headers(user_id, user_role),
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def render_sources(sources):
    if not sources:
        st.write("No sources available.")
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
        st.write("No retrieval debug data available.")
        return

    st.dataframe(debug_rows, use_container_width=True)



def render_agent_meta(agent_meta):
    if not agent_meta:
        st.write("No agent metadata available.")
        return

    st.json(agent_meta)


with st.sidebar:
    st.subheader("Connection")

    api_base = st.text_input(
        "API Base URL",
        value=DEFAULT_API_BASE,
    )

    st.caption(f"session_id: {st.session_state.session_id}")

    st.subheader("Auth")

    user_id = st.text_input(
        "User ID",
        value=DEFAULT_USER_ID,
        help="Sent as X-User-ID. Required for operator write actions.",
    )

    role_options = ["viewer", "operator"]
    default_role = DEFAULT_USER_ROLE if DEFAULT_USER_ROLE in role_options else "viewer"
    user_role = st.selectbox(
        "User role",
        options=role_options,
        index=role_options.index(default_role),
        help="Sent as X-User-Role. Use operator for write-capable actions.",
    )

    if user_role == "operator" and not user_id.strip():
        st.warning("Operator mode needs a User ID, or the backend will reject write actions.")

    backend_mode = st.selectbox(
        "Backend mode",
        options=["multi_agent", "agent", "rag"],
        index=0,
        help="multi_agent = LangGraph supervisor; agent = single-agent routing; rag = base pipeline",
    )

    agent_mode = st.selectbox(
        "Agent mode",
        options=["auto", "answer", "search"],
        index=0,
        help="auto = agent decides, answer = force synthesis, search = return sources only",
    )

    debug_mode = st.checkbox(
        "Show retrieval debug",
        value=True,
    )

    confirm_write_actions = st.checkbox(
        "Confirm write actions",
        value=False,
        help="Send confirmed=true for write actions such as create issue, create repo, or commit.",
    )

    top_k = st.slider(
        "top_k",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
    )

    st.subheader("Health Check")
    if st.button("Check /health"):
        try:
            health = call_health(api_base, user_id, user_role)
            st.success("API is reachable")
            st.json(health)
        except Exception as exc:
            st.error(f"Failed to connect to the API: {exc}")

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ask about GitHub Docs, GitLab Handbook, or GitHub Issues. "
                "You can inspect sources, debug retrieval, compare backend modes, "
                "and propose write actions like create issue/create repo with confirmation."
            ),
            "sources": [],
            "debug": [],
            "agent": None,
        }
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            with st.expander("Sources", expanded=False):
                render_sources(message.get("sources", []))

            if debug_mode:
                with st.expander("Retrieval Debug", expanded=False):
                    render_debug(message.get("debug", []))

            if message.get("agent"):
                with st.expander("Agent Metadata", expanded=False):
                    render_agent_meta(message.get("agent"))


user_query = st.chat_input("Example: How do I sign in with a passkey?")

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
            with st.spinner("Querying the backend..."):
                result = call_backend(
                    api_base=api_base,
                    question=user_query,
                    debug=debug_mode,
                    top_k=top_k,
                    confirmed=confirm_write_actions,
                    backend_mode=backend_mode,
                    agent_mode=agent_mode,
                    session_id=st.session_state.session_id,
                    user_id=user_id,
                    user_role=user_role,
                )

            answer = result.get("answer", "No answer returned.")
            sources = result.get("sources", [])
            debug_rows = result.get("debug", [])
            agent_meta = result.get("agent")
            stats = result.get("stats", {})

            st.markdown(answer)

            if backend_mode in ["multi_agent", "agent"]:
                effective_question = stats.get("effective_question")
                if effective_question:
                    st.caption(f"effective_question: {effective_question}")

                if stats.get("guardrail_action") and stats.get("guardrail_action") != "allow_answer":
                    st.caption(
                        f"guardrail: {stats.get('guardrail_action')} | "
                        f"reason: {stats.get('guardrail_reason', '')}"
                    )

                selected_agents = stats.get("selected_agents")
                if selected_agents:
                    st.caption(f"selected_agents: {selected_agents}")

            with st.expander("Sources", expanded=False):
                render_sources(sources)

            if debug_mode:
                with st.expander("Retrieval Debug", expanded=False):
                    render_debug(debug_rows)

            if agent_meta:
                with st.expander("Agent Metadata", expanded=False):
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
            error_text = f"HTTP error from backend: {exc}"
            try:
                error_payload = exc.response.json()
                error_text += f"\n\nDetails: {error_payload}"
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
            error_text = f"Unexpected error while calling backend: {exc}"
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
