import json
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend import (
    chatbot,
    generate_thread_title,
    get_thread_title,
    ingest_document,
    reset_thread_source,
    retrieve_all_threads,
    set_thread_title,
    thread_document_metadata,
)


# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def add_thread(thread_id):
    tid = str(thread_id)
    if tid not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(tid)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["thread_titled"] = False


def auto_title_from_first_message(text: str) -> str:
    text = " ".join(text.split())
    return (text[:57] + "…") if len(text) > 60 else text


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    return state.values.get("messages", [])


# =========================== Message hygiene ===========================
# These helpers are used BOTH when streaming live responses AND when
# reloading a thread from the checkpointer, so the two views always agree.

_TOOL_PAYLOAD_KEYS = ("query", "context", "first_num", "second_num",
                      "operation", "symbol", "thread_id")


def _looks_like_tool_payload(text: str) -> bool:
    """True if `text` is just a JSON dump of tool-call arguments."""
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not (t.startswith("{") and t.endswith("}")):
        return False
    try:
        obj = json.loads(t)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    return any(k in obj for k in _TOOL_PAYLOAD_KEYS)


def _extract_text_parts(content) -> str:
    """Return only human-readable text parts from a message content field."""
    if content is None:
        return ""
    if isinstance(content, str):
        return "" if _looks_like_tool_payload(content) else content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                if not _looks_like_tool_payload(part):
                    parts.append(part)
            elif isinstance(part, dict):
                # Keep only true text parts; drop tool_use / function_call / etc.
                if part.get("type") == "text":
                    txt = part.get("text")
                    if isinstance(txt, str) and not _looks_like_tool_payload(txt):
                        parts.append(txt)
                elif isinstance(part.get("text"), str) and "type" not in part:
                    # Some providers return {"text": "..."} without "type"
                    if not _looks_like_tool_payload(part["text"]):
                        parts.append(part["text"])
        return "".join(parts)
    return str(content) if not _looks_like_tool_payload(str(content)) else ""


def _is_tool_call_message(msg) -> bool:
    """True if this AIMessage is only a tool-call request (not a real answer)."""
    if not isinstance(msg, AIMessage):
        return False
    if getattr(msg, "tool_calls", None):
        return True
    if isinstance(msg.content, list):
        for part in msg.content:
            if isinstance(part, dict) and part.get("type") in ("tool_use", "function_call"):
                return True
    if isinstance(msg.content, str) and _looks_like_tool_payload(msg.content):
        return True
    return False


def _clean_assistant_message(msg) -> str:
    """Return the *user-visible* text of an assistant message, or '' if it's
    a tool-call request / JSON payload / empty."""
    if not isinstance(msg, AIMessage):
        return ""
    if _is_tool_call_message(msg):
        return ""
    return _extract_text_parts(msg.content).strip()


def _history_from_messages(messages) -> list[dict]:
    """Convert checkpointer messages into clean {role, content} dicts."""
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = _extract_text_parts(msg.content).strip()
            if text:
                history.append({"role": "user", "content": text})
        elif isinstance(msg, AIMessage):
            text = _clean_assistant_message(msg)
            if text:
                history.append({"role": "assistant", "content": text})
        # ToolMessages / SystemMessages are intentionally dropped from the UI
    return history


def switch_thread(thread_id):
    st.session_state["thread_id"] = thread_id
    messages = load_conversation(thread_id)
    st.session_state["message_history"] = _history_from_messages(messages)
    st.session_state["ingested_docs"].setdefault(str(thread_id), {})
    st.session_state["thread_titled"] = str(thread_id) in st.session_state["chat_threads"]
    st.rerun()


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "thread_titled" not in st.session_state:
    st.session_state["thread_titled"] = False

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = list(st.session_state["chat_threads"])[::-1]
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("LangGraph Multi-Source Chatbot")
st.sidebar.markdown(f"**Thread:** `{get_thread_title(thread_key)}`")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.subheader("📎 Add sources to this chat")
source_type = st.sidebar.radio(
    "Source type", ["PDF", "Excel", "YouTube"], horizontal=True, key="source_type_radio"
)

# ---------- PDF ----------
if source_type == "PDF":
    uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"], key="pdf_uploader")
    if uploaded_pdf:
        key = f"pdf:{uploaded_pdf.name}"
        if key in thread_docs:
            if st.sidebar.button(f"♻️ Re-index `{uploaded_pdf.name}`", key="reindex_pdf"):
                reset_thread_source(thread_key, key)
                thread_docs.pop(key, None)
                with st.sidebar.status("Re-indexing PDF…", expanded=True) as sb:
                    summary = ingest_document(
                        "pdf", uploaded_pdf.getvalue(),
                        thread_id=thread_key, filename=uploaded_pdf.name,
                    )
                    thread_docs[key] = summary
                    sb.update(label="✅ Re-indexed", state="complete", expanded=False)
                st.rerun()
            else:
                st.sidebar.info(f"`{uploaded_pdf.name}` already indexed. Click Re-index to refresh.")
        else:
            with st.sidebar.status("Indexing PDF…", expanded=True) as sb:
                summary = ingest_document(
                    "pdf", uploaded_pdf.getvalue(),
                    thread_id=thread_key, filename=uploaded_pdf.name,
                )
                thread_docs[key] = summary
                sb.update(label="✅ PDF indexed", state="complete", expanded=False)

# ---------- Excel ----------
elif source_type == "Excel":
    uploaded_xls = st.sidebar.file_uploader(
        "Upload an Excel file", type=["xlsx", "xls"], key="excel_uploader"
    )
    if uploaded_xls:
        key = f"excel:{uploaded_xls.name}"
        if key in thread_docs:
            if st.sidebar.button(f"♻️ Re-index `{uploaded_xls.name}`", key="reindex_excel"):
                reset_thread_source(thread_key, key)
                thread_docs.pop(key, None)
                with st.sidebar.status("Re-indexing Excel…", expanded=True) as sb:
                    summary = ingest_document(
                        "excel", uploaded_xls.getvalue(),
                        thread_id=thread_key, filename=uploaded_xls.name,
                    )
                    thread_docs[key] = summary
                    sb.update(label="✅ Re-indexed", state="complete", expanded=False)
                st.rerun()
            else:
                st.sidebar.info(f"`{uploaded_xls.name}` already indexed. Click Re-index to refresh.")
        else:
            with st.sidebar.status("Indexing Excel…", expanded=True) as sb:
                summary = ingest_document(
                    "excel", uploaded_xls.getvalue(),
                    thread_id=thread_key, filename=uploaded_xls.name,
                )
                thread_docs[key] = summary
                sb.update(label="✅ Excel indexed", state="complete", expanded=False)

# ---------- YouTube ----------
else:
    yt_url = st.sidebar.text_input("YouTube URL or video ID", key="yt_input")
    if st.sidebar.button("Index YouTube video", key="yt_index_btn") and yt_url.strip():
        key = f"youtube:{yt_url.strip()}"
        if key in thread_docs:
            st.sidebar.info("That YouTube video is already indexed for this chat.")
        else:
            with st.sidebar.status("Fetching transcript…", expanded=True) as sb:
                try:
                    summary = ingest_document("youtube", yt_url.strip(), thread_id=thread_key)
                    thread_docs[key] = summary
                    sb.update(label="✅ Transcript indexed", state="complete", expanded=False)
                except Exception as e:
                    sb.update(label=f"❌ Failed: {e}", state="error", expanded=True)

# ---------- Source summary ----------
if thread_docs:
    st.sidebar.success(f"**{len(thread_docs)} source(s) indexed in this chat:**")
    for k, v in thread_docs.items():
        st.sidebar.caption(f"• `{v.get('filename')}` ({v.get('chunks', '?')} chunks)")
else:
    st.sidebar.info("No source indexed yet.")

# ---------- Past conversations ----------
st.sidebar.subheader("🗂 Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for tid in threads:
        label = get_thread_title(tid)
        if st.sidebar.button(label, key=f"side-thread-{tid}"):
            selected_thread = tid

# ============================ Main Layout ========================
st.title("Multi-Source Chatbot")
st.caption(f"Chatting in: **{get_thread_title(thread_key)}**")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask about your PDF / Excel / YouTube content or use tools")

if user_input:
    if not st.session_state["thread_titled"]:
        with st.spinner("Naming chat…"):
            new_title = generate_thread_title(user_input)
        set_thread_title(thread_key, new_title)
        st.session_state["thread_titled"] = True

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running", expanded=True,
                        )
                    continue

                if isinstance(message_chunk, AIMessage):
                    # Skip pure tool-call requests — no JSON in the UI
                    if _is_tool_call_message(message_chunk):
                        continue
                    cleaned = _extract_text_parts(message_chunk.content)
                    if cleaned:
                        yield cleaned

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save the cleaned final text (never the raw payload)
    final_text = _extract_text_parts(ai_message).strip() if ai_message else ""
    if final_text:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": final_text}
        )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Latest source: {doc_meta.get('filename')} — "
            f"all sources: {', '.join(doc_meta.get('all_sources', []))}"
        )

st.divider()

if selected_thread:
    switch_thread(selected_thread)