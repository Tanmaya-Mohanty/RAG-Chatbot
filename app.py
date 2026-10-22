import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend import (
    chatbot,
    generate_chat_title,
    ingest_pdf,
    thread_document_metadata,
)


# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["titles"][thread_id] = "New Chat"


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def extract_text(content) -> str:
    """Helper to extract plain text from various LangChain/Gemini content formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    return str(content)


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "titles" not in st.session_state:
    st.session_state["titles"] = {}

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])
st.session_state["titles"].setdefault(st.session_state["thread_id"], "New Chat")

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(
    f"**Chat:** {st.session_state['titles'].get(thread_key, 'New Chat')}"
)

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        label = st.session_state["titles"].get(thread_id, "New Chat")
        if st.sidebar.button(label, key=f"side-thread-{thread_id}"):
            selected_thread = thread_id

# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")

# Chat area (render history safely)
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        # Ensure history is rendered as a string, not a raw dict/list
        st.text(extract_text(message["content"]))

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    is_first_message = len(st.session_state["message_history"]) == 0

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    if is_first_message:
        with st.spinner("Naming chat…"):
            st.session_state["titles"][thread_key] = generate_chat_title(user_input)

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
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    # FIX: Extract just the text string instead of yielding the whole list
                    text = extract_text(message_chunk.content)
                    if text:
                        yield text

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": extract_text(msg.content)})
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()