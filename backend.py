from __future__ import annotations

import os
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
primary_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", timeout=5)

# Local fallback LLM
fallback_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# -------------------
# 2. PDF retriever store (per thread, in-memory)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

# -------------------
# 3. Tools
# -------------------
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div"""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage."""
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    try:
        r = requests.get(url, timeout=5)
        return r.json()
    except Exception as e:
        return {"error": f"Network offline or API error: {e}"}

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool."""
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

tools = [get_stock_price, calculator, rag_tool]

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call (with fallback)."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the stock price and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]

    # Try Primary (Gemini)
    try:
        bound_primary = primary_llm.bind_tools(tools)
        response = bound_primary.invoke(messages, config=config)
    except Exception as e:
        print(f"[fallback] Primary LLM failed ({e}); switching to local Ollama.")
        # Fallback to Ollama
        bound_fallback = fallback_llm.bind_tools(tools)
        response = bound_fallback.invoke(messages)

    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 6. In-memory checkpointer (threading, no DB)
# -------------------
checkpointer = InMemorySaver()

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def generate_chat_title(first_message: str) -> str:
    """Generate a concise title for the chat from the first user query."""
    prompt = [
        SystemMessage(
            content=(
                "Generate a very short title (max 5 words, no quotes, no period) "
                "that summarizes the user's first message. Return only the title."
            )
        ),
        HumanMessage(content=first_message),
    ]
    
    # Try Gemini for title
    try:
        title = primary_llm.invoke(prompt).content
    except Exception as e:
        print(f"[fallback] Title generation failed with primary ({e}); using local Ollama.")
        try:
            title = fallback_llm.invoke(prompt).content
        except Exception as fallback_e:
            print(f"[fallback] Title generation failed with Ollama: {fallback_e}")
            title = first_message[:40]
            
    if isinstance(title, list):
        title = " ".join(str(t) for t in title)
    title = str(title).strip().strip('"').strip("'")
    return title[:60] if title else "New Chat"

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})