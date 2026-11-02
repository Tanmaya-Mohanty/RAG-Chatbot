from __future__ import annotations

import os
import re
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm = ChatGroq(
    model="openai/gpt-oss-120b",  # A popular, capable production model
    temperature=0,
)

# Two embedding objects:
#   - doc_embeddings   -> used when INDEXING chunks into FAISS
#   - query_embeddings -> used when SEARCHING FAISS at query time
# gemini-embedding-001 expects matching task_type for best accuracy.
doc_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="retrieval_document",
)
query_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="retrieval_query",
)

# -------------------
# 2. Per-thread stores (in memory only)
# -------------------
# NOTE: we now store the FAISS *vector store* (not a retriever), so we can
#       pass query_embeddings at search time.
_THREAD_SOURCE_DOCS: Dict[str, Dict[str, list[Document]]] = {}
_THREAD_VECTORSTORES: Dict[str, FAISS] = {}
_THREAD_METADATA: Dict[str, Dict[str, dict]] = {}
_THREAD_TITLES: Dict[str, str] = {}


def _get_vectorstore(thread_id: Optional[str]) -> Optional[FAISS]:
    if thread_id and str(thread_id) in _THREAD_VECTORSTORES:
        return _THREAD_VECTORSTORES[str(thread_id)]
    return None


# ---------- helpers ----------
def _youtube_id(url_or_id: str) -> Optional[str]:
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    patterns = [
        r"(?:youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    return None


def _split(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


def _rebuild_vectorstore(thread_id: str) -> None:
    tid = str(thread_id)
    sources = _THREAD_SOURCE_DOCS.get(tid, {})
    all_chunks = [c for chunks in sources.values() for c in chunks]
    if not all_chunks:
        _THREAD_VECTORSTORES.pop(tid, None)
        _THREAD_METADATA.pop(tid, None)
        _THREAD_SOURCE_DOCS.pop(tid, None)
        return
    # Build the FAISS store with DOC embeddings (retrieval_document).
    _THREAD_VECTORSTORES[tid] = FAISS.from_documents(all_chunks, doc_embeddings)


def _add_source(thread_id: str, source_key: str, chunks: list[Document], meta: dict) -> None:
    tid = str(thread_id)
    _THREAD_SOURCE_DOCS.setdefault(tid, {})[source_key] = chunks
    _THREAD_METADATA.setdefault(tid, {})[source_key] = meta
    _rebuild_vectorstore(tid)


def reset_thread_source(thread_id: str, source_key: str) -> None:
    tid = str(thread_id)
    if tid not in _THREAD_SOURCE_DOCS:
        return
    _THREAD_SOURCE_DOCS[tid].pop(source_key, None)
    _THREAD_METADATA.get(tid, {}).pop(source_key, None)
    _rebuild_vectorstore(tid)


def reset_thread(thread_id: str) -> None:
    tid = str(thread_id)
    _THREAD_SOURCE_DOCS.pop(tid, None)
    _THREAD_VECTORSTORES.pop(tid, None)
    _THREAD_METADATA.pop(tid, None)


def generate_thread_title(first_user_message: str, max_words: int = 6) -> str:
    """Ask the LLM for a short, human-friendly title based on the first prompt."""
    prompt = (
        "You generate concise chat titles. Given the user's first message below, "
        f"reply with a title of at most {max_words} words. "
        "Use Title Case, no quotes, no trailing punctuation, no 'Chat about' prefix. "
        "Reply with the title only.\n\n"
        f"User message:\n{first_user_message}"
    )
    try:
        response = llm.invoke(prompt)
        raw = (response.content or "").strip()
        raw = raw.strip("\"'` \n")
        raw = " ".join(raw.split())
        words = raw.split()
        if len(words) > max_words:
            raw = " ".join(words[:max_words])
        raw = raw[:60]
        return raw or first_user_message[:57]
    except Exception:
        text = " ".join(first_user_message.split())
        return (text[:57] + "…") if len(text) > 60 else text


# ---------- PDF ----------
def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        path = f.name
    try:
        docs = PyPDFLoader(path).load()
        chunks = _split(docs)
        fname = filename or os.path.basename(path)
        meta = {"type": "pdf", "filename": fname,
                "documents": len(docs), "chunks": len(chunks)}
        _add_source(thread_id, f"pdf:{fname}", chunks, meta)
        return meta
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ---------- Excel ----------
def ingest_excel(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    suffix = os.path.splitext(filename or "")[1].lower() or ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(file_bytes)
        path = f.name
    try:
        xls = pd.ExcelFile(path)
        docs: list[Document] = []
        total_rows = 0
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            if df.empty:
                continue
            df.columns = [str(c).strip() for c in df.columns]
            for idx, row in df.iterrows():
                pairs = [
                    f"{col}: {row[col]}"
                    for col in df.columns
                    if pd.notna(row[col]) and str(row[col]).strip() != ""
                ]
                if not pairs:
                    continue
                total_rows += 1
                docs.append(Document(
                    page_content=f"[Sheet: {sheet} | Row {idx + 1}] " + ", ".join(pairs),
                    metadata={"sheet": sheet, "row": int(idx) + 1},
                ))
            csv_dump = df.to_csv(index=False).strip()
            if csv_dump:
                docs.append(Document(
                    page_content=f"[Sheet: {sheet} | CSV dump]\n{csv_dump}",
                    metadata={"sheet": sheet, "kind": "sheet_summary"},
                ))
        if not docs:
            raise ValueError("Excel file had no non-empty rows.")
        chunks = _split(docs)
        fname = filename or os.path.basename(path)
        meta = {"type": "excel", "filename": fname,
                "sheets": xls.sheet_names, "rows": total_rows, "chunks": len(chunks)}
        _add_source(thread_id, f"excel:{fname}", chunks, meta)
        return meta
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ---------- YouTube ----------
def ingest_youtube(url_or_id: str, thread_id: str, title: Optional[str] = None) -> dict:
    vid = _youtube_id(url_or_id.strip())
    if not vid:
        raise ValueError("Could not parse a YouTube video ID from the input.")

    preferred_langs = ["en", "en-US", "en-GB", "hi", "es", "fr", "de", "pt", "ja", "zh"]
    text = ""
    used_lang = "unknown"
    last_error = None

    # Attempt 1: youtube-transcript-api with proxy
    try:
        # --- IMPORTANT: Replace this with your actual proxy if you have one ---
        # Example formats: "http://user:pass@ip:port", "socks5://ip:port"
        PROXY_URL = os.getenv("YOUTUBE_PROXY_URL", None) 
        
        ytt_api = YouTubeTranscriptApi(proxy_config=PROXY_URL) if PROXY_URL else YouTubeTranscriptApi()
        
        fetched = None
        try:
            fetched = ytt_api.fetch(vid, languages=preferred_langs)
            used_lang = getattr(fetched, "language_code", None)
        except Exception as e:
            last_error = e
            
        if fetched is None:
            try:
                listing = ytt_api.list(vid)
                candidates = list(listing)
                if not candidates:
                    raise ValueError("No transcripts are available for this video.")
                candidates.sort(key=lambda t: (not getattr(t, "is_generated", False)))
                chosen = candidates[0]
                fetched = chosen.fetch()
                used_lang = getattr(chosen, "language_code", None)
            except Exception as e:
                last_error = e
                
        if fetched is None:
            raise last_error or ValueError("No transcripts could be fetched.")
            
        try:
            raw = fetched.to_raw_data()
        except Exception:
            raw = list(fetched)

        text = " ".join(
            (seg.get("text") if isinstance(seg, dict) else seg.text) for seg in raw
        ).strip()
        
    except Exception as e:
        # If youtube-transcript-api fails (e.g., IP Blocked), try yt-dlp as a fallback
        last_error = e
        text = "" # reset text to trigger fallback below

    # Attempt 2: Fallback to yt-dlp (highly effective against IP blocks)
    if not text:
        try:
            import yt_dlp
            ydl_opts = {
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': preferred_langs,
                'quiet': True,
            }
            
            # Add proxy to yt-dlp if you have one
            PROXY_URL = os.getenv("YOUTUBE_PROXY_URL", None)
            if PROXY_URL:
                ydl_opts['proxy'] = PROXY_URL

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
                
                # Check for manual subtitles first, then auto-generated
                subs = info.get('subtitles', {})
                auto_subs = info.get('automatic_captions', {})
                
                chosen_lang = next((lang for lang in preferred_langs if lang in subs or lang in auto_subs), None)
                
                if chosen_lang:
                    used_lang = chosen_lang
                    sub_info = subs.get(chosen_lang, auto_subs.get(chosen_lang))
                    
                    # Find the json3 or vtt format
                    sub_url = next((f['url'] for f in sub_info if f['ext'] in ['json3', 'vtt']), None)
                    if sub_url:
                        import urllib.request
                        import json
                        
                        req = urllib.request.Request(sub_url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req) as response:
                            sub_data = response.read().decode('utf-8')
                            
                            if sub_url.endswith('json3'):
                                data = json.loads(sub_data)
                                text = " ".join(
                                    "".join(seg.get("utf8", "") for seg in event.get("segs", []))
                                    for event in data.get("events", [])
                                    if "segs" in event
                                )
                            else: # vtt format
                                import re
                                lines = sub_data.split('\n')
                                text = " ".join(
                                    line for line in lines 
                                    if line.strip() and not line.startswith(('WEBVTT', 'Kind:', 'Language:', 'NOTE')) 
                                    and '-->' not in line and not line.strip().isdigit()
                                ).replace('<c>', '').replace('</c>', '').replace('<v ', '').replace('</v>', '')
        except Exception as fallback_err:
            raise ValueError(f"Could not fetch transcript for {vid}. API Error: {last_error}. Fallback Error: {fallback_err}")

    text = " ".join(text.split()) # Clean up whitespace
    if not text:
        raise ValueError(f"Transcript for {vid} was empty.")

    docs = [Document(
        page_content=text,
        metadata={"video_id": vid, "url": f"https://youtu.be/{vid}", "language": used_lang},
    )]
    chunks = _split(docs)
    fname = title or f"YouTube {vid}"
    meta = {"type": "youtube", "filename": fname,
            "video_id": vid, "language": used_lang, "chunks": len(chunks)}
    _add_source(thread_id, f"youtube:{vid}", chunks, meta)
    return meta


# ---------- Dispatcher ----------
def ingest_document(kind: str, payload: Any, thread_id: str, filename: Optional[str] = None) -> dict:
    if kind == "pdf":
        return ingest_pdf(payload, thread_id, filename)
    if kind == "excel":
        return ingest_excel(payload, thread_id, filename)
    if kind == "youtube":
        return ingest_youtube(payload, thread_id, filename)
    raise ValueError(f"Unsupported kind: {kind}")


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers. Supported: add, sub, mul, div."""
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
        return {"first_num": first_num, "second_num": second_num,
                "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a symbol (e.g. 'AAPL') via Alpha Vantage."""
    url = ("https://www.alphavantage.co/query"
           f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM")
    return requests.get(url).json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """Retrieve relevant chunks from PDFs / Excel sheets / YouTube transcripts in this thread."""
    vectorstore = _get_vectorstore(thread_id)
    if vectorstore is None:
        return {"error": "No document indexed for this chat. Upload a file or add a YouTube link first.",
                "query": query}

    query_vector = query_embeddings.embed_query(query)
    results_with_scores = vectorstore.similarity_search_with_score_by_vector(query_vector, k=4)
    result = [doc for doc, _ in results_with_scores]

    def _has_content(doc: Document) -> bool:
        body = doc.page_content.split("]", 1)[-1].strip()
        return bool(body)

    result = [d for d in result if _has_content(d)]

    return {
        "query": query,
        "context": [d.page_content for d in result],
        "metadata": [d.metadata for d in result],
        "sources": list(_THREAD_METADATA.get(str(thread_id), {}).keys()),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)


# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    sources = list(_THREAD_METADATA.get(str(thread_id), {}).keys())
    sources_hint = (
        f"Indexed sources for this chat: {sources}. "
        if sources else "No document is indexed for this chat yet. "
    )

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about indexed content "
            "(PDFs, Excel sheets, YouTube transcripts), call `rag_tool` and include "
            f"the thread_id `{thread_id}`. {sources_hint}"
            "You can also use web search, stock price, and calculator tools when helpful. "
            "If no source is indexed and the user asks about a document, ask them to upload one. "
            "Never echo raw tool arguments back to the user; answer in natural language."
        )
    )
    response = llm_with_tools.invoke([system_message, *state["messages"]], config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer (in-memory)
# -------------------
checkpointer = MemorySaver()

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
# 8. Public helpers
# -------------------
def retrieve_all_threads() -> list[str]:
    return list(_THREAD_TITLES.keys())


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_VECTORSTORES


def thread_document_metadata(thread_id: str) -> dict:
    meta = _THREAD_METADATA.get(str(thread_id), {})
    if not meta:
        return {}
    latest_key = list(meta.keys())[-1]
    latest = dict(meta[latest_key])
    latest["all_sources"] = [m.get("filename") for m in meta.values()]
    return latest


def set_thread_title(thread_id: str, title: str) -> None:
    tid = str(thread_id)
    if tid and title:
        _THREAD_TITLES[tid] = title.strip()[:60]


def get_thread_title(thread_id: str) -> str:
    return _THREAD_TITLES.get(str(thread_id), str(thread_id))