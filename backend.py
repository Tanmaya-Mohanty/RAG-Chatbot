from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from typing import Annotated, Any, Dict, Optional, TypedDict
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
# timeout / max_retries make transient upstream 502/503/429 errors from Groq
# retry automatically instead of crashing the app.
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    temperature=0,
    max_retries=3,
    timeout=60,
)

# Embeddings are loaded lazily: importing torch + the model at import time makes
# startup slow / memory-heavy, which is a classic cause of 502 on hosted platforms
# (the health check fails before Streamlit is ready, or the container is OOM-killed).
_embeddings: Optional[HuggingFaceEmbeddings] = None
_emb_lock = threading.Lock()


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    with _emb_lock:
        if _embeddings is None:
            _embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"batch_size": 32},
            )
        return _embeddings


# -------------------
# 2. Per-thread stores (in memory only)
# -------------------
_LOCK = threading.RLock()
_THREAD_SOURCE_DOCS: Dict[str, Dict[str, list[Document]]] = {}
_THREAD_STORES: Dict[str, FAISS] = {}
_THREAD_METADATA: Dict[str, Dict[str, dict]] = {}
_THREAD_TITLES: Dict[str, str] = {}

MAX_EXCEL_ROWS_PER_SHEET = int(os.getenv("MAX_EXCEL_ROWS_PER_SHEET", "5000"))


def _get_retriever(thread_id: Optional[str]):
    if not thread_id:
        return None
    store = _THREAD_STORES.get(str(thread_id))
    if store is None:
        return None
    return store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def _split(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


def _rebuild_store(tid: str) -> None:
    sources = _THREAD_SOURCE_DOCS.get(tid, {})
    all_chunks = [c for chunks in sources.values() for c in chunks]
    if not all_chunks:
        _THREAD_STORES.pop(tid, None)
        _THREAD_METADATA.pop(tid, None)
        _THREAD_SOURCE_DOCS.pop(tid, None)
        return
    _THREAD_STORES[tid] = FAISS.from_documents(all_chunks, get_embeddings())


def _add_source(thread_id: str, source_key: str, chunks: list[Document], meta: dict) -> None:
    if not chunks:
        raise ValueError("No extractable text found in this source.")
    tid = str(thread_id)
    with _LOCK:
        sources = _THREAD_SOURCE_DOCS.setdefault(tid, {})
        replacing = source_key in sources
        sources[source_key] = chunks
        _THREAD_METADATA.setdefault(tid, {})[source_key] = meta
        if replacing or tid not in _THREAD_STORES:
            _rebuild_store(tid)
        else:
            # Incremental: only embed the NEW chunks instead of re-embedding everything.
            _THREAD_STORES[tid].add_documents(chunks)


def reset_thread_source(thread_id: str, source_key: str) -> None:
    tid = str(thread_id)
    with _LOCK:
        if tid not in _THREAD_SOURCE_DOCS:
            return
        _THREAD_SOURCE_DOCS[tid].pop(source_key, None)
        _THREAD_METADATA.get(tid, {}).pop(source_key, None)
        _rebuild_store(tid)


def reset_thread(thread_id: str) -> None:
    tid = str(thread_id)
    with _LOCK:
        _THREAD_SOURCE_DOCS.pop(tid, None)
        _THREAD_STORES.pop(tid, None)
        _THREAD_METADATA.pop(tid, None)


def generate_thread_title(first_user_message: str, max_words: int = 6) -> str:
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
        docs = [d for d in docs if d.page_content.strip()]
        if not docs:
            raise ValueError("No text found in the PDF (it may be a scanned image; OCR is needed).")
        chunks = _split(docs)
        fname = filename or os.path.basename(path)
        meta = {"type": "pdf", "filename": fname, "documents": len(docs), "chunks": len(chunks)}
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
            df = df.head(MAX_EXCEL_ROWS_PER_SHEET)  # keep embedding time bounded
            df.columns = [str(c).strip() for c in df.columns]
            df = df.loc[:, ~pd.Index(df.columns).duplicated()]
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
            # A full CSV dump is only useful (and cheap) for small sheets.
            if len(df) <= 200:
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
        meta = {"type": "excel", "filename": fname, "sheets": xls.sheet_names,
                "rows": total_rows, "chunks": len(chunks)}
        _add_source(thread_id, f"excel:{fname}", chunks, meta)
        return meta
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ---------- YouTube ----------
PREFERRED_LANGS = ["en", "en-US", "en-GB", "hi", "es", "fr", "de", "pt", "ja", "zh"]

# Errors where retrying is pointless (video has no captions, or YouTube is blocking our IP).
_NO_RETRY = {
    "TranscriptsDisabled", "NoTranscriptFound", "VideoUnavailable", "VideoUnplayable",
    "InvalidVideoId", "AgeRestricted", "IpBlocked", "RequestBlocked", "NoSubtitles",
}
_BLOCK_ERRORS = {"IpBlocked", "RequestBlocked"}
_TRANSCRIPT_CACHE: Dict[str, tuple] = {}


class NoSubtitles(Exception):
    pass


def _youtube_id(url_or_id: str) -> Optional[str]:
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    if "://" not in s:
        s = "https://" + s
    try:
        u = urlparse(s)
    except ValueError:
        return None
    host = (u.hostname or "").lower()
    for prefix in ("www.", "m.", "music."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    cand = ""
    if host == "youtu.be":
        cand = u.path.lstrip("/").split("/")[0]
    elif host in ("youtube.com", "youtube-nocookie.com"):
        if u.path == "/watch":
            cand = parse_qs(u.query).get("v", [""])[0]
        else:
            parts = [p for p in u.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in ("embed", "shorts", "live", "v"):
                cand = parts[1]
    return cand if re.fullmatch(r"[A-Za-z0-9_-]{11}", cand) else None


def _proxy_config():
    """Optional proxy for youtube-transcript-api. YouTube blocks most cloud/datacenter IPs,
    so on a hosted deployment you normally NEED a (residential) proxy."""
    try:
        from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig
    except Exception:
        return None
    user = os.getenv("WEBSHARE_PROXY_USERNAME")
    pwd = os.getenv("WEBSHARE_PROXY_PASSWORD")
    if user and pwd:
        return WebshareProxyConfig(proxy_username=user, proxy_password=pwd)
    url = os.getenv("YT_PROXY_URL")
    if url:
        return GenericProxyConfig(http_url=url, https_url=url)
    return None


def _fetch_with_ytapi(vid: str):
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi(proxy_config=_proxy_config())
    try:
        fetched = api.fetch(vid, languages=PREFERRED_LANGS)
        lang = getattr(fetched, "language_code", None)
    except Exception as e:
        if type(e).__name__ != "NoTranscriptFound":
            raise
        # Preferred languages missing: take any available track, manual before auto-generated.
        candidates = sorted(list(api.list(vid)), key=lambda t: bool(getattr(t, "is_generated", False)))
        if not candidates:
            raise
        chosen = candidates[0]
        fetched = chosen.fetch()
        lang = getattr(chosen, "language_code", None)
    raw = fetched.to_raw_data()
    text = " ".join(seg["text"] for seg in raw if seg.get("text"))
    return " ".join(text.split()), lang, None


def _fetch_with_ytdlp(vid: str):
    import yt_dlp

    proxy = os.getenv("YT_PROXY_URL")
    opts: dict = {"skip_download": True, "quiet": True, "no_warnings": True, "noplaylist": True}
    if proxy:
        opts["proxy"] = proxy
    cookies = os.getenv("YT_COOKIES_FILE")
    if cookies and os.path.exists(cookies):
        opts["cookiefile"] = cookies

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)

    title = info.get("title")
    proxies = {"http": proxy, "https": proxy} if proxy else None
    for pool in (info.get("subtitles") or {}, info.get("automatic_captions") or {}):
        langs = [l for l in PREFERRED_LANGS if l in pool] + [l for l in pool if l not in PREFERRED_LANGS]
        for lang in langs:
            if lang == "live_chat":
                continue
            track = next((t for t in pool[lang] if t.get("ext") == "json3"), None)
            if not track:
                continue
            r = requests.get(track["url"], timeout=20, proxies=proxies)
            r.raise_for_status()
            events = r.json().get("events", [])
            text = " ".join(
                "".join(s.get("utf8", "") for s in e["segs"]) for e in events if e.get("segs")
            )
            text = " ".join(text.split())
            if text:
                return text, lang, title
    raise NoSubtitles("yt-dlp found no subtitle tracks")


def _fetch_transcript(vid: str) -> tuple:
    """Try youtube-transcript-api, then yt-dlp. Returns (text, language, title)."""
    if vid in _TRANSCRIPT_CACHE:
        return _TRANSCRIPT_CACHE[vid]

    errors: list[str] = []
    blocked = False
    for name, fn in (("youtube-transcript-api", _fetch_with_ytapi), ("yt-dlp", _fetch_with_ytdlp)):
        for attempt in range(2):
            try:
                text, lang, title = fn(vid)
                if not text:
                    raise ValueError("empty transcript")
                _TRANSCRIPT_CACHE[vid] = (text, lang, title)
                return text, lang, title
            except Exception as e:
                ename = type(e).__name__
                errors.append(f"{name}: {ename}: {str(e).splitlines()[0][:160] if str(e) else ''}")
                if ename in _BLOCK_ERRORS or "Sign in to confirm" in str(e) or "429" in str(e):
                    blocked = True
                if ename in _NO_RETRY:
                    break
                time.sleep(1.5 * (attempt + 1))

    hint = ""
    if blocked:
        hint = (" YouTube appears to be blocking this server's IP. Run the app locally, "
                "or set WEBSHARE_PROXY_USERNAME/PASSWORD or YT_PROXY_URL in your environment.")
    raise ValueError(f"Could not fetch transcript for {vid}. " + " | ".join(errors) + hint)


def ingest_youtube(url_or_id: str, thread_id: str, title: Optional[str] = None) -> dict:
    vid = _youtube_id(url_or_id)
    if not vid:
        raise ValueError("Could not parse a YouTube video ID from the input.")

    text, used_lang, fetched_title = _fetch_transcript(vid)
    docs = [Document(
        page_content=text,
        metadata={"video_id": vid, "url": f"https://youtu.be/{vid}", "language": used_lang},
    )]
    chunks = _split(docs)
    fname = title or fetched_title or f"YouTube {vid}"
    meta = {"type": "youtube", "filename": fname, "video_id": vid,
            "language": used_lang, "chunks": len(chunks)}
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
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHAVANTAGE_API_KEY is not configured on the server."}
    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"Stock lookup failed: {e}"}


@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """Retrieve relevant chunks from the PDFs / Excel sheets / YouTube transcripts indexed in this chat."""
    # thread_id comes from the run config, so the LLM can never pass a wrong/hallucinated id.
    thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No document indexed for this chat. Upload a file or add a YouTube link first.",
                "query": query}
    result = retriever.invoke(query)
    result = [d for d in result if d.page_content.split("]", 1)[-1].strip()]
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
def chat_node(state: ChatState, config: RunnableConfig = None):
    thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
    sources = list(_THREAD_METADATA.get(str(thread_id), {}).keys())
    sources_hint = (f"Indexed sources for this chat: {sources}. "
                    if sources else "No document is indexed for this chat yet. ")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about indexed content "
            "(PDFs, Excel sheets, YouTube transcripts), call `rag_tool` with just the query. "
            f"{sources_hint}"
            "You can also use web search, stock price, and calculator tools when helpful. "
            "If no source is indexed and the user asks about a document, ask them to upload one. "
            "Never echo raw tool arguments back to the user; answer in natural language."
        )
    )
    response = llm_with_tools.invoke([system_message, *state["messages"]], config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)
checkpointer = MemorySaver()

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 7. Public helpers
# -------------------
def retrieve_all_threads() -> list[str]:
    return list(_THREAD_TITLES.keys())


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_STORES


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
