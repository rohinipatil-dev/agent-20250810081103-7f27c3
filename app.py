import os
import io
import uuid
import time
from typing import List, Dict, Any, Tuple

import streamlit as st

# OpenAI SDK (required by spec)
from openai import OpenAI

# Optional/External dependencies for RAG
try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

try:
    from pypdf import PdfReader
except Exception as e:
    PdfReader = None


# ------------- Utility: Dependency checks -------------
def check_dependencies():
    missing = []
    if chromadb is None:
        missing.append("chromadb")
    if SentenceTransformer is None:
        missing.append("sentence-transformers")
    if PdfReader is None:
        missing.append("pypdf")
    if missing:
        st.error(
            "Missing required packages: "
            + ", ".join(missing)
            + "\nInstall them via:\n"
            "pip install chromadb sentence-transformers pypdf"
        )
        st.stop()


# ------------- Embedding Model Loader -------------
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str = "all-MiniLM-L6-v2"):
    model_full_name = model_name
    # sentence-transformers convention supports both "all-MiniLM-L6-v2" and "sentence-transformers/all-MiniLM-L6-v2"
    # We try a sensible default fallback.
    try_names = [model_full_name, f"sentence-transformers/{model_full_name}"]
    last_err = None
    for name in try_names:
        try:
            model = SentenceTransformer(name)
            return model
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load embedding model '{model_name}'. Last error: {last_err}")


def embed_texts(embedder, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    vectors = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [v.tolist() for v in vectors]


# ------------- Chroma Client/Collection -------------
@st.cache_resource(show_spinner=False)
def get_chroma_client(persist_dir: str):
    # Try new API first
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        return client
    except Exception:
        pass
    # Fallback to legacy API
    try:
        client = chromadb.Client(Settings(persist_directory=persist_dir))
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChromaDB client: {e}")


def get_or_create_collection(client, name: str):
    try:
        col = client.get_or_create_collection(name=name)
        return col
    except Exception as e:
        raise RuntimeError(f"Failed to get or create collection '{name}': {e}")


# ------------- Text Extraction and Chunking -------------
def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    return "\n".join(pages_text)


def read_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="ignore")


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(end - chunk_overlap, 0)
    return [c.strip() for c in chunks if c.strip()]


def process_uploaded_files(files, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    docs = []
    for f in files:
        filename = f.name
        data = f.read()
        if filename.lower().endswith(".pdf"):
            content = read_pdf(data)
        elif filename.lower().endswith(".txt"):
            content = read_txt(data)
        else:
            # For safety, try text decode
            content = read_txt(data)

        chunks = split_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, ch in enumerate(chunks):
            docs.append(
                {
                    "id": f"{uuid.uuid4()}",
                    "text": ch,
                    "metadata": {
                        "source": filename,
                        "chunk_index": idx,
                        "ts": int(time.time()),
                    },
                }
            )
    return docs


# ------------- Ingestion into Chroma -------------
def upsert_documents(collection, embedder, docs: List[Dict[str, Any]]) -> int:
    if not docs:
        return 0
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metas = [d["metadata"] for d in docs]
    embeddings = embed_texts(embedder, texts)
    # batched add to avoid large payloads
    batch_size = 64
    total = 0
    for i in range(0, len(ids), batch_size):
        j = i + batch_size
        collection.add(
            ids=ids[i:j],
            documents=texts[i:j],
            metadatas=metas[i:j],
            embeddings=embeddings[i:j],
        )
        total += len(ids[i:j])
    return total


# ------------- Retrieval -------------
def retrieve_top_k(collection, embedder, query: str, k: int = 5) -> Dict[str, Any]:
    q_emb = embed_texts(embedder, [query])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    # results keys: ids, distances, documents, metadatas
    return results


def build_context_from_results(results: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    entries = []
    for d, m in zip(docs, metas):
        entries.append({"text": d, "meta": m})
    context_parts = []
    for i, e in enumerate(entries, 1):
        src = e["meta"].get("source", "unknown")
        idx = e["meta"].get("chunk_index", -1)
        context_parts.append(f"[{i}] (source: {src}, chunk: {idx})\n{e['text']}")
    context = "\n\n".join(context_parts)
    return context, entries


# ------------- Generation -------------
def generate_with_openai(context: str, question: str, temperature: float = 0.2) -> str:
    client = OpenAI()
    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer the question. "
        "If the answer is not in the context, say you don't know."
    )
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly and concisely. Cite sources by their index like [1], [2] when relevant."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"(OpenAI generation failed: {e})"


def generate_locally_heuristic(context_entries: List[Dict[str, Any]], question: str) -> str:
    # Simple keyword-based extractive response when no LLM/API key is available.
    q_terms = {w.lower() for w in question.split() if len(w) > 2}
    ranked = []
    for i, e in enumerate(context_entries, 1):
        text = e["text"]
        tokens = [w.strip(".,;:!?()[]{}\"'").lower() for w in text.split()]
        score = sum(1 for t in tokens if t in q_terms)
        ranked.append((score, i, e))
    ranked.sort(reverse=True, key=lambda x: x[0])
    snippets = []
    for score, idx, e in ranked[:3]:
        src = e["meta"].get("source", "unknown")
        cidx = e["meta"].get("chunk_index", -1)
        snippet = e["text"]
        if len(snippet) > 600:
            snippet = snippet[:600] + "..."
        snippets.append(f"[{idx}] (source: {src}, chunk: {cidx})\n{snippet}")
    if not snippets:
        return "No relevant information found in the knowledge base."
    answer = (
        "Here are the most relevant excerpts based on your query (no LLM used):\n\n"
        + "\n\n".join(snippets)
        + "\n\nConsider enabling OpenAI GPT-4 for a synthesized answer."
    )
    return answer


# ------------- Streamlit UI -------------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "embedder_name" not in st.session_state:
        st.session_state.embedder_name = "all-MiniLM-L6-v2"
    if "persist_dir" not in st.session_state:
        st.session_state.persist_dir = "chroma_db"
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = "rag_collection"
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 800
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 200


def sidebar_controls():
    st.sidebar.header("Settings")
    st.session_state.persist_dir = st.sidebar.text_input(
        "ChromaDB directory", value=st.session_state.persist_dir
    )
    st.session_state.collection_name = st.sidebar.text_input(
        "Collection name", value=st.session_state.collection_name
    )
    st.session_state.embedder_name = st.sidebar.text_input(
        "Embedding model (sentence-transformers)", value=st.session_state.embedder_name
    )
    st.session_state.chunk_size = st.sidebar.number_input(
        "Chunk size (chars)", min_value=200, max_value=5000, value=st.session_state.chunk_size, step=50
    )
    st.session_state.chunk_overlap = st.sidebar.number_input(
        "Chunk overlap (chars)", min_value=0, max_value=1000, value=st.session_state.chunk_overlap, step=10
    )
    st.session_state.top_k = st.sidebar.number_input(
        "Top-K retrieval", min_value=1, max_value=20, value=st.session_state.top_k, step=1
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear chat history"):
        st.session_state.messages = []
        st.sidebar.success("Chat history cleared.")

    if st.sidebar.button("Delete collection"):
        try:
            client = get_chroma_client(st.session_state.persist_dir)
            client.delete_collection(st.session_state.collection_name)
            st.sidebar.success("Collection deleted.")
        except Exception as e:
            st.sidebar.error(f"Failed to delete collection: {e}")

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Generation modes:\n"
        "- If an OpenAI API key is configured in your environment, the app will use GPT-4.\n"
        "- Otherwise, it will use a local extractive heuristic (no external API)."
    )


def main():
    st.set_page_config(page_title="Internal RAG Chatbot", page_icon="🧠", layout="wide")
    st.title("🧠 Internal RAG Chatbot")
    st.caption(
        "Upload PDFs or text files, store them in a local ChromaDB with vector embeddings, and chat with your knowledge base."
    )
    st.markdown(
        "Instructions:\n"
        "1) Upload one or more PDF/TXT files.\n"
        "2) Click 'Ingest to ChromaDB' to index them with embeddings.\n"
        "3) Ask questions below. The app retrieves relevant chunks and answers your query.\n"
        "Note: Without an OpenAI API key, the app will return extractive excerpts instead of a synthesized response."
    )

    init_session_state()
    sidebar_controls()
    check_dependencies()

    # Initialize services
    embedder = load_embedder(st.session_state.embedder_name)
    chroma_client = get_chroma_client(st.session_state.persist_dir)
    collection = get_or_create_collection(chroma_client, st.session_state.collection_name)

    # File uploader
    st.subheader("Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Drop your files here to add them to the knowledge base.",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Ingest to ChromaDB", type="primary", disabled=not uploaded_files):
            with st.spinner("Processing and embedding documents..."):
                docs = process_uploaded_files(
                    uploaded_files, st.session_state.chunk_size, st.session_state.chunk_overlap
                )
                added = upsert_documents(collection, embedder, docs)
                st.success(f"Ingested {added} chunks into collection '{st.session_state.collection_name}'.")
    with col2:
        if st.button("Show collection stats"):
            try:
                count = collection.count()
                st.info(f"Collection '{st.session_state.collection_name}' has {count} chunks.")
            except Exception as e:
                st.error(f"Failed to get stats: {e}")

    st.markdown("---")
    st.subheader("Chat")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask a question about your documents...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context..."):
                try:
                    results = retrieve_top_k(collection, embedder, user_input, st.session_state.top_k)
                    context, entries = build_context_from_results(results)
                except Exception as e:
                    st.error(f"Retrieval failed: {e}")
                    return

            # Decide generation mode
            use_openai = bool(os.environ.get("OPENAI_API_KEY"))
            if use_openai:
                with st.spinner("Generating answer (GPT-4)..."):
                    answer = generate_with_openai(context, user_input, temperature=0.2)
            else:
                with st.spinner("Generating extractive answer (no external API)..."):
                    answer = generate_locally_heuristic(entries, user_input)

            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            with st.expander("View retrieved sources", expanded=False):
                for i, e in enumerate(entries, 1):
                    src = e["meta"].get("source", "unknown")
                    idx = e["meta"].get("chunk_index", -1)
                    preview = e["text"][:500].replace("\n", " ")
                    st.write(f"[{i}] Source: {src} | Chunk: {idx}")
                    st.caption(preview + ("..." if len(e["text"]) > 500 else ""))


if __name__ == "__main__":
    main()