
from pathlib import Path

import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import fitz  # PyMuPDF

import pdf_loader

VECTORSTORE_DIR = Path(__file__).parent.parent / "vectorstore"
COLLECTION_NAME = "research_agent_docs"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


class IndexingError(Exception):
    """Raised when a PDF can't be extracted or indexed."""


def _get_chroma_collection():
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def _get_embed_model(embed_model=None):
    if embed_model is not None:
        return embed_model
    from llama_index.embeddings.ollama import OllamaEmbedding
    return OllamaEmbedding(model_name="nomic-embed-text")


def _extract_text(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    except Exception as exc:
        raise IndexingError(f"Failed to extract text from {pdf_path}: {exc}") from exc

    if not text.strip():
        raise IndexingError(
            f"No extractable text found in {pdf_path} (possibly a scanned "
            f"image PDF with no OCR layer)."
        )
    return text


def index_pdf(file_hash: str, embed_model=None, force: bool = False) -> dict:
    """
    Index a single already-uploaded PDF (looked up by hash via
    pdf_loader's manifest) into the Chroma vector store.

    Returns {"status": "indexed"|"skipped"|"error", "chunks": int, ...}
    """
    entries = {e["hash"]: e for e in pdf_loader.list_uploaded_pdfs()}
    entry = entries.get(file_hash)
    if entry is None:
        return {"status": "error", "message": f"No uploaded PDF with hash {file_hash}"}

    if entry.get("indexed") and not force:
        return {"status": "skipped", "message": "Already indexed", "chunks": 0}

    text = _extract_text(entry["path"])

    document = Document(
        text=text,
        metadata={"source_filename": entry["filename"], "file_hash": file_hash},
    )

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents([document])

    collection = _get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=_get_embed_model(embed_model),
    )

    pdf_loader.mark_indexed(file_hash)

    return {"status": "indexed", "chunks": len(nodes), "filename": entry["filename"]}


def index_all_unindexed(embed_model=None) -> list[dict]:
    pending = pdf_loader.list_uploaded_pdfs(unindexed_only=True)
    results = []
    for entry in pending:
        result = index_pdf(entry["hash"], embed_model=embed_model)
        results.append({"filename": entry["filename"], **result})
    return results


def collection_stats() -> dict:
    collection = _get_chroma_collection()
    return {"total_chunks": collection.count(), "collection_name": COLLECTION_NAME}