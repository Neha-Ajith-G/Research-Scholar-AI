
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
 
UPLOADS_DIR = Path(__file__).parent.parent / "uploads"
MANIFEST_PATH = UPLOADS_DIR / "manifest.json"
 
PDF_MAGIC_BYTES = b"%PDF-"
 
 
class InvalidPDFError(Exception):
    """Raised when the uploaded content isn't actually a PDF."""
 
 
def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
 
 
def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
 
 
def _compute_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
 
 
def _validate_pdf(content: bytes) -> None:
    if not content.startswith(PDF_MAGIC_BYTES):
        raise InvalidPDFError(
            "File content doesn't look like a PDF (missing %PDF- header)."
        )
 
 
def save_pdf(content: bytes, original_filename: str) -> dict:
    """
    Save uploaded PDF bytes to uploads/, keyed by content hash.
 
    Returns a dict:
      {"hash": ..., "filename": ..., "path": ..., "duplicate": bool}
 
    Raises InvalidPDFError if content isn't a real PDF.
    """
    _validate_pdf(content)
 
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    file_hash = _compute_hash(content)
    manifest = _load_manifest()
 
    if file_hash in manifest:
        entry = manifest[file_hash]
        return {
            "hash": file_hash,
            "filename": entry["filename"],
            "path": entry["path"],
            "duplicate": True,
        }
 
    # Store under the hash to avoid filename collisions between different
    # uploads, while keeping the original name for display purposes.
    safe_name = f"{file_hash[:12]}_{Path(original_filename).name}"
    dest_path = UPLOADS_DIR / safe_name
    dest_path.write_bytes(content)
 
    manifest[file_hash] = {
        "filename": original_filename,
        "path": str(dest_path),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "indexed": False,  # Step 4 flips this to True once chunked+embedded
    }
    _save_manifest(manifest)
 
    return {
        "hash": file_hash,
        "filename": original_filename,
        "path": str(dest_path),
        "duplicate": False,
    }
 
 
def list_uploaded_pdfs(indexed_only: bool = False, unindexed_only: bool = False) -> list[dict]:
    """List all uploaded PDFs from the manifest, optionally filtered by
    indexing status. Step 4 uses unindexed_only=True to know what's left
    to process."""
    manifest = _load_manifest()
    results = []
    for file_hash, entry in manifest.items():
        if indexed_only and not entry.get("indexed"):
            continue
        if unindexed_only and entry.get("indexed"):
            continue
        results.append({"hash": file_hash, **entry})
    return results
 
 
def mark_indexed(file_hash: str) -> None:
    """Called by rag/indexer.py after successfully indexing a PDF."""
    manifest = _load_manifest()
    if file_hash in manifest:
        manifest[file_hash]["indexed"] = True
        _save_manifest(manifest)
 