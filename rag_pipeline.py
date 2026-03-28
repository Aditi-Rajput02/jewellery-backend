"""
rag_pipeline.py
---------------
RAG Pipeline using ChromaDB.

HOW IT WORKS:
─────────────
SAVE (ingest):
  description text  →  CLIP text encoder  →  embedding vector
  → stored in ChromaDB collection "rag_transcripts" as a document (transcript)

QUERY (retrieve):
  user query text   →  CLIP text encoder  →  embedding vector
  → ChromaDB cosine similarity search
  → returns top-K most matching transcripts/descriptions

Think of each saved description as a "transcript" — a chunk of knowledge.
When the user asks a question, we find the most relevant transcripts.
"""

import os
import uuid
import datetime
import chromadb

# ── ChromaDB persistent store ──────────────────────────────────────────────────
RAG_DB_PATH    = os.path.join(os.path.dirname(__file__), "rag_chroma_db")
COLLECTION_NAME = "rag_transcripts"

_client     = None
_collection = None


# Expected embedding dimension for OpenAI text-embedding-3-small
EMBEDDING_DIM = 1536


def _get_collection():
    """
    Lazy-init ChromaDB client + collection.
    If the existing collection was built with a different embedding dimension
    (e.g. 512-dim CLIP), it is automatically deleted and recreated so the new
    1536-dim OpenAI embeddings are accepted without errors.
    """
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=RAG_DB_PATH)

        # Check if collection exists and has wrong dimension
        existing_names = [c.name for c in _client.list_collections()]
        if COLLECTION_NAME in existing_names:
            col = _client.get_collection(COLLECTION_NAME)
            try:
                # peek() with include=["embeddings"] to get actual vectors
                peek = col.peek(limit=1)
                embeddings = peek.get("embeddings") or []
                if embeddings and len(embeddings[0]) != EMBEDDING_DIM:
                    stored_dim = len(embeddings[0])
                    print(
                        f"[RAG] Dimension mismatch: stored={stored_dim}, "
                        f"expected={EMBEDDING_DIM}. Deleting old collection..."
                    )
                    _client.delete_collection(COLLECTION_NAME)
                elif not embeddings:
                    # Collection is empty — safe to reuse regardless of old dim
                    print("[RAG] Collection is empty, reusing.")
            except Exception as e:
                # If we can't probe, delete and recreate to be safe
                print(f"[RAG] Could not probe collection, recreating: {e}")
                try:
                    _client.delete_collection(COLLECTION_NAME)
                except Exception:
                    pass

        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── SAVE: store a description as a transcript ──────────────────────────────────

def save_transcript(
    description: str,
    embedding: list,          # CLIP text embedding (512-dim float32)
    sku_name: str = "",
    source: str = "manual",   # 'manual' | 'image_search' | 'text_search'
) -> str:
    """
    Save a jewellery description + its CLIP embedding into ChromaDB.

    Each entry is a "transcript" — a searchable chunk of knowledge.
    If sku_name is provided, it is used as the document ID (upsert = update if exists).
    Otherwise a random UUID is assigned.

    Returns the ID of the saved entry.
    """
    col = _get_collection()

    # Stable ID: use sku_name if given, else random UUID
    doc_id = sku_name.strip() if sku_name.strip() else str(uuid.uuid4())

    metadata = {
        "sku_name": sku_name.strip() or "",
        "source":   source,
        "saved_at": datetime.datetime.utcnow().isoformat(),
        "length":   str(len(description)),
    }

    # upsert: insert new OR update existing entry with same ID
    col.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[description],   # the raw text stored as the "transcript"
        metadatas=[metadata],
    )

    print(f"[RAG] Saved transcript id={doc_id!r} sku={sku_name!r} len={len(description)}")
    return doc_id


# ── QUERY: find best matching transcripts ─────────────────────────────────────

def query_transcripts(
    query_embedding: list,    # CLIP text embedding of the user's query
    top_k: int = 8,
) -> list:
    """
    Given a query embedding, find the top-K most similar transcripts in ChromaDB.

    Returns a list of dicts:
      {
        rank, id, description, sku_name, source, saved_at,
        similarity_pct,   # raw cosine similarity 0-100
        confidence_pct,   # scaled 40-100 for display
        distance,
      }
    """
    col = _get_collection()
    total = col.count()
    if total == 0:
        return []

    k = min(top_k, total)

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    ids       = results["ids"][0]
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    # Scale confidence: best match → 100%, worst match → 40%
    min_d = min(distances) if distances else 0.0
    max_d = max(distances) if distances else 1.0

    matches = []
    for rank, (doc_id, doc, meta, dist) in enumerate(
        zip(ids, docs, metas, distances), start=1
    ):
        raw_sim = max(0.0, 1.0 - dist)          # cosine similarity 0→1

        if max_d > min_d:
            relative = 1.0 - (dist - min_d) / (max_d - min_d)
        else:
            relative = 1.0
        confidence_pct = round(40 + relative * 60, 1)   # 40–100 %

        matches.append({
            "rank":           rank,
            "id":             doc_id,
            "description":    doc,               # the stored transcript text
            "sku_name":       meta.get("sku_name", ""),
            "item_name":      meta.get("item_name", ""),
            "style":          meta.get("style", ""),
            "metal":          meta.get("metal", ""),
            "color":          meta.get("color", ""),
            "price":          meta.get("price", ""),
            "tag_price":      meta.get("tag_price", ""),
            "currency":       meta.get("currency", ""),
            "gross_wt":       meta.get("gross_wt", ""),
            "net_wt":         meta.get("net_wt", ""),
            "dia_wt":         meta.get("dia_wt", ""),
            "cs_wt":          meta.get("cs_wt", ""),
            "ref_d_no":       meta.get("ref_d_no", ""),
            "picture1":       meta.get("picture1", ""),
            "image_url":      meta.get("image_url", ""),
            "sku_image1":     meta.get("sku_image1", ""),
            "sku_image2":     meta.get("sku_image2", ""),
            "source":         meta.get("source", ""),
            "saved_at":       meta.get("saved_at", ""),
            "similarity_pct": round(raw_sim * 100, 1),
            "confidence_pct": confidence_pct,
            "distance":       round(dist, 4),
        })

    return matches


# ── UTILS ──────────────────────────────────────────────────────────────────────

def count_transcripts() -> int:
    """Return total number of transcripts stored."""
    try:
        return _get_collection().count()
    except Exception:
        return 0


def list_transcripts(limit: int = 200) -> list:
    """Return up to `limit` stored transcripts (for admin/debug view)."""
    try:
        col = _get_collection()
        total = col.count()
        if total == 0:
            return []
        results = col.get(
            limit=min(limit, total),
            include=["documents", "metadatas"],
        )
        out = []
        for doc_id, doc, meta in zip(
            results["ids"], results["documents"], results["metadatas"]
        ):
            out.append({
                "id":          doc_id,
                "description": doc,
                "sku_name":    meta.get("sku_name", ""),
                "source":      meta.get("source", ""),
                "saved_at":    meta.get("saved_at", ""),
                "length":      meta.get("length", ""),
            })
        return out
    except Exception:
        return []


def delete_transcript(doc_id: str) -> bool:
    """Delete a transcript by ID. Returns True on success."""
    try:
        _get_collection().delete(ids=[doc_id])
        return True
    except Exception:
        return False
