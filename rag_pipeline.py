"""
rag_pipeline.py
---------------
RAG Pipeline using MongoDB as the vector store.

WHY MONGODB (not ChromaDB on disk):
  Render's free tier has ephemeral disk — local files are wiped on every deploy.
  MongoDB Atlas persists forever, so embeddings survive restarts and redeploys.

HOW IT WORKS:
─────────────
SAVE (ingest):
  description text → OpenAI text-embedding-3-small → 1536-dim embedding
  → stored in MongoDB collection "rag_embeddings"

QUERY (retrieve):
  user query text → OpenAI text-embedding-3-small → 1536-dim embedding
  → cosine similarity against all stored embeddings (in Python)
  → returns top-K most matching transcripts

Each saved description is a "transcript" — a searchable chunk of knowledge.
"""

import os
import uuid
import datetime
import numpy as np
from mongo_db import get_db

COLLECTION_NAME = "rag_embeddings"
EMBEDDING_DIM   = 1536


def _col():
    """Return the MongoDB rag_embeddings collection."""
    db = get_db()
    col = db[COLLECTION_NAME]
    # Ensure unique index on sku_name for fast upserts
    col.create_index("sku_name", unique=False, background=True)
    return col


# ── SAVE ──────────────────────────────────────────────────────────────────────

def save_transcript(
    description: str,
    embedding: list,
    sku_name: str = "",
    source: str = "manual",
) -> str:
    """
    Save a description + its embedding into MongoDB rag_embeddings.
    If sku_name is provided, upserts (updates existing entry for that SKU).
    Returns the document ID.
    """
    col = _col()
    doc_id = sku_name.strip() if sku_name.strip() else str(uuid.uuid4())

    doc = {
        "_id":         doc_id,
        "sku_name":    sku_name.strip() or "",
        "description": description,
        "embedding":   embedding,          # list of 1536 floats
        "source":      source,
        "saved_at":    datetime.datetime.now(datetime.UTC).isoformat(),
        "length":      len(description),
    }

    col.replace_one({"_id": doc_id}, doc, upsert=True)
    print(f"[RAG] Saved transcript id={doc_id!r} sku={sku_name!r} len={len(description)}")
    return doc_id


# ── QUERY ─────────────────────────────────────────────────────────────────────

def query_transcripts(
    query_embedding: list,
    top_k: int = 8,
) -> list:
    """
    Find top-K most similar transcripts using cosine similarity.
    Loads all embeddings from MongoDB and computes similarity in NumPy.
    Fast enough for up to ~10,000 documents.
    """
    col = _col()
    total = col.count_documents({})
    if total == 0:
        return []

    # Load all docs with embeddings
    docs = list(col.find({}, {
        "_id": 1, "sku_name": 1, "description": 1, "embedding": 1,
        "source": 1, "saved_at": 1,
        "item_name": 1, "style": 1, "metal": 1, "color": 1,
        "price": 1, "picture1": 1, "image_url": 1,
    }))

    if not docs:
        return []

    # Build matrix for vectorised cosine similarity
    q = np.array(query_embedding, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-10)

    embeddings = np.array([d["embedding"] for d in docs], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    embeddings_norm = embeddings / norms

    similarities = embeddings_norm @ q_norm   # cosine similarity for each doc

    # Get top-K indices
    k = min(top_k, len(docs))
    top_indices = np.argsort(similarities)[::-1][:k]

    sims = similarities[top_indices]
    min_s = float(sims[-1]) if len(sims) > 1 else float(sims[0])
    max_s = float(sims[0])

    matches = []
    for rank, idx in enumerate(top_indices, start=1):
        doc  = docs[idx]
        sim  = float(similarities[idx])
        dist = 1.0 - sim   # convert to distance (0=identical, 2=opposite)

        if max_s > min_s:
            relative = (sim - min_s) / (max_s - min_s)
        else:
            relative = 1.0
        confidence_pct = round(40 + relative * 60, 1)   # 40–100 %

        matches.append({
            "rank":           rank,
            "id":             str(doc["_id"]),
            "description":    doc.get("description", ""),
            "sku_name":       doc.get("sku_name", ""),
            "item_name":      doc.get("item_name", ""),
            "style":          doc.get("style", ""),
            "metal":          doc.get("metal", ""),
            "color":          doc.get("color", ""),
            "price":          doc.get("price", ""),
            "tag_price":      doc.get("tag_price", ""),
            "currency":       doc.get("currency", ""),
            "gross_wt":       doc.get("gross_wt", ""),
            "net_wt":         doc.get("net_wt", ""),
            "dia_wt":         doc.get("dia_wt", ""),
            "cs_wt":          doc.get("cs_wt", ""),
            "ref_d_no":       doc.get("ref_d_no", ""),
            "picture1":       doc.get("picture1", ""),
            "image_url":      doc.get("image_url", ""),
            "sku_image1":     doc.get("sku_image1", ""),
            "sku_image2":     doc.get("sku_image2", ""),
            "source":         doc.get("source", ""),
            "saved_at":       doc.get("saved_at", ""),
            "similarity_pct": round(sim * 100, 1),
            "confidence_pct": confidence_pct,
            "distance":       round(dist, 4),
        })

    return matches


# ── UTILS ──────────────────────────────────────────────────────────────────────

def count_transcripts() -> int:
    """Return total number of transcripts stored."""
    try:
        return _col().count_documents({})
    except Exception:
        return 0


def list_transcripts(limit: int = 200) -> list:
    """Return up to `limit` stored transcripts (for admin/debug view)."""
    try:
        docs = list(_col().find(
            {},
            {"_id": 1, "sku_name": 1, "description": 1, "source": 1, "saved_at": 1, "length": 1}
        ).limit(limit))
        return [{
            "id":          str(d["_id"]),
            "sku_name":    d.get("sku_name", ""),
            "description": d.get("description", ""),
            "source":      d.get("source", ""),
            "saved_at":    d.get("saved_at", ""),
            "length":      d.get("length", ""),
        } for d in docs]
    except Exception:
        return []


def delete_transcript(doc_id: str) -> bool:
    """Delete a transcript by ID. Returns True on success."""
    try:
        _col().delete_one({"_id": doc_id})
        return True
    except Exception:
        return False


# ── LEGACY COMPAT (ChromaDB stubs — no longer used) ───────────────────────────
def _get_collection():
    """Compatibility shim — returns the MongoDB collection wrapper."""
    return _col()
