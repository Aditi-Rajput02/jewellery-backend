"""
rag_routes.py
─────────────
Complete RAG (Retrieval-Augmented Generation) pipeline as a FastAPI APIRouter.

CONCEPT
═══════
Think of every jewellery description you save as a "transcript" — a chunk of
knowledge stored in ChromaDB with a CLIP text embedding attached to it.

SAVE (ingest a transcript):
  description text
       │
       ▼
  CLIP text encoder  (ViT-B/32, 512-dim)
       │
       ▼
  embedding vector
       │
       ▼
  ChromaDB  ──  collection: "rag_transcripts"
               stores: { id, document (text), embedding, metadata }

SEARCH (retrieve matching transcripts):
  user query text
       │
       ▼
  CLIP text encoder  (same model, same space)
       │
       ▼
  embedding vector
       │
       ▼
  ChromaDB cosine similarity search
       │
       ▼
  top-K most relevant transcripts  ←── returned to the user

ENDPOINTS
═════════
  POST   /rag/save    – save a description as a transcript
  GET    /rag/search  – query: find best matching transcripts
  GET    /rag/count   – how many transcripts are stored

HOW TO MOUNT IN main.py
════════════════════════
  from rag_routes import rag_router
  app.include_router(rag_router)
"""

import torch
import clip

from fastapi import APIRouter, Form, Query, HTTPException
from fastapi.responses import JSONResponse

from rag_pipeline import (
    save_transcript,
    query_transcripts,
    count_transcripts,
)

# ── Router ─────────────────────────────────────────────────────────────────────
rag_router = APIRouter(prefix="/rag", tags=["RAG Pipeline"])

# ── CLIP device (shared with main.py via module-level globals) ─────────────────
# We load CLIP here lazily so this file can be imported without crashing if
# CLIP is not yet loaded.  main.py loads CLIP at startup; if you mount this
# router into main.py the model is already in memory.
_device     = None
_clip_model = None


def _get_clip():
    """Lazy-load CLIP ViT-B/32 (only once per process)."""
    global _device, _clip_model
    if _clip_model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[RAG] Loading CLIP ViT-B/32 on {_device}…")
        _clip_model, _ = clip.load("ViT-B/32", device=_device)
        _clip_model.eval()
        print("[RAG] CLIP loaded OK")
    return _clip_model, _device


def _embed(text: str) -> list:
    """
    Encode `text` with CLIP and return a normalised 512-dim float32 list.

    This is the core embedding step used by both SAVE and SEARCH.
    The same model + same vector space ensures that similar descriptions
    end up close together in ChromaDB.
    """
    model, dev = _get_clip()
    tokens = clip.tokenize([text.strip()], truncate=True).to(dev)
    with torch.no_grad():
        feat = model.encode_text(tokens)
    feat = feat / feat.norm(dim=-1, keepdim=True)          # L2-normalise
    return feat.cpu().numpy()[0].astype("float32").tolist()


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — SAVE
# ══════════════════════════════════════════════════════════════════════════════

@rag_router.post("/save", summary="Save a description as a RAG transcript")
async def rag_save(
    description: str = Form(...,  description="Jewellery description to store"),
    sku_name:    str = Form("",   description="Optional SKU (used as document ID)"),
    source:      str = Form("manual", description="Origin: manual | image_search | text_search"),
):
    """
    **SAVE** — Ingest a jewellery description into the RAG knowledge base.

    Pipeline
    --------
    1. Receive `description` text from the user / system.
    2. Encode it with **CLIP text encoder** → 512-dim embedding.
    3. **Upsert** into ChromaDB collection `rag_transcripts`:
       - `id`       → `sku_name` if provided, else a random UUID
       - `document` → the raw description text (the "transcript")
       - `embedding`→ the CLIP vector
       - `metadata` → { sku_name, source, saved_at, length }

    If the same `sku_name` is saved again, the existing entry is **updated**
    (upsert semantics), so you can refine descriptions without duplicates.

    Returns
    -------
    ```json
    {
      "saved": true,
      "id": "BGSNL-1018",
      "sku_name": "BGSNL-1018",
      "description": "A gold necklace with diamond pendant…",
      "total_transcripts": 42,
      "message": "Description saved to ChromaDB (id=BGSNL-1018). Total transcripts: 42."
    }
    ```
    """
    if not description or not description.strip():
        raise HTTPException(status_code=400, detail="description cannot be empty.")

    try:
        # Step 1 — embed
        embedding = _embed(description.strip())

        # Step 2 — store in ChromaDB
        doc_id = save_transcript(
            description=description.strip(),
            embedding=embedding,
            sku_name=sku_name.strip(),
            source=source.strip() or "manual",
        )

        total = count_transcripts()
        print(f"[RAG/save] id={doc_id!r}  sku={sku_name!r}  total={total}")

        return JSONResponse({
            "saved":             True,
            "id":                doc_id,
            "sku_name":          sku_name.strip(),
            "description":       description.strip(),
            "total_transcripts": total,
            "message": (
                f"Description saved to ChromaDB (id={doc_id}). "
                f"Total transcripts: {total}."
            ),
        })

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG save error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — SEARCH
# ══════════════════════════════════════════════════════════════════════════════

async def _do_search(query: str, top_k: int):
    """Shared search logic for both GET and POST endpoints."""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty.")

    total_stored = count_transcripts()

    if total_stored == 0:
        return JSONResponse({
            "query":         query.strip(),
            "total_stored":  0,
            "total_matches": 0,
            "matches":       [],
            "message": "No transcripts stored yet. Use POST /rag/save to add descriptions first.",
        })

    try:
        embedding = _embed(query.strip())
        matches   = query_transcripts(embedding, top_k=top_k)

        print(f"[RAG/search] query={query!r} -> {len(matches)} matches (stored={total_stored})")

        return JSONResponse({
            "query":         query.strip(),
            "total_stored":  total_stored,
            "total_matches": len(matches),
            "matches":       matches,
        })

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG search error: {exc}")


@rag_router.get("/search", summary="Search for matching RAG transcripts (GET)")
async def rag_search_get(
    query: str = Query(..., description="Natural-language search query"),
    top_k: int = Query(default=8, ge=1, le=50, description="Number of results"),
):
    """
    **SEARCH (GET)** — Find the most relevant transcripts via query params.

    Example: `GET /rag/search?query=silver+necklace&top_k=8`

    Each match includes: `rank`, `id`, `sku_name`, `item_name`, `style`, `metal`,
    `color`, `price`, `tag_price`, `currency`, `gross_wt`, `net_wt`, `dia_wt`,
    `cs_wt`, `ref_d_no`, `picture1`, `image_url`, `sku_image1`, `sku_image2`,
    `description`, `source`, `saved_at`, `similarity_pct`, `confidence_pct`, `distance`.
    """
    return await _do_search(query, top_k)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — COUNT
# ══════════════════════════════════════════════════════════════════════════════

@rag_router.get("/count", summary="Count stored RAG transcripts")
async def rag_count():
    """Return the total number of transcripts currently stored in ChromaDB."""
    total = count_transcripts()
    return JSONResponse({
        "total_transcripts": total,
        "ready": total > 0,
        "message": (
            f"{total} transcript(s) stored in ChromaDB."
            if total > 0
            else "No transcripts yet. Use POST /rag/save to add descriptions."
        ),
    })


