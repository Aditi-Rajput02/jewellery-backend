"""
rag_routes.py
─────────────
RAG (Retrieval-Augmented Generation) pipeline as a FastAPI APIRouter.
Uses OpenAI text-embedding-3-small for embeddings (no local model needed).

ENDPOINTS
═════════
  POST   /rag/save    – save a description as a transcript
  GET    /rag/search  – query: find best matching transcripts
  GET    /rag/count   – how many transcripts are stored
"""

import os
from openai import OpenAI
from fastapi import APIRouter, Form, Query, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from rag_pipeline import (
    save_transcript,
    query_transcripts,
    count_transcripts,
)

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# ── Router ─────────────────────────────────────────────────────────────────────
rag_router = APIRouter(prefix="/rag", tags=["RAG Pipeline"])


def _embed(text: str) -> list:
    """
    Encode text with OpenAI text-embedding-3-small → normalised 1536-dim float list.
    No local model download. Uses the same OpenAI key as the rest of the app.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model="text-embedding-3-small", input=text.strip())
    return resp.data[0].embedding


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

    Pipeline: description → OpenAI embedding → ChromaDB upsert
    """
    if not description or not description.strip():
        raise HTTPException(status_code=400, detail="description cannot be empty.")

    try:
        embedding = _embed(description.strip())
        doc_id = save_transcript(
            description=description.strip(),
            embedding=embedding,
            sku_name=sku_name.strip(),
            source=source.strip() or "manual",
        )
        total = count_transcripts()
        return JSONResponse({
            "saved":             True,
            "id":                doc_id,
            "sku_name":          sku_name.strip(),
            "description":       description.strip(),
            "total_transcripts": total,
            "message": f"Description saved to ChromaDB (id={doc_id}). Total transcripts: {total}.",
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG save error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — SEARCH
# ══════════════════════════════════════════════════════════════════════════════

async def _do_search(query: str, top_k: int):
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
        return JSONResponse({
            "query":         query.strip(),
            "total_stored":  total_stored,
            "total_matches": len(matches),
            "matches":       matches,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG search error: {exc}")


@rag_router.get("/search", summary="Search for matching RAG transcripts")
async def rag_search_get(
    query: str = Query(..., description="Natural-language search query"),
    top_k: int = Query(default=8, ge=1, le=50, description="Number of results"),
):
    return await _do_search(query, top_k)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — COUNT
# ══════════════════════════════════════════════════════════════════════════════

@rag_router.get("/count", summary="Count stored RAG transcripts")
async def rag_count():
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


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 4 — SYNC (manual trigger)
# ══════════════════════════════════════════════════════════════════════════════

@rag_router.post("/sync", summary="Sync all MongoDB product descriptions → ChromaDB")
async def rag_sync():
    """
    **SYNC** — Pull all product descriptions from MongoDB and upsert them into
    ChromaDB. Safe to call multiple times — only missing SKUs are added.

    This is the same logic that runs automatically on server startup.
    Call this after adding new descriptions via /catalog/update-description
    or /catalog/describe-from-url.
    """
    import asyncio
    import datetime as dt
    from mongo_db import get_db
    from rag_pipeline import save_transcript, count_transcripts, _get_collection

    try:
        db = get_db()
        sku_desc_map: dict = {}

        for coll_name in ("catalog", "product_details", "product_list"):
            for doc in db[coll_name].find(
                {"description": {"$exists": True, "$nin": [None, ""]}},
                {"sku_name": 1, "description": 1, "item_name": 1,
                 "style_name": 1, "metal_code": 1, "color_name": 1,
                 "show_catalog_price": 1, "picture1": 1}
            ):
                sku  = doc.get("sku_name", "").strip()
                desc = (doc.get("description") or "").strip()
                if sku and desc and sku not in sku_desc_map:
                    sku_desc_map[sku] = doc

        total_mongo = len(sku_desc_map)
        if total_mongo == 0:
            return JSONResponse({"synced": 0, "errors": 0, "total_in_chroma": 0,
                                 "message": "No product descriptions found in MongoDB."})

        col = _get_collection()
        existing_ids = set()
        try:
            existing = col.get(include=[])
            existing_ids = set(existing.get("ids", []))
        except Exception:
            pass

        to_sync = {k: v for k, v in sku_desc_map.items() if k not in existing_ids}

        if not to_sync:
            return JSONResponse({
                "synced": 0, "errors": 0,
                "total_in_chroma": count_transcripts(),
                "message": f"Already up to date. {len(existing_ids)} SKUs in ChromaDB.",
            })

        client = OpenAI(api_key=OPENAI_API_KEY)
        IMG_BASE = "https://bgjewels.jewelscore.com/f/shortimage/"
        synced = 0
        errors = 0

        for sku, doc in to_sync.items():
            try:
                desc = doc["description"].strip()
                resp = client.embeddings.create(model="text-embedding-3-small", input=desc)
                embedding = resp.data[0].embedding
                price = doc.get("show_catalog_price")
                pic   = doc.get("picture1", "")
                col.upsert(
                    ids=[sku],
                    embeddings=[embedding],
                    documents=[desc],
                    metadatas=[{
                        "sku_name":  sku,
                        "item_name": doc.get("item_name", ""),
                        "style":     doc.get("style_name", ""),
                        "metal":     doc.get("metal_code", ""),
                        "color":     doc.get("color_name", ""),
                        "price":     str(price) if price else "",
                        "picture1":  pic,
                        "image_url": (IMG_BASE + pic) if pic else "",
                        "source":    "mongo_sync",
                        "saved_at":  dt.datetime.utcnow().isoformat(),
                        "length":    str(len(desc)),
                    }],
                )
                synced += 1
                await asyncio.sleep(0.05)
            except Exception as e:
                errors += 1
                print(f"[rag/sync] Failed SKU {sku}: {e}")

        total_chroma = count_transcripts()
        return JSONResponse({
            "synced":          synced,
            "errors":          errors,
            "total_in_mongo":  total_mongo,
            "total_in_chroma": total_chroma,
            "message": f"Synced {synced} new SKUs. ChromaDB now has {total_chroma} transcripts.",
        })

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sync error: {exc}")
