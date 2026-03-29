"""
rag_routes.py
─────────────
RAG (Retrieval-Augmented Generation) pipeline as a FastAPI APIRouter.
Embeddings are stored in MongoDB Atlas (rag_embeddings collection) —
persistent across Render deploys, no local disk needed.

ENDPOINTS
═════════
  POST   /rag/save    – save a description as a transcript
  GET    /rag/search  – query: find best matching transcripts
  GET    /rag/count   – how many transcripts are stored
  POST   /rag/sync    – sync all MongoDB product descriptions → rag_embeddings
"""

import os
import asyncio
import datetime
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
IMG_BASE = "https://bgjewels.jewelscore.com/f/shortimage/"

# ── Router ─────────────────────────────────────────────────────────────────────
rag_router = APIRouter(prefix="/rag", tags=["RAG Pipeline"])


def _embed(text: str) -> list:
    """Encode text with OpenAI text-embedding-3-small → 1536-dim float list."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model="text-embedding-3-small", input=text.strip())
    return resp.data[0].embedding


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — SAVE
# ══════════════════════════════════════════════════════════════════════════════

@rag_router.post("/save", summary="Save a description as a RAG transcript")
async def rag_save(
    description: str = Form(...,      description="Jewellery description to store"),
    sku_name:    str = Form("",       description="Optional SKU (used as document ID)"),
    source:      str = Form("manual", description="Origin: manual | image_search | text_search"),
):
    """Save a jewellery description into MongoDB rag_embeddings."""
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
            "message": f"Description saved (id={doc_id}). Total transcripts: {total}.",
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
            "message": "No transcripts stored yet. Run bulk_describe.py or POST /rag/sync.",
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
            f"{total} transcript(s) stored in MongoDB rag_embeddings."
            if total > 0
            else "No transcripts yet. Run bulk_describe.py or POST /rag/sync."
        ),
    })


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 4 — SYNC (manual trigger)
# ══════════════════════════════════════════════════════════════════════════════

@rag_router.post("/sync", summary="Sync all MongoDB product descriptions → rag_embeddings")
async def rag_sync():
    """
    Pull all product descriptions from MongoDB catalog/product_details/product_list
    and upsert their embeddings into MongoDB rag_embeddings.
    Safe to call multiple times — only missing SKUs are added.
    """
    from mongo_db import get_db

    try:
        db = get_db()
        rag_col = db["rag_embeddings"]

        # Collect all described SKUs from MongoDB
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
            return JSONResponse({
                "synced": 0, "errors": 0, "total_in_rag": 0,
                "message": "No product descriptions found in MongoDB.",
            })

        # Find which SKUs are not yet in rag_embeddings
        existing_ids = set(d["_id"] for d in rag_col.find({}, {"_id": 1}))
        to_sync = {k: v for k, v in sku_desc_map.items() if k not in existing_ids}

        if not to_sync:
            return JSONResponse({
                "synced": 0, "errors": 0,
                "total_in_rag": count_transcripts(),
                "message": f"Already up to date. {len(existing_ids)} SKUs in rag_embeddings.",
            })

        client = OpenAI(api_key=OPENAI_API_KEY)
        synced = 0
        errors = 0

        for sku, doc in to_sync.items():
            try:
                desc = doc["description"].strip()
                resp = client.embeddings.create(model="text-embedding-3-small", input=desc)
                embedding = resp.data[0].embedding
                price = doc.get("show_catalog_price")
                pic   = doc.get("picture1", "")

                rag_col.replace_one(
                    {"_id": sku},
                    {
                        "_id":         sku,
                        "sku_name":    sku,
                        "description": desc,
                        "embedding":   embedding,
                        "item_name":   doc.get("item_name", ""),
                        "style":       doc.get("style_name", ""),
                        "metal":       doc.get("metal_code", ""),
                        "color":       doc.get("color_name", ""),
                        "price":       str(price) if price else "",
                        "picture1":    pic,
                        "image_url":   (IMG_BASE + pic) if pic else "",
                        "source":      "rag_sync",
                        "saved_at":    datetime.datetime.now(datetime.UTC).isoformat(),
                        "length":      len(desc),
                    },
                    upsert=True,
                )
                synced += 1
                await asyncio.sleep(0.05)
            except Exception as e:
                errors += 1
                print(f"[rag/sync] Failed SKU {sku}: {e}")

        total_rag = count_transcripts()
        return JSONResponse({
            "synced":         synced,
            "errors":         errors,
            "total_in_mongo": total_mongo,
            "total_in_rag":   total_rag,
            "message": f"Synced {synced} new SKUs. rag_embeddings now has {total_rag} transcripts.",
        })

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sync error: {exc}")
