"""
main.py
-------
FastAPI backend for BG Jewels Search.

Endpoints in use:
  POST /search          -- Upload image → GPT-4o Vision → FAISS (fallback)
  POST /search/describe -- Upload image → GPT-4o Vision → description text only
  POST /catalog/parse   -- Natural-language query → filter fields (GPT + regex)
  GET  /catalog/options -- Distinct filter values for dropdowns
  GET  /catalog/filter  -- Filter catalog with pagination
  POST /catalog/describe-from-url -- Image URL → GPT-4o → save description
  POST /catalog/update-description -- Manually update SKU description in MongoDB
  GET  /image/proxy     -- Server-side image proxy (hotlink protection bypass)
  POST /rag/save        -- Save description as ChromaDB transcript
  GET  /rag/search      -- Search ChromaDB transcripts by query
  GET  /rag/count       -- Count stored transcripts

Start:
  python -m uvicorn main:app --port 8001 --app-dir backend
"""

import os
import io
import json
import datetime
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional
from dotenv import load_dotenv
from rag_routes import rag_router
from mongo_db import get_db, col as mongo_col, safe

# ── Load environment variables from .env ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ── Paths ─────────────────────────────────────────────────────────────────────
INDEX_PATH = os.path.join(BASE_DIR, "index.faiss")
META_PATH  = os.path.join(BASE_DIR, "metadata.json")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# ── Image base URL (same as build_index.py) ───────────────────────────────────
IMG_BASE     = "https://bgjewels.jewelscore.com/f/shortimage/"
SKU_IMG_BASE = "https://sjadau.jewelscore.com/f/skuimage/"

# ── OpenAI embedding helper (replaces CLIP — no model download needed) ────────
def _embed_text(text: str) -> list:
    """Embed text using OpenAI text-embedding-3-small (1536-dim). No local model needed."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model="text-embedding-3-small", input=text.strip())
    return resp.data[0].embedding


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="BG Jewels Search API",
    description="Image search (CLIP image encoder) + Text search (CLIP text encoder) via FAISS",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount RAG pipeline router (/rag/save, /rag/search, /rag/count)
app.include_router(rag_router)


@app.on_event("startup")
async def startup_event():
    _ensure_search_history_table()


# ── Helper: describe image with OpenAI Vision ─────────────────────────────────
def _describe_image_with_openai(image_bytes: bytes) -> str:
    """Send image to OpenAI GPT-4o Vision and get a jewellery description."""
    import base64
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a jewellery expert. Describe this jewellery image in detail. "
                                "Include: type of jewellery (necklace, ring, earring, bracelet, etc.), "
                                "metal color (gold, silver, rose gold, etc.), "
                                "stones or gems visible (diamond, ruby, emerald, etc.), "
                                "design style (traditional, modern, polki, kundan, etc.), "
                                "and any other notable features. "
                                "Keep the description concise but informative (2-4 sentences)."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        description = response.choices[0].message.content.strip()
        print(f"[vision] Image described: {description[:80]}...")
        return description
    except Exception as e:
        print(f"[vision] OpenAI Vision failed: {e}")
        return ""


# ── POST /search  (image upload) ──────────────────────────────────────────────
@app.post("/search")
async def search_image(
    image: UploadFile = File(...),
    top_k: int = Query(default=8, ge=1, le=20),
):
    """
    Upload a jewellery photo.
    Pipeline:
      1. image → GPT-4o Vision → natural language description
      2. description → CLIP text encoder → FAISS search (against text-embedded catalog)
    This gives semantic description-based matching instead of pixel-level similarity.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image. Got: {image.content_type}")

    try:
        contents = await image.read()
        Image.open(io.BytesIO(contents)).convert("RGB")  # validate image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # ── Step 1: GPT-4o Vision → describe the uploaded image ──────────────────
    image_description = _describe_image_with_openai(contents)
    if not image_description:
        raise HTTPException(status_code=500, detail="Could not describe image. Check OpenAI API key.")

    print(f"[search/image] GPT-4o description: {image_description[:100]}")

    # ── Step 2: Embed description → search RAG (ChromaDB) ────────────────────
    from rag_pipeline import query_transcripts
    embedding = _embed_text(image_description)
    matches = query_transcripts(embedding, top_k=top_k)

    print(f"[search/image] Returned {len(matches)} RAG matches")

    _save_search_history(
        search_mode="image",
        description=image_description,
        matches=matches,
        top_k=top_k,
        image_filename=image.filename or "",
    )

    return JSONResponse({
        "query_processed": True,
        "search_mode": "image → GPT-4o Vision → OpenAI embedding → ChromaDB RAG",
        "image_description": image_description,
        "total_matches": len(matches),
        "matches": matches,
    })


# ── POST /search/describe  (image → GPT-4o description only, no FAISS) ───────
@app.post("/search/describe")
async def describe_image_only(image: UploadFile = File(...)):
    """Upload an image → GPT-4o Vision → return description text only.
    The frontend then uses this description as a RAG query."""
    contents = await image.read()
    description = _describe_image_with_openai(contents)
    if not description:
        raise HTTPException(status_code=500, detail="Could not describe image. Check OpenAI API key.")
    return {"description": description}


# ── MongoDB: no-op for search_history table (MongoDB creates collections automatically) ──
def _ensure_search_history_table():
    """MongoDB creates collections automatically — nothing to do."""
    try:
        db = get_db()
        # Ensure index on searched_at for efficient queries
        db["search_history"].create_index([("searched_at", -1)], background=True)
        print("[startup] MongoDB search_history collection ready.")
    except Exception as e:
        print(f"[startup] WARNING: Could not init search_history collection: {e}")


def _save_search_history(
    search_mode: str,
    description: str,
    matches: list,
    top_k: int,
    query_text: str = "",
    image_filename: str = "",
):
    """
    Persist one search event to MongoDB search_history collection AND update
    the description field on every matched SKU in catalog/product_details/product_list.
    """
    try:
        matched_sku_list = [
            m.get("sku_name", m.get("skuName", ""))
            for m in matches
            if m.get("sku_name") or m.get("skuName")
        ]

        db = get_db()

        # 1. Insert into search_history
        db["search_history"].insert_one({
            "searched_at":    datetime.datetime.utcnow(),
            "search_mode":    search_mode,
            "query_text":     query_text or None,
            "image_filename": image_filename or None,
            "description":    description or None,
            "top_k":          top_k,
            "total_matches":  len(matches),
            "matched_skus":   matched_sku_list,
        })

        # 2. Update description on matched SKUs (only if currently empty)
        if matched_sku_list and description:
            for sku in matched_sku_list:
                if not sku:
                    continue
                for coll_name in ("catalog", "product_details", "product_list"):
                    db[coll_name].update_many(
                        {"sku_name": sku, "$or": [{"description": {"$exists": False}}, {"description": None}, {"description": ""}]},
                        {"$set": {"description": description}},
                    )
            print(f"[search_history] Updated description for {len(matched_sku_list)} SKUs in MongoDB.")

        print(f"[search_history] Saved: mode={search_mode}, matches={len(matches)}, desc={description[:60]}...")
    except Exception as e:
        print(f"[search_history] WARNING: Could not save search history: {e}")


# ── GET /image/proxy ─────────────────────────────────────────────────────────
@app.get("/image/proxy")
async def image_proxy(url: str = Query(...)):
    """
    Server-side image proxy.
    Fetches an image from an external URL (e.g. sjadau.jewelscore.com which
    blocks direct browser requests via hotlink protection) and streams it back
    to the browser with the correct Referer header set server-side.

    Usage: /image/proxy?url=https://sjadau.jewelscore.com/f/skuimage/BGSNL-1018_1.jpg
    """
    import requests as req_lib
    import urllib.parse

    if not url:
        raise HTTPException(status_code=400, detail="url parameter is required.")

    # Only proxy known jewelscore domains for security
    allowed_hosts = ("sjadau.jewelscore.com", "bgjewels.jewelscore.com")
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc not in allowed_hosts:
            raise HTTPException(status_code=403, detail=f"Proxying not allowed for host: {parsed.netloc}")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid URL.")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer":    "https://bgjewels.jewelscore.com/",
            "Accept":     "image/webp,image/apng,image/*,*/*;q=0.8",
        }
        r = req_lib.get(url, headers=headers, timeout=15, stream=True)
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "image/jpeg")

        def iter_content():
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk

        return StreamingResponse(
            iter_content(),
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=86400",  # cache 24h in browser
                "Access-Control-Allow-Origin": "*",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch image: {e}")


# ── GET /catalog/options ──────────────────────────────────────────────────────
@app.get("/catalog/options")
async def catalog_options():
    """Return distinct values for filter dropdowns (from MongoDB catalog + product_details)."""
    try:
        db = get_db()

        def distinct_mongo(field, collection="catalog"):
            vals = db[collection].distinct(field)
            return sorted([v for v in vals if v and v != "-"])

        # Collect distinct stone attributes from stone_list field in product_details
        stone_names_set  = set()
        stone_shapes_set = set()
        stone_quals_set  = set()
        stone_sizes_set  = set()
        for doc in db["product_details"].find({"stone_list": {"$exists": True, "$ne": []}}, {"stone_list": 1}):
            for s in (doc.get("stone_list") or []):
                name  = (s.get("stonename") or s.get("stoneGroupName") or "").strip()
                shape = (s.get("shapeName") or "").strip()
                qual  = (s.get("qualityName") or "").strip()
                size  = (s.get("stoneSizeName") or "").strip()
                if name  and name  != "-": stone_names_set.add(name)
                if shape and shape != "-": stone_shapes_set.add(shape)
                if qual  and qual  != "-": stone_quals_set.add(qual)
                if size  and size  != "-": stone_sizes_set.add(size)

        result = {
            "itemNames":   distinct_mongo("item_name"),
            "styleNames":  distinct_mongo("style_name"),
            "metalCodes":  distinct_mongo("metal_code"),
            "colorNames":  distinct_mongo("color_name"),
            "stoneNames":  sorted(stone_names_set),
            "stoneShapes": sorted(stone_shapes_set),
            "stoneQuals":  sorted(stone_quals_set),
            "stoneSizes":  sorted(stone_sizes_set),
        }
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


# ── POST /catalog/parse ───────────────────────────────────────────────────────
@app.post("/catalog/parse")
async def catalog_parse(query: str = Form(...)):
    """
    Parse a natural-language query into catalog filter fields using OpenAI GPT.
    Falls back to regex matching if OpenAI fails.
    Returns: { item_name, style_name, metal_code, color_name, sku_name, min_price, max_price, in_stock }
    """
    import re
    import json as _json

    try:
        db = get_db()

        def distinct(field):
            vals = db["catalog"].distinct(field)
            return sorted([v for v in vals if v and v != "-"])

        item_names  = distinct("item_name")
        style_names = distinct("style_name")
        metal_codes = distinct("metal_code")
        color_names = distinct("color_name")

        # Get stone names from product_details stone_list
        stone_set2 = set()
        for doc in db["product_details"].find({"stone_list": {"$exists": True, "$ne": []}}, {"stone_list": 1}):
            for s in (doc.get("stone_list") or []):
                n = (s.get("stonename") or s.get("stoneGroupName") or "").strip()
                if n and n != "-": stone_set2.add(n)
        stone_names = sorted(stone_set2)

        # ── Try OpenAI GPT first ──────────────────────────────────────────────
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            system_prompt = f"""You are a jewellery catalog filter assistant.
Given a user query, extract filter values ONLY from the lists provided below.
Return a JSON object with these exact keys (use empty string "" or null if not found):
  item_name, style_name, metal_code, color_name, sku_name, min_price, max_price, in_stock

Rules:
- item_name must be one of: {item_names}
- style_name must be one of: {style_names}
- metal_code must be one of: {metal_codes} (also map: silver→SS, gold→YG, yellow gold→YG, white gold→WG, rose gold→RG, platinum→PT)
- color_name must be one of: {color_names}
- sku_name: extract SKU codes like BGSER-1000 or KR1086 (string, else "")
- min_price / max_price: numbers or null
- in_stock: true if user mentions "in stock", "available", "stock", else false
- For style_name, do fuzzy matching — e.g. "pulki jewellery" → match closest style from the list
- Return ONLY valid JSON, no explanation."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            parsed = _json.loads(raw)

            # Validate values against DB lists
            def validate(val, allowed):
                if not val:
                    return ""
                val_str = str(val).strip()
                # Treat "-" or "none" or "n/a" as empty (not a real filter)
                if val_str in ("-", "none", "n/a", "N/A", "None", "null"):
                    return ""
                if val_str in allowed:
                    return val_str
                # Case-insensitive fallback
                for a in allowed:
                    if a.lower() == val_str.lower():
                        return a
                return ""

            # Extract stone attributes via regex from query (GPT doesn't know stone lists)
            q_lower = query.lower()

            stone_name_gpt = ""
            for sn in stone_names:
                if sn.lower() in q_lower:
                    stone_name_gpt = sn
                    break

            # Get all stone shapes/qualities/sizes from MongoDB for matching
            shape_set2, qual_set2, size_set2 = set(), set(), set()
            for doc2 in db["product_details"].find({"stone_list": {"$exists": True, "$ne": []}}, {"stone_list": 1}):
                for s2 in (doc2.get("stone_list") or []):
                    sh = (s2.get("shapeName") or "").strip()
                    qu = (s2.get("qualityName") or "").strip()
                    si = (s2.get("stoneSizeName") or "").strip()
                    if sh and sh != "-": shape_set2.add(sh)
                    if qu and qu != "-": qual_set2.add(qu)
                    if si and si != "-": size_set2.add(si)

            stone_shape_gpt = ""
            for sh in sorted(shape_set2, key=len, reverse=True):
                if sh.lower() in q_lower:
                    stone_shape_gpt = sh
                    break

            stone_quality_gpt = ""
            for qu in sorted(qual_set2, key=len, reverse=True):
                if qu.lower() in q_lower:
                    stone_quality_gpt = qu
                    break

            stone_size_gpt = ""
            for si in sorted(size_set2, key=len, reverse=True):
                if si.lower() in q_lower:
                    stone_size_gpt = si
                    break

            return JSONResponse({
                "item_name":    validate(parsed.get("item_name", ""),  item_names),
                "style_name":   validate(parsed.get("style_name", ""), style_names),
                "metal_code":   validate(parsed.get("metal_code", ""), metal_codes),
                "color_name":   validate(parsed.get("color_name", ""), color_names),
                "sku_name":     str(parsed.get("sku_name", "") or ""),
                "stone_name":   stone_name_gpt,
                "stone_shape":  stone_shape_gpt,
                "stone_quality": stone_quality_gpt,
                "stone_size":   stone_size_gpt,
                "min_price":    float(parsed["min_price"]) if parsed.get("min_price") not in (None, "", "null") else None,
                "max_price":    float(parsed["max_price"]) if parsed.get("max_price") not in (None, "", "null") else None,
                "in_stock":     bool(parsed.get("in_stock", False)),
                "source":       "gpt",
            })

        except Exception as gpt_err:
            print(f"[catalog/parse] GPT failed ({gpt_err}), falling back to regex")

        # ── Regex fallback ────────────────────────────────────────────────────
        q = query.lower()

        # Match item name — word boundary, handle plurals
        item_name = ""
        for name in sorted(item_names, key=len, reverse=True):
            n = name.lower()
            if (re.search(r'\b' + re.escape(n) + r's?\b', q) or
                re.search(r'\b' + re.escape(n.rstrip('s')) + r's?\b', q)):
                item_name = name
                break

        # Match style name — word boundary
        style_name = ""
        for s in style_names:
            sl = s.lower()
            if re.search(r'\b' + re.escape(sl) + r'\b', q):
                style_name = s
                break

        # Match metal code
        metal_code = ""
        metal_map = [
            ("yellow gold", "YG"), ("white gold", "WG"), ("rose gold", "RG"),
            ("silver", "SS"), ("platinum", "PT"), ("gold", "YG"),
        ]
        for keyword, code in metal_map:
            if re.search(r'\b' + re.escape(keyword) + r'\b', q):
                metal_code = code
                break
        if not metal_code:
            for code in metal_codes:
                if re.search(r'\b' + re.escape(code.lower()) + r'\b', q):
                    metal_code = code
                    break

        # Match color name — word boundary
        color_name = ""
        for c in color_names:
            if not c or c == '-':
                continue
            if re.search(r'\b' + re.escape(c.lower()) + r'\b', q):
                color_name = c
                break

        # SKU pattern
        sku_match = re.search(r'\b([A-Z]{2,6}[-/]?\d{3,6})\b', query, re.IGNORECASE)
        sku_name = sku_match.group(1) if sku_match else ""

        # Price range
        min_price = None
        max_price = None
        under = re.search(r'(?:under|below|less than|max|upto|up to)\s*(?:rs\.?|₹)?\s*(\d+)', q)
        if under:
            max_price = float(under.group(1))
        above = re.search(r'(?:above|over|more than|min|minimum|from)\s*(?:rs\.?|₹)?\s*(\d+)', q)
        if above:
            min_price = float(above.group(1))
        between = re.search(r'between\s*(?:rs\.?|₹)?\s*(\d+)\s*(?:and|to|-)\s*(?:rs\.?|₹)?\s*(\d+)', q)
        if between:
            min_price = float(between.group(1))
            max_price = float(between.group(2))

        in_stock = any(w in q for w in ["in stock", "available", "stock", "instock"])

        return JSONResponse({
            "item_name":  item_name,
            "style_name": style_name,
            "metal_code": metal_code,
            "color_name": color_name,
            "sku_name":   sku_name,
            "min_price":  min_price,
            "max_price":  max_price,
            "in_stock":   in_stock,
            "source":     "regex",
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse error: {e}")


# ── POST /catalog/describe-from-url ──────────────────────────────────────────
@app.post("/catalog/describe-from-url")
async def describe_from_url(
    image_url: str = Form(...),
    sku_name:  str = Form(...),
):
    """
    Given an image URL and a SKU name:
      1. Download the image from the URL
      2. Send it to GPT-4o Vision to generate a jewellery description
      3. Save the description to catalog, product_details, product_list for the matching SKU(s)
      4. Return the generated description + matched SKUs
    """
    import requests as req_lib

    if not image_url.strip():
        raise HTTPException(status_code=400, detail="image_url cannot be empty.")
    if not sku_name.strip():
        raise HTTPException(status_code=400, detail="sku_name cannot be empty.")

    # ── Step 1: Download image from URL ──────────────────────────────────────
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://bgjewels.jewelscore.com/",
        }
        r = req_lib.get(image_url.strip(), headers=headers, timeout=15)
        r.raise_for_status()
        image_bytes = r.content
        # Validate it's an image
        Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not download/read image from URL: {e}")

    # ── Step 2: GPT-4o Vision → generate description ──────────────────────────
    description = _describe_image_with_openai(image_bytes)
    if not description:
        raise HTTPException(status_code=500, detail="GPT-4o Vision could not generate a description. Check OpenAI API key.")

    print(f"[describe-from-url] SKU='{sku_name}' URL='{image_url}' → desc: {description[:80]}...")

    # ── Step 3: Save description to MongoDB for matching SKUs ─────────────────
    try:
        db = get_db()
        pattern = {"sku_name": {"$regex": sku_name.strip(), "$options": "i"}}
        matched_docs = list(db["catalog"].find(pattern, {"sku_name": 1}))
        matched_skus = [d["sku_name"] for d in matched_docs]

        updated_count = 0
        for sku in matched_skus:
            for coll_name in ("catalog", "product_details", "product_list"):
                res = db[coll_name].update_many({"sku_name": sku}, {"$set": {"description": description}})
                updated_count += res.modified_count

        print(f"[describe-from-url] Matched {len(matched_skus)} SKUs, updated {updated_count} docs.")
        return JSONResponse({
            "description":   description,
            "matched_skus":  matched_skus,
            "updated_count": updated_count,
            "message": (
                f"Generated description and updated {len(matched_skus)} SKU(s)."
                if matched_skus
                else f"Description generated but no SKUs found matching '{sku_name}'."
            ),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


# ── POST /catalog/update-description ─────────────────────────────────────────
@app.post("/catalog/update-description")
async def update_description(
    sku_name:    str = Form(...),
    description: str = Form(...),
):
    """
    Search for products whose sku_name contains the given search term (LIKE %sku_name%),
    then update the description column in catalog, product_details, and product_list
    for every matching SKU.

    Returns the list of matched SKUs and how many rows were updated.
    """
    if not sku_name.strip():
        raise HTTPException(status_code=400, detail="sku_name cannot be empty.")
    if not description.strip():
        raise HTTPException(status_code=400, detail="description cannot be empty.")

    try:
        db = get_db()
        pattern = {"sku_name": {"$regex": sku_name.strip(), "$options": "i"}}
        matched_docs = list(db["catalog"].find(pattern, {"sku_name": 1}))
        matched_skus = [d["sku_name"] for d in matched_docs]

        if not matched_skus:
            return JSONResponse({
                "matched_skus": [],
                "updated_count": 0,
                "message": f"No SKUs found matching '{sku_name}'.",
            })

        updated_count = 0
        for sku in matched_skus:
            for coll_name in ("catalog", "product_details", "product_list"):
                res = db[coll_name].update_many({"sku_name": sku}, {"$set": {"description": description.strip()}})
                updated_count += res.modified_count

        print(f"[update-description] SKU search='{sku_name}' → matched {len(matched_skus)} SKUs, updated {updated_count} docs.")
        return JSONResponse({
            "matched_skus":  matched_skus,
            "updated_count": updated_count,
            "message":       f"Updated description for {len(matched_skus)} SKU(s) across catalog, product_details, and product_list.",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


# ── GET /catalog/filter ───────────────────────────────────────────────────────
@app.get("/catalog/filter")
async def catalog_filter(
    item_name:  Optional[str] = Query(default=None),
    style_name: Optional[str] = Query(default=None),
    metal_code: Optional[str] = Query(default=None),
    color_name: Optional[str] = Query(default=None),
    sku_name:   Optional[str] = Query(default=None),
    stone_name:    Optional[str] = Query(default=None),
    stone_shape:   Optional[str] = Query(default=None),
    stone_quality: Optional[str] = Query(default=None),
    stone_size:    Optional[str] = Query(default=None),
    min_price:  Optional[float] = Query(default=None),
    max_price:  Optional[float] = Query(default=None),
    in_stock:   Optional[bool]  = Query(default=None),
    page:       int = Query(default=1, ge=1),
    page_size:  int = Query(default=50, ge=1, le=200),
):
    """
    Filter catalog items joined with product_details and product_list.
    Supports stone_name filtering via stone_list_json.
    """
    import json as _json

    def safe(v):
        if v is None: return None
        try: return float(v)
        except Exception: return v

    try:
        db = get_db()

        # Build MongoDB filter query
        mongo_filter = {}
        if item_name:
            mongo_filter["item_name"] = item_name
        if style_name:
            mongo_filter["style_name"] = style_name
        if metal_code:
            mongo_filter["metal_code"] = metal_code
        if color_name:
            mongo_filter["color_name"] = color_name
        if sku_name:
            mongo_filter["sku_name"] = {"$regex": sku_name, "$options": "i"}
        if min_price is not None or max_price is not None:
            price_q = {}
            if min_price is not None:
                price_q["$gte"] = min_price
            if max_price is not None:
                price_q["$lte"] = max_price
            mongo_filter["show_catalog_price"] = price_q
        if in_stock:
            mongo_filter["stock_qty"] = {"$gt": 0}

        # Fetch all matching catalog docs
        catalog_docs = list(db["catalog"].find(mongo_filter).sort("skuid", 1))

        # Build lookup maps from product_details and product_list
        sku_list = [d["sku_name"] for d in catalog_docs if d.get("sku_name")]
        pd_map = {d["sku_name"]: d for d in db["product_details"].find({"sku_name": {"$in": sku_list}})}
        pl_map = {d["sku_name"]: d for d in db["product_list"].find({"sku_name": {"$in": sku_list}})}

        # Apply stone filters
        def stone_matches(pd_doc, name_f, shape_f, qual_f, size_f):
            if not any([name_f, shape_f, qual_f, size_f]):
                return True
            stones = pd_doc.get("stone_list") or []
            for s in stones:
                sname  = (s.get("stonename") or s.get("stoneGroupName") or "").lower()
                sshape = (s.get("shapeName") or "").lower()
                squal  = (s.get("qualityName") or "").lower()
                ssize  = (s.get("stoneSizeName") or "").lower()
                if name_f  and name_f.lower()  not in sname:  continue
                if shape_f and shape_f.lower() not in sshape: continue
                if qual_f  and qual_f.lower()  not in squal:  continue
                if size_f  and size_f.lower()  not in ssize:  continue
                return True
            return False

        if stone_name or stone_shape or stone_quality or stone_size:
            catalog_docs = [
                d for d in catalog_docs
                if stone_matches(pd_map.get(d.get("sku_name"), {}), stone_name, stone_shape, stone_quality, stone_size)
            ]

        total = len(catalog_docs)
        offset = (page - 1) * page_size
        page_docs = catalog_docs[offset: offset + page_size]

        items = []
        for c in page_docs:
            sku = c.get("sku_name", "")
            pd  = pd_map.get(sku, {})
            pl  = pl_map.get(sku, {})

            # description: prefer catalog → product_details → product_list
            description = (
                c.get("description") or
                pd.get("description") or
                pl.get("description") or ""
            )

            # Parse stone list
            stones = []
            for s in (pd.get("stone_list") or []):
                name = (s.get("stonename") or s.get("stoneGroupName") or "").strip()
                if name and name != "-":
                    stones.append({
                        "name":    name,
                        "shape":   s.get("shapeName", ""),
                        "quality": s.get("qualityName", ""),
                        "pcs":     s.get("pcs", 0),
                        "wt":      s.get("weight", 0),
                    })

            items.append({
                "id":               str(c.get("_id", "")),
                "skuid":            c.get("skuid", ""),
                "skuName":          sku,
                "readyId":          c.get("ready_id", ""),
                "metalCode":        c.get("metal_code", ""),
                "itemName":         c.get("item_name", ""),
                "itemGroupName":    c.get("item_group_name", ""),
                "styleName":        c.get("style_name", ""),
                "colorName":        c.get("color_name", ""),
                "sGroupName":       c.get("s_group_name", ""),
                "subGroupName":     pd.get("sub_group_name", ""),
                "size":             c.get("size", ""),
                "grossWt":          safe(c.get("gross_wt")),
                "weight":           safe(c.get("weight")),
                "fineWt":           safe(pd.get("fine_wt")),
                "diaWt":            safe(c.get("dia_wt")),
                "csWt":             safe(c.get("cs_wt")),
                "diaPcs":           c.get("dia_pcs", 0),
                "csPcs":            c.get("cs_pcs", 0),
                "stockQty":         c.get("stock_qty", 0),
                "picture1":         c.get("picture1", ""),
                "image_url":        (IMG_BASE + c["picture1"]) if c.get("picture1") else "",
                "skuImage1":        pd.get("sku_image1", "") or "",
                "skuImage2":        pd.get("sku_image2", "") or "",
                "currencySymbol":   c.get("currency_symbol", ""),
                "showCatalogPrice": safe(c.get("show_catalog_price")),
                "pricePerPcs":      safe(pd.get("price_per_pcs")),
                "tagPrice":         safe(pd.get("tag_price")),
                "stoneAmt":         safe(pd.get("stone_amt")),
                "metalAmt":         safe(pd.get("metal_amt")),
                "labour":           safe(pd.get("labour")),
                "hidePrice":        bool(c.get("hide_price", False)),
                "priceType":        c.get("price_type", ""),
                "refDNo":           c.get("ref_d_no", ""),
                "length":           c.get("length", ""),
                "height":           c.get("height", ""),
                "width":            c.get("width", ""),
                "itemDesc":         pd.get("item_desc", "") or "",
                "description":      description,
                "otherCode":        pd.get("other_code", "") or "",
                "stones":           stones,
            })

        return JSONResponse({
            "total":    total,
            "page":     page,
            "pageSize": page_size,
            "pages":    max(1, (total + page_size - 1) // page_size),
            "items":    items,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
