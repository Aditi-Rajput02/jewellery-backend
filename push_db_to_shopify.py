# -*- coding: utf-8 -*-
"""
MongoDB -> Shopify Product Sync  (bgluxe-official.myshopify.com)
================================================================
Reads product data from MongoDB Atlas (catalog + product_details + product_list)
and pushes each product to Shopify via the Admin REST API.

HOW TO GET A NEW SHOPIFY ACCESS TOKEN (if you get 401 errors):
---------------------------------------------------------------
1. Go to: https://bgluxe-official.myshopify.com/admin
2. Click "Settings" (bottom-left gear icon)
3. Click "Apps and sales channels"
4. Click "Develop apps" button (top-right)
5. If prompted, click "Allow custom app development"
6. Click "Create an app" -> name it "BG Jewels Sync"
7. Click "Configure Admin API scopes"
8. Enable: write_products, read_products
9. Click "Save" -> then "Install app"
10. Click "Reveal token once" -> copy the shpat_... token
11. Replace SHOPIFY_TOKEN below with the new token

Features:
  - Reads ALL products from MongoDB Atlas
  - Joins catalog + product_details + product_list collections
  - Skips products already in Shopify (matched by SKU)
  - Maps all MongoDB fields to Shopify product fields
  - Attaches product images from BG Jewels CDN
  - Saves a sync report to sync_report.json
  - Respects Shopify rate limits (2 req/sec)
"""

import sys
import time
import json
import requests
from datetime import datetime

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------
SHOPIFY_SHOP  = "bgluxe-official.myshopify.com"
SHOPIFY_TOKEN = "SHOPIFY_SECRET_REMOVED"
SHOPIFY_API   = "2026-04"

SYNC_REPORT_FILE = "sync_report.json"

# BG Jewels image base URLs
IMG_BASE     = "https://bgjewels.jewelscore.com/f/shortimage/"
SKU_IMG_BASE = "https://sjadau.jewelscore.com/f/skuimage/"

SHOPIFY_HEADERS = {
    "X-Shopify-Access-Token": SHOPIFY_TOKEN,
    "Content-Type": "application/json",
}
SHOPIFY_PRODUCTS_URL = f"https://{SHOPIFY_SHOP}/admin/api/{SHOPIFY_API}/products.json"


# -----------------------------------------------
# MongoDB connection
# -----------------------------------------------
def get_mongo_products():
    """
    Fetch all products from MongoDB Atlas by joining:
      catalog + product_details + product_list
    Returns a list of merged product dicts.
    """
    import os
    from pymongo import MongoClient
    from dotenv import load_dotenv

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(BASE_DIR, ".env"))

    MONGO_URI = os.getenv(
        "MONGO_URI",
        "MONGO_URI_REMOVED?appName=Cluster0"
    )
    MONGO_DB = os.getenv("MONGO_DATABASE", "jewellery")

    print("  Connecting to MongoDB Atlas ...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=15000)
    db = client[MONGO_DB]

    # Fetch all catalog docs
    catalog_docs = list(db["catalog"].find({}))
    print(f"  catalog      : {len(catalog_docs)} documents")

    # Build lookup maps for product_details and product_list by sku_name
    sku_list = [d.get("sku_name", "") for d in catalog_docs if d.get("sku_name")]
    pd_map = {d["sku_name"]: d for d in db["product_details"].find({"sku_name": {"$in": sku_list}})}
    pl_map = {d["sku_name"]: d for d in db["product_list"].find({"sku_name": {"$in": sku_list}})}
    print(f"  product_details: {len(pd_map)} documents")
    print(f"  product_list   : {len(pl_map)} documents")

    products = []
    for c in catalog_docs:
        sku = c.get("sku_name", "")
        if not sku:
            continue
        pd = pd_map.get(sku, {})
        pl = pl_map.get(sku, {})

        # Merge all fields into one flat dict (catalog fields take priority)
        merged = {
            "skuName":       sku,
            "skuid":         c.get("skuid", ""),
            "metalCode":     c.get("metal_code", ""),
            "itemName":      c.get("item_name", "Jewellery"),
            "itemGroupName": c.get("item_group_name", ""),
            "styleName":     c.get("style_name", ""),
            "colorName":     c.get("color_name", ""),
            "sGroupName":    c.get("s_group_name", ""),
            "subGroupName":  pd.get("sub_group_name", ""),
            "size":          c.get("size", ""),
            "grossWt":       c.get("gross_wt") or c.get("weight") or 0.0,
            "weight":        c.get("weight") or 0.0,
            "stockQty":      c.get("stock_qty", 0),
            "picture1":      c.get("picture1", ""),
            "skuImage1":     pd.get("sku_image1", "") or "",
            "skuImage2":     pd.get("sku_image2", "") or "",
            "skuImage3":     pd.get("sku_image3", "") or "",
            "skuImage4":     pd.get("sku_image4", "") or "",
            "showCatalogPrice": c.get("show_catalog_price") or 0.0,
            "pricePerPcs":   pd.get("price_per_pcs") or 0.0,
            "tagPrice":      pd.get("tag_price") or 0.0,
            "wsPrice":       pl.get("ws_price") or pl.get("wsPrice") or 0.0,
            "itemDesc":      pd.get("item_desc", "") or "",
            "description":   (
                c.get("description") or
                pd.get("description") or
                pl.get("description") or ""
            ),
            "otherCode":     pd.get("other_code", "") or "",
            "baseImagePath": IMG_BASE,
        }
        products.append(merged)

    client.close()
    return products


# -----------------------------------------------
# Helper: get all existing Shopify SKUs
# -----------------------------------------------
def get_existing_shopify_skus():
    """Fetch all existing product SKUs from Shopify to avoid duplicates."""
    skus = set()
    url = SHOPIFY_PRODUCTS_URL
    params = {"limit": 250, "fields": "id,variants"}
    while url:
        res = requests.get(url, headers=SHOPIFY_HEADERS, params=params, timeout=30)
        if res.status_code == 401:
            print("\n  ERROR 401 - Unauthorized!")
            print("  The SHOPIFY_TOKEN is invalid for this store.")
            print("  Please follow the instructions at the top of this file to get a new token.")
            sys.exit(1)
        res.raise_for_status()
        products = res.json().get("products", [])
        for p in products:
            for v in p.get("variants", []):
                if v.get("sku"):
                    skus.add(v["sku"])
        # Pagination via Link header
        link = res.headers.get("Link", "")
        url = None
        params = {}
        if 'rel="next"' in link:
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip().strip("<>")
                    break
    return skus


# -----------------------------------------------
# Helper: clean a tag string
# -----------------------------------------------
def clean_tag(t):
    """Strip surrounding quotes and problematic characters from a tag."""
    if not t:
        return ""
    t = str(t).strip().strip("'\"")
    t = t.replace("&", "and")
    t = t.strip()
    return t


# -----------------------------------------------
# Helper: validate that a CDN image URL is a real image
# -----------------------------------------------
def is_valid_image_url(url):
    """Returns True only if the URL serves a real image (not an HTML error page)."""
    try:
        head = requests.head(url, timeout=8, allow_redirects=True)
        ct = head.headers.get("Content-Type", "")
        if ct.startswith("image/"):
            return True
        if "text/html" in ct:
            return False
        # Fallback: check JPEG magic bytes (FF D8 FF)
        r = requests.get(url, timeout=10, stream=True)
        chunk = next(r.iter_content(16), b"")
        r.close()
        return chunk[:3] == b"\xff\xd8\xff"
    except Exception:
        return False


# -----------------------------------------------
# Helper: build Shopify product payload from MongoDB merged doc
# -----------------------------------------------
def build_shopify_payload(item):
    sku_name     = item.get("skuName", "")
    title        = item.get("itemDesc") or item.get("skuName", "Jewellery")
    description  = item.get("description", "") or ""
    vendor       = "BG Jewels"
    product_type = item.get("itemName", "Jewellery")

    # Build clean tags from available fields
    raw_tags = []
    sub = (item.get("subGroupName", "") or "").strip().strip("'\"")
    for t in sub.split(","):
        t = clean_tag(t)
        if t and t != "-":
            raw_tags.append(t)
    for field in ["colorName", "styleName", "itemName", "metalCode", "sGroupName"]:
        val = clean_tag(item.get(field) or "")
        if val and val != "-":
            raw_tags.append(val)

    # Remove duplicates, keep order
    seen = set()
    clean_tags = []
    for t in raw_tags:
        if t not in seen:
            seen.add(t)
            clean_tags.append(t)
    tags = ", ".join(clean_tags)

    # Price: prefer tagPrice -> showCatalogPrice -> pricePerPcs -> wsPrice
    price = (
        item.get("tagPrice") or
        item.get("showCatalogPrice") or
        item.get("pricePerPcs") or
        item.get("wsPrice") or
        0.0
    )
    try:
        price = round(float(price), 2)
    except Exception:
        price = 0.0

    # Weight in grams
    try:
        weight_g = float(item.get("grossWt") or item.get("weight") or 0.0)
    except Exception:
        weight_g = 0.0

    # Stock quantity
    try:
        stock_qty = int(item.get("stockQty") or 1)
        if stock_qty < 1:
            stock_qty = 1
    except Exception:
        stock_qty = 1

    # Images — try picture1 (shortimage) first, then skuImage1-4 (skuimage CDN)
    images = []

    # picture1 from shortimage CDN
    pic1 = (item.get("picture1") or "").strip()
    if pic1:
        src = IMG_BASE + pic1
        if is_valid_image_url(src):
            images.append({"src": src, "alt": title})

    # skuImage1-4 from skuimage CDN
    for img_field in ["skuImage1", "skuImage2", "skuImage3", "skuImage4"]:
        img_name = (item.get(img_field) or "").strip()
        if img_name:
            src = SKU_IMG_BASE + img_name
            if is_valid_image_url(src):
                images.append({"src": src, "alt": title})

    payload = {
        "product": {
            "title":        title,
            "body_html":    description,
            "vendor":       vendor,
            "product_type": product_type,
            "tags":         tags,
            "status":       "active",
            "variants": [
                {
                    "sku":                  sku_name,
                    "price":               str(price),
                    "inventory_management": "shopify",
                    "inventory_quantity":   stock_qty,
                    "weight":              weight_g,
                    "weight_unit":         "g",
                    "requires_shipping":   True,
                    "taxable":             True,
                }
            ],
            "images": images,
        }
    }
    return payload


# -----------------------------------------------
# Main sync function
# -----------------------------------------------
def main():
    print("=" * 65)
    print("  MongoDB -> Shopify Product Sync")
    print("  Shop    : " + SHOPIFY_SHOP)
    print("  Started : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 65)

    # Step 1: Load products from MongoDB
    print("\n[1] Loading products from MongoDB Atlas ...")
    try:
        products = get_mongo_products()
    except Exception as e:
        print(f"\n  ERROR connecting to MongoDB: {e}")
        sys.exit(1)
    print(f"    OK: {len(products)} products loaded from MongoDB.")

    if not products:
        print("  No products found in MongoDB. Exiting.")
        sys.exit(0)

    # Step 2: Fetch existing Shopify SKUs
    print("\n[2] Fetching existing Shopify SKUs (to skip duplicates) ...")
    try:
        existing_skus = get_existing_shopify_skus()
    except Exception as e:
        print(f"\n  ERROR connecting to Shopify: {e}")
        print("  Check your SHOPIFY_TOKEN and SHOPIFY_SHOP values.")
        sys.exit(1)
    print(f"    OK: {len(existing_skus)} SKUs already in Shopify.")

    # Step 3: Sync loop
    total   = len(products)
    created = 0
    skipped = 0
    errors  = 0
    report  = []

    print(f"\n[3] Syncing {total} products to Shopify ...")
    print("    (Image validation may take a moment per product)\n")

    for i, item in enumerate(products, 1):
        sku   = item.get("skuName", f"ITEM-{i}")
        title = item.get("itemDesc") or sku

        # Skip if already exists in Shopify
        if sku in existing_skus:
            print(f"  [{str(i).rjust(4)}/{total}] SKIP     {sku} (already in Shopify)")
            skipped += 1
            report.append({"sku": sku, "status": "skipped", "reason": "already exists"})
            continue

        payload = build_shopify_payload(item)

        try:
            res = requests.post(
                SHOPIFY_PRODUCTS_URL,
                headers=SHOPIFY_HEADERS,
                json=payload,
                timeout=30,
            )

            if res.status_code == 201:
                shopify_id = res.json()["product"]["id"]
                print(f"  [{str(i).rjust(4)}/{total}] CREATED  {sku} -> Shopify ID {shopify_id}")
                created += 1
                report.append({"sku": sku, "status": "created", "shopify_id": shopify_id})
                existing_skus.add(sku)

            elif res.status_code == 429:
                print(f"  [{str(i).rjust(4)}/{total}] RATE LIMITED, waiting 15s ...")
                time.sleep(15)
                res = requests.post(SHOPIFY_PRODUCTS_URL, headers=SHOPIFY_HEADERS, json=payload, timeout=30)
                if res.status_code == 201:
                    shopify_id = res.json()["product"]["id"]
                    print(f"  [{str(i).rjust(4)}/{total}] CREATED (retry) {sku} -> {shopify_id}")
                    created += 1
                    report.append({"sku": sku, "status": "created", "shopify_id": shopify_id})
                    existing_skus.add(sku)
                else:
                    print(f"  [{str(i).rjust(4)}/{total}] FAILED   {sku} -> {res.status_code}: {res.text[:200]}")
                    errors += 1
                    report.append({"sku": sku, "status": "error", "code": res.status_code, "msg": res.text[:200]})

            elif res.status_code == 401:
                print(f"\n  ERROR 401 - Unauthorized! Token is invalid for {SHOPIFY_SHOP}.")
                print("  Please generate a new token. See instructions at the top of this file.")
                sys.exit(1)

            else:
                print(f"  [{str(i).rjust(4)}/{total}] FAILED   {sku} -> {res.status_code}: {res.text[:200]}")
                errors += 1
                report.append({"sku": sku, "status": "error", "code": res.status_code, "msg": res.text[:200]})

        except requests.exceptions.RequestException as e:
            print(f"  [{str(i).rjust(4)}/{total}] ERROR    {sku} -> {e}")
            errors += 1
            report.append({"sku": sku, "status": "error", "msg": str(e)})

        # Shopify rate limit: ~2 req/sec on Basic plan
        time.sleep(0.6)

    # Save report
    summary = {
        "synced_at": datetime.now().isoformat(),
        "shop":      SHOPIFY_SHOP,
        "total":     total,
        "created":   created,
        "skipped":   skipped,
        "errors":    errors,
        "results":   report,
    }
    with open(SYNC_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("  SYNC COMPLETE")
    print(f"  Shop    : {SHOPIFY_SHOP}")
    print(f"  Total   : {total}")
    print(f"  Created : {created}")
    print(f"  Skipped : {skipped} (already existed)")
    print(f"  Errors  : {errors}")
    print(f"  Report  : {SYNC_REPORT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()


