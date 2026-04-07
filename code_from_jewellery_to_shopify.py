# -*- coding: utf-8 -*-
"""
BG Jewels -> Shopify Product Sync
==================================
Reads product data from product_list.json (already fetched from BG Jewels API)
and pushes each product to Shopify via the Admin REST API.

Features:
  - Skips products already in Shopify (matched by SKU)
  - Maps all BG Jewels fields to Shopify product fields
  - Attaches product images from BG Jewels CDN
  - Saves a sync report to sync_report.json
  - Respects Shopify rate limits (2 req/sec)
"""

import json
import time
import sys
import requests
from datetime import datetime

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -----------------------------------------------
# Configuration
# -----------------------------------------------
SHOPIFY_SHOP  = "aisolv-2.myshopify.com"
SHOPIFY_TOKEN = "SHOPIFY_TOKEN_REMOVED"
SHOPIFY_API   = "2024-01"

PRODUCT_LIST_FILE = "product_list.json"
SYNC_REPORT_FILE  = "sync_report.json"

# BG Jewels image base URL
BG_IMAGE_BASE = "https://sjadau.jewelscore.com/f/skuimage/"

SHOPIFY_HEADERS = {
    "X-Shopify-Access-Token": SHOPIFY_TOKEN,
    "Content-Type": "application/json",
}

SHOPIFY_PRODUCTS_URL = f"https://{SHOPIFY_SHOP}/admin/api/{SHOPIFY_API}/products.json"


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
# Helper: clean a tag string
# -----------------------------------------------
def clean_tag(t):
    """Strip surrounding quotes and problematic characters from a tag."""
    t = t.strip().strip("'\"")   # remove surrounding single/double quotes
    t = t.replace("&", "and")    # & is not allowed in Shopify tags
    t = t.strip()
    return t


# -----------------------------------------------
# Helper: map BG Jewels product to Shopify payload
# -----------------------------------------------
def build_shopify_payload(item):
    sku_name     = item.get("skuName", "")
    title        = item.get("itemDesc") or item.get("skuName", "Jewellery")
    description  = item.get("description", "") or ""
    vendor       = "BG Jewels"
    product_type = item.get("itemName", "Jewellery")

    # Build clean tags
    raw_tags = []
    sub = (item.get("subGroupName", "") or "").strip().strip("'\"")
    for t in sub.split(","):
        t = clean_tag(t)
        if t and t != "-":
            raw_tags.append(t)
    for field in ["colorName", "stylename", "itemName"]:
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

    # Price
    price = item.get("tagPrice") or item.get("wsPrice") or item.get("pricePerPcs") or 0.0
    price = round(float(price), 2)

    # Weight in grams
    weight_g = float(item.get("grossWt") or item.get("weight") or 0.0)

    # Images - validate each URL before including (skip HTML error pages)
    images = []
    base = item.get("baseImagePath", BG_IMAGE_BASE)
    for img_field in ["picture1", "skuImage1", "skuImage2", "skuImage3", "skuImage4"]:
        img_name = (item.get(img_field) or "").strip()
        if img_name:
            src = base + img_name
            if is_valid_image_url(src):
                images.append({"src": src, "alt": title})

    payload = {
        "product": {
            "title": title,
            "body_html": description,
            "vendor": vendor,
            "product_type": product_type,
            "tags": tags,
            "status": "active",
            "variants": [
                {
                    "sku": sku_name,
                    "price": str(price),
                    "inventory_management": "shopify",
                    "inventory_quantity": 1,
                    "weight": weight_g,
                    "weight_unit": "g",
                    "requires_shipping": True,
                    "taxable": True,
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
    print("  BG Jewels -> Shopify Product Sync")
    print("  Started : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 65)

    # Load local product list
    print("\n[1] Loading products from '" + PRODUCT_LIST_FILE + "' ...")
    with open(PRODUCT_LIST_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        products = raw
    elif isinstance(raw, dict):
        products = raw.get("data") or raw.get("Data") or raw.get("result") or []
    else:
        products = []

    print("    OK: " + str(len(products)) + " products loaded.")

    # Fetch existing Shopify SKUs
    print("\n[2] Fetching existing Shopify SKUs (to skip duplicates) ...")
    existing_skus = get_existing_shopify_skus()
    print("    OK: " + str(len(existing_skus)) + " SKUs already in Shopify.")

    # Sync loop
    total = len(products)
    print("\n[3] Syncing " + str(total) + " products to Shopify ...")
    created = 0
    skipped = 0
    errors  = 0
    report  = []

    for i, item in enumerate(products, 1):
        sku   = item.get("skuName", "ITEM-" + str(i))
        title = item.get("itemDesc") or sku

        # Skip if already exists
        if sku in existing_skus:
            print("  [" + str(i).rjust(3) + "/" + str(total) + "] SKIP     " + sku + " (already in Shopify)")
            skipped += 1
            report.append({"sku": sku, "status": "skipped", "reason": "already exists"})
            continue

        payload = build_shopify_payload(item)

        try:
            res = requests.post(
                SHOPIFY_PRODUCTS_URL,
                headers=SHOPIFY_HEADERS,
                json=payload,
                timeout=30
            )

            if res.status_code == 201:
                shopify_id = res.json()["product"]["id"]
                print("  [" + str(i).rjust(3) + "/" + str(total) + "] CREATED  " + sku + " -> Shopify ID " + str(shopify_id))
                created += 1
                report.append({"sku": sku, "status": "created", "shopify_id": shopify_id})
                existing_skus.add(sku)

            elif res.status_code == 429:
                print("  [" + str(i).rjust(3) + "/" + str(total) + "] RATE LIMITED, waiting 10s ...")
                time.sleep(10)
                res = requests.post(SHOPIFY_PRODUCTS_URL, headers=SHOPIFY_HEADERS, json=payload, timeout=30)
                if res.status_code == 201:
                    shopify_id = res.json()["product"]["id"]
                    print("  [" + str(i).rjust(3) + "/" + str(total) + "] CREATED (retry) " + sku + " -> " + str(shopify_id))
                    created += 1
                    report.append({"sku": sku, "status": "created", "shopify_id": shopify_id})
                    existing_skus.add(sku)
                else:
                    print("  [" + str(i).rjust(3) + "/" + str(total) + "] FAILED   " + sku + " -> " + str(res.status_code) + ": " + res.text[:200])
                    errors += 1
                    report.append({"sku": sku, "status": "error", "code": res.status_code, "msg": res.text[:200]})
            else:
                print("  [" + str(i).rjust(3) + "/" + str(total) + "] FAILED   " + sku + " -> " + str(res.status_code) + ": " + res.text[:200])
                errors += 1
                report.append({"sku": sku, "status": "error", "code": res.status_code, "msg": res.text[:200]})

        except requests.exceptions.RequestException as e:
            print("  [" + str(i).rjust(3) + "/" + str(total) + "] ERROR    " + sku + " -> " + str(e))
            errors += 1
            report.append({"sku": sku, "status": "error", "msg": str(e)})

        # Shopify rate limit: ~2 req/sec on Basic plan
        time.sleep(0.6)

    # Save report
    summary = {
        "synced_at": datetime.now().isoformat(),
        "total": total,
        "created": created,
        "skipped": skipped,
        "errors": errors,
        "results": report,
    }
    with open(SYNC_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("  SYNC COMPLETE")
    print("  Total   : " + str(total))
    print("  Created : " + str(created))
    print("  Skipped : " + str(skipped) + " (already existed)")
    print("  Errors  : " + str(errors))
    print("  Report  : " + SYNC_REPORT_FILE)
    print("=" * 65)


if __name__ == "__main__":
    main()
