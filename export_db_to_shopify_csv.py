# -*- coding: utf-8 -*-
"""
MongoDB -> Shopify CSV Export (matching Shopify product_template.csv format)
=============================================================================
Reads all products from MongoDB Atlas and generates a Shopify-compatible CSV.

Usage:
  cd backend
  python export_db_to_shopify_csv.py

Output:
  shopify_products.csv  (ready to import into Shopify Admin -> Products -> Import)
"""

import sys
import csv
import os
import requests as req_lib
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------
OUTPUT_CSV   = "shopify_products.csv"
IMG_BASE     = "https://bgjewels.jewelscore.com/f/shortimage/"
SKU_IMG_BASE = "https://sjadau.jewelscore.com/f/skuimage/"

# API base URL - set to your running FastAPI server
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

# Style names to EXCLUDE from tags (e.g. generic/unhelpful values)
EXCLUDE_STYLE_NAMES = {"Silver Jewellery", "silver jewellery", "SILVER JEWELLERY"}

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
MONGO_URI = os.getenv(
    "MONGO_URI",
    "MONGO_URI_REMOVED?appName=Cluster0"
)
MONGO_DB = os.getenv("MONGO_DATABASE", "jewellery")

# -----------------------------------------------
# Shopify CSV headers (matching product_template.csv exactly)
# -----------------------------------------------
HEADERS = [
    "Title",
    "URL handle",
    "Description",
    "Vendor",
    "Product category",
    "Type",
    "Tags",
    "Published on online store",
    "Status",
    "SKU",
    "Barcode",
    "Option1 name",
    "Option1 value",
    "Option1 Linked To",
    "Option2 name",
    "Option2 value",
    "Option2 Linked To",
    "Option3 name",
    "Option3 value",
    "Option3 Linked To",
    "Price",
    "Compare-at price",
    "Cost per item",
    "Charge tax",
    "Tax code",
    "Unit price total measure",
    "Unit price total measure unit",
    "Unit price base measure",
    "Unit price base measure unit",
    "Inventory tracker",
    "Inventory quantity",
    "Continue selling when out of stock",
    "Weight value (grams)",
    "Weight unit for display",
    "Requires shipping",
    "Fulfillment service",
    "Product image URL",
    "Image position",
    "Image alt text",
    "Variant image URL",
    "Gift card",
    "SEO title",
    "SEO description",
    "Color (product.metafields.shopify.color-pattern)",
    "Google Shopping / Google product category",
    "Google Shopping / Gender",
    "Google Shopping / Age group",
    "Google Shopping / Manufacturer part number (MPN)",
    "Google Shopping / Ad group name",
    "Google Shopping / Ads labels",
    "Google Shopping / Condition",
    "Google Shopping / Custom product",
    "Google Shopping / Custom label 0",
    "Google Shopping / Custom label 1",
    "Google Shopping / Custom label 2",
    "Google Shopping / Custom label 3",
    "Google Shopping / Custom label 4",
    "ai-description",
]


def clean_tag(t):
    if not t:
        return ""
    t = str(t).strip().strip("'\"").replace("&", "and").strip()
    return t


def make_handle(sku):
    return sku.lower().replace("_", "-").replace(" ", "-")


BGJEWELS_API_URL = "https://bgjewels.jewelscore.com/api/catalog/CatalogListOptimize"
BGJEWELS_API_PAYLOAD_BASE = {
    "minprice": "0", "maxprice": "0", "pageno": "1",
    "itemList": [], "vendorList": [], "CollectionList": [], "GroupList": [],
    "MetalList": [], "MtColorArr": [], "StoneList": [], "SubGrpList": [],
    "showOnlyInStock": False, "maxCsWt": "", "minCsWt": "", "minNwt": "",
    "maxNwt": "", "maxDWt": "", "minDWt": "", "multipleSKU": "",
    "selectserchtype": "", "selectcheckboxserchtype": "", "currencyid": "1",
    "locationid": "0", "Type": "Catalog", "approvalNo": None, "purType": "PurApp",
    "PlatingList": [], "MicronList": [], "SizeList": "", "consideSingleStone": False,
    "clientId": None, "stoneShape": [], "stoneSize": [], "stoneColor": []
}


def build_description_from_api_item(item):
    """Build a rich product description from BG Jewels API catalog item fields."""
    parts = []
    item_name  = (item.get("itemName") or "").strip().title()
    style_name = (item.get("stylename") or "").strip()
    metal_code = (item.get("metalCode") or "").strip()
    color_name = (item.get("colorName") or "").strip()
    size       = (item.get("size") or "").strip()
    gross_wt   = item.get("grossWt") or 0
    dia_wt     = item.get("diaWt") or 0
    cs_wt      = item.get("csWt") or 0
    dia_pcs    = item.get("diaPcs") or 0
    cs_pcs     = item.get("csPcs") or 0

    metal_map = {"YG": "Yellow Gold", "WG": "White Gold", "RG": "Rose Gold",
                 "SS": "Silver", "PT": "Platinum", "18K": "18K Gold", "14K": "14K Gold"}
    color_map = {"YG": "Yellow", "WG": "White", "RG": "Rose", "SS": "Silver"}
    metal_str = metal_map.get(metal_code, metal_code)
    color_str = color_map.get(color_name, color_name)

    if item_name:
        if color_str and color_str not in metal_str:
            parts.append(f"Elegant {color_str} {metal_str} {item_name}")
        else:
            parts.append(f"Elegant {metal_str} {item_name}")

    if style_name and style_name not in EXCLUDE_STYLE_NAMES:
        parts.append(f"from the {style_name} collection")

    stone_parts = []
    if dia_wt and float(dia_wt) > 0:
        stone_parts.append(f"{float(dia_wt):.2f}ct diamonds ({int(dia_pcs)} pcs)")
    if cs_wt and float(cs_wt) > 0:
        stone_parts.append(f"{float(cs_wt):.2f}ct colour stones ({int(cs_pcs)} pcs)")
    if stone_parts:
        parts.append("set with " + " and ".join(stone_parts))

    if gross_wt and float(gross_wt) > 0:
        parts.append(f"Total weight: {float(gross_wt):.2f}g")
    if size and size not in ("-", ""):
        parts.append(f"Size: {size}")

    return ". ".join(parts) + "." if parts else ""


def fetch_api_descriptions(_total_products=None):
    """
    Fetch all products from BG Jewels external API with pagination.
    Builds a description string from each item's fields.
    Returns a dict: { sku_name -> description_string }
    """
    api_desc_map = {}
    try:
        print(f"  Fetching products from BG Jewels API ...")
        headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        payload = dict(BGJEWELS_API_PAYLOAD_BASE)
        payload["pageno"] = "1"
        resp = req_lib.post(BGJEWELS_API_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        total_items = data.get("totalItems", 0)
        page_size   = data.get("pagesize", 50)
        total_pages = max(1, (total_items + page_size - 1) // page_size)
        print(f"  BG Jewels API: {total_items} total items, {page_size}/page, {total_pages} pages")

        for item in data.get("cataloglist", []):
            sku = (item.get("skuName") or "").strip()
            if sku:
                api_desc_map[sku] = build_description_from_api_item(item)

        for page in range(2, total_pages + 1):
            payload = dict(BGJEWELS_API_PAYLOAD_BASE)
            payload["pageno"] = str(page)
            try:
                resp = req_lib.post(BGJEWELS_API_URL, json=payload, headers=headers, timeout=30)
                resp.raise_for_status()
                page_data = resp.json()
                for item in page_data.get("cataloglist", []):
                    sku = (item.get("skuName") or "").strip()
                    if sku:
                        api_desc_map[sku] = build_description_from_api_item(item)
                print(f"  Page {page}/{total_pages}: {len(page_data.get('cataloglist', []))} items fetched")
            except Exception as pe:
                print(f"  WARNING: Page {page} failed: {pe}")

        print(f"  BG Jewels API descriptions built: {len(api_desc_map)} SKUs")
    except Exception as e:
        print(f"  WARNING: Could not fetch from BG Jewels API ({e}). Description column will be empty.")
    return api_desc_map


def get_mongo_products():
    print("  Connecting to MongoDB Atlas ...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=15000)
    db = client[MONGO_DB]

    catalog_docs = list(db["catalog"].find({}))
    print(f"  catalog        : {len(catalog_docs)} documents")

    sku_list = [d.get("sku_name", "") for d in catalog_docs if d.get("sku_name")]
    pd_map = {d["sku_name"]: d for d in db["product_details"].find({"sku_name": {"$in": sku_list}})}
    pl_map = {d["sku_name"]: d for d in db["product_list"].find({"sku_name": {"$in": sku_list}})}
    print(f"  product_details: {len(pd_map)} documents")
    print(f"  product_list   : {len(pl_map)} documents")

    # Fetch descriptions from API (these go into the "Description" column)
    api_desc_map = fetch_api_descriptions(len(catalog_docs))

    products = []
    for c in catalog_docs:
        sku = c.get("sku_name", "")
        if not sku:
            continue
        pd = pd_map.get(sku, {})
        pl = pl_map.get(sku, {})

        # ai-description = the existing description stored in DB (generated by GPT-4o Vision)
        ai_description = (
            c.get("description") or
            pd.get("description") or
            pl.get("description") or ""
        )

        # Description = fetched fresh from the API
        api_description = api_desc_map.get(sku, "")

        merged = {
            "skuName":          sku,
            "metalCode":        c.get("metal_code", ""),
            "itemName":         c.get("item_name", "Jewellery"),
            "styleName":        c.get("style_name", ""),
            "colorName":        c.get("color_name", ""),
            "sGroupName":       c.get("s_group_name", ""),
            "subGroupName":     pd.get("sub_group_name", ""),
            "grossWt":          c.get("gross_wt") or c.get("weight") or 0.0,
            "stockQty":         c.get("stock_qty", 0),
            "picture1":         c.get("picture1", ""),
            "skuImage1":        pd.get("sku_image1", "") or "",
            "skuImage2":        pd.get("sku_image2", "") or "",
            "skuImage3":        pd.get("sku_image3", "") or "",
            "skuImage4":        pd.get("sku_image4", "") or "",
            "showCatalogPrice": c.get("show_catalog_price") or 0.0,
            "tagPrice":         pd.get("tag_price") or 0.0,
            "pricePerPcs":      pd.get("price_per_pcs") or 0.0,
            "wsPrice":          pl.get("ws_price") or pl.get("wsPrice") or 0.0,
            "itemDesc":         pd.get("item_desc", "") or "",
            "description":      api_description,   # from API
            "ai_description":   ai_description,    # from DB (GPT-4o Vision generated)
        }
        products.append(merged)

    client.close()
    return products


def empty_row(handle):
    """Return a blank row with only the handle filled (for extra image rows)."""
    row = {h: "" for h in HEADERS}
    row["URL handle"] = handle
    return row


def build_rows(item):
    sku    = item["skuName"]
    handle = make_handle(sku)
    title  = item.get("itemDesc") or sku
    desc   = item.get("description", "") or ""
    vendor = "BG Jewels"
    ptype  = item.get("itemName", "Jewellery")

    # Tags - exclude EXCLUDE_STYLE_NAMES (e.g. "Silver Jewellery")
    raw_tags = []
    sub = (item.get("subGroupName", "") or "").strip().strip("'\"")
    for t in sub.split(","):
        t = clean_tag(t)
        if t and t != "-" and t not in EXCLUDE_STYLE_NAMES:
            raw_tags.append(t)
    for field in ["colorName", "styleName", "itemName", "metalCode", "sGroupName"]:
        val = clean_tag(item.get(field) or "")
        if val and val != "-" and val not in EXCLUDE_STYLE_NAMES:
            raw_tags.append(val)
    seen = set()
    clean_tags = []
    for t in raw_tags:
        if t not in seen:
            seen.add(t)
            clean_tags.append(t)
    tags = ", ".join(clean_tags)

    # Price
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

    # Weight
    try:
        weight_g = round(float(item.get("grossWt") or 0.0), 2)
    except Exception:
        weight_g = 0.0

    # Stock
    try:
        stock_qty = max(1, int(item.get("stockQty") or 1))
    except Exception:
        stock_qty = 1

    # Images
    # NOTE: sjadau.jewelscore.com/f/skuimage/ returns text/html (not real images)
    # Only use picture1 from bgjewels.jewelscore.com/f/shortimage/ which returns image/jpeg
    images = []
    pic1 = (item.get("picture1") or "").strip()
    if pic1:
        images.append(IMG_BASE + pic1)

    first_img = images[0] if images else ""
    extra_imgs = images[1:]

    seo_title = f"{title} | BG Jewels"[:70]
    seo_desc  = desc[:320] if desc else f"{title} - {ptype} by BG Jewels"

    # First (main) row
    first_row = {
        "Title":                    title,
        "URL handle":               handle,
        "Description":              desc,
        "Vendor":                   vendor,
        "Product category":         "Apparel & Accessories > Jewelry",
        "Type":                     ptype,
        "Tags":                     tags,
        "Published on online store": "TRUE",
        "Status":                   "Active",
        "SKU":                      sku,
        "Barcode":                  "",
        "Option1 name":             "Title",
        "Option1 value":            "Default Title",
        "Option1 Linked To":        "",
        "Option2 name":             "",
        "Option2 value":            "",
        "Option2 Linked To":        "",
        "Option3 name":             "",
        "Option3 value":            "",
        "Option3 Linked To":        "",
        "Price":                    str(price),
        "Compare-at price":         "",
        "Cost per item":            "",
        "Charge tax":               "TRUE",
        "Tax code":                 "",
        "Unit price total measure": "",
        "Unit price total measure unit": "",
        "Unit price base measure":  "",
        "Unit price base measure unit": "",
        "Inventory tracker":        "shopify",
        "Inventory quantity":       str(stock_qty),
        "Continue selling when out of stock": "DENY",
        "Weight value (grams)":     str(weight_g),
        "Weight unit for display":  "g",
        "Requires shipping":        "TRUE",
        "Fulfillment service":      "manual",
        "Product image URL":        first_img,
        "Image position":           "1" if first_img else "",
        "Image alt text":           title if first_img else "",
        "Variant image URL":        "",
        "Gift card":                "FALSE",
        "SEO title":                seo_title,
        "SEO description":          seo_desc,
        "Color (product.metafields.shopify.color-pattern)": clean_tag(item.get("colorName") or ""),
        "Google Shopping / Google product category": "Apparel & Accessories > Jewelry",
        "Google Shopping / Gender": "",
        "Google Shopping / Age group": "",
        "Google Shopping / Manufacturer part number (MPN)": sku,
        "Google Shopping / Ad group name": "",
        "Google Shopping / Ads labels": "",
        "Google Shopping / Condition": "New",
        "Google Shopping / Custom product": "FALSE",
        "Google Shopping / Custom label 0": "",
        "Google Shopping / Custom label 1": "",
        "Google Shopping / Custom label 2": "",
        "Google Shopping / Custom label 3": "",
        "Google Shopping / Custom label 4": "",
        "ai-description":   item.get("ai_description", "") or "",
    }

    rows = [first_row]

    # Extra image rows
    for idx, img_src in enumerate(extra_imgs, 2):
        row = empty_row(handle)
        row["Product image URL"] = img_src
        row["Image position"]    = str(idx)
        row["Image alt text"]    = title
        rows.append(row)

    return rows


def main():
    print("=" * 65)
    print("  MongoDB -> Shopify CSV Export")
    print("  Started : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 65)

    print("\n[1] Loading products from MongoDB Atlas ...")
    try:
        products = get_mongo_products()
    except Exception as e:
        print(f"\n  ERROR connecting to MongoDB: {e}")
        sys.exit(1)
    print(f"    OK: {len(products)} products loaded.\n")

    print(f"[2] Writing CSV -> {OUTPUT_CSV} ...")
    total_rows = 0

    out_path = os.path.join(BASE_DIR, OUTPUT_CSV)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for i, item in enumerate(products, 1):
            sku = item.get("skuName", f"ITEM-{i}")
            try:
                rows = build_rows(item)
                for row in rows:
                    writer.writerow(row)
                total_rows += len(rows)
                print(f"  [{str(i).rjust(4)}/{len(products)}] OK  {sku}  ({len(rows)} row(s))")
            except Exception as e:
                print(f"  [{str(i).rjust(4)}/{len(products)}] ERR {sku} -> {e}")

    print("\n" + "=" * 65)
    print("  CSV EXPORT COMPLETE")
    print(f"  Products : {len(products)}")
    print(f"  CSV rows : {total_rows}")
    print(f"  File     : {out_path}")
    print("=" * 65)
    print("\nNEXT STEPS - Import into Shopify:")
    print("  1. Go to: https://bgluxe-official.myshopify.com/admin/products")
    print("  2. Click 'Import' button (top right)")
    print(f"  3. Click 'Add file' and select: {out_path}")
    print("  4. Click 'Upload and preview'")
    print("  5. Click 'Import products'")
    print("=" * 65)


if __name__ == "__main__":
    main()
