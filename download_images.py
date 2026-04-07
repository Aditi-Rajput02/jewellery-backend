"""
Download all product images from BG Jewels CDN to a local folder.
Then you can:
  1. Upload the folder to any public hosting (Cloudinary, S3, GitHub, etc.)
  2. Or use the Shopify Admin to manually import images

Usage:
  python backend/download_images.py

Output:
  backend/product_images/  (folder with all .jpg files)
  backend/image_url_map.json  (mapping: sku -> local filename)
"""
import os
import json
import requests
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
MONGO_URI = os.getenv("MONGO_URI", "MONGO_URI_REMOVED?appName=Cluster0")
MONGO_DB = os.getenv("MONGO_DATABASE", "jewellery")
IMG_BASE = "https://bgjewels.jewelscore.com/f/shortimage/"

OUT_DIR = os.path.join(os.path.dirname(__file__), "product_images")
os.makedirs(OUT_DIR, exist_ok=True)

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=15000)
db = client[MONGO_DB]
catalog_docs = list(db["catalog"].find({}, {"sku_name": 1, "picture1": 1}))
client.close()

print(f"Downloading {len(catalog_docs)} images to: {OUT_DIR}")
print("=" * 60)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://bgjewels.jewelscore.com/"
})

url_map = {}
ok = 0
skip = 0
fail = 0

for i, doc in enumerate(catalog_docs, 1):
    sku = doc.get("sku_name", "")
    pic = (doc.get("picture1") or "").strip()
    if not pic:
        print(f"  [{i:4d}/{len(catalog_docs)}] SKIP {sku} (no picture1)")
        skip += 1
        continue

    out_file = os.path.join(OUT_DIR, pic)
    url = IMG_BASE + pic

    # Skip if already downloaded
    if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        url_map[sku] = pic
        ok += 1
        if i % 100 == 0:
            print(f"  [{i:4d}/{len(catalog_docs)}] CACHED {sku} -> {pic}")
        continue

    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image/"):
            with open(out_file, "wb") as f:
                f.write(r.content)
            url_map[sku] = pic
            ok += 1
            print(f"  [{i:4d}/{len(catalog_docs)}] OK   {sku} -> {pic} ({len(r.content):,} bytes)")
        else:
            print(f"  [{i:4d}/{len(catalog_docs)}] FAIL {sku} -> HTTP {r.status_code} ct={r.headers.get('Content-Type','?')}")
            fail += 1
    except Exception as e:
        print(f"  [{i:4d}/{len(catalog_docs)}] ERR  {sku} -> {e}")
        fail += 1

# Save URL map
map_file = os.path.join(os.path.dirname(__file__), "image_url_map.json")
with open(map_file, "w") as f:
    json.dump(url_map, f, indent=2)

print()
print("=" * 60)
print(f"  Downloaded : {ok}")
print(f"  Skipped    : {skip}")
print(f"  Failed     : {fail}")
print(f"  Images dir : {OUT_DIR}")
print(f"  URL map    : {map_file}")
print("=" * 60)
print()
print("NEXT STEPS:")
print("  Option A - Upload to Cloudinary (free):")
print("    1. Sign up at https://cloudinary.com (free)")
print("    2. Run: python backend/upload_to_cloudinary.py")
print()
print("  Option B - Ask store owner to create Shopify API token:")
print("    Shopify Admin > Settings > Apps > Develop apps")
print("    Scopes needed: write_products, write_files")
