"""
sync_chroma.py
--------------
One-time script to sync all MongoDB product descriptions → MongoDB rag_embeddings.
Embeddings are now stored in MongoDB Atlas (not ChromaDB on disk),
so they persist across Render deploys.

Usage:
  python sync_chroma.py
"""

import os
import time
import datetime
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from openai import OpenAI
from mongo_db import get_db
from rag_pipeline import save_transcript, count_transcripts

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
IMG_BASE = "https://bgjewels.jewelscore.com/f/shortimage/"


def main():
    print("=== RAG Sync: MongoDB descriptions -> rag_embeddings ===")
    client = OpenAI(api_key=OPENAI_API_KEY)
    db = get_db()

    # Collect all SKUs with descriptions from all collections
    sku_desc_map = {}
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
    print(f"Found {total_mongo} SKUs with descriptions in MongoDB.")

    if total_mongo == 0:
        print("Nothing to sync. Add descriptions first via bulk_describe.py or /catalog/describe-from-url")
        return

    # Check which SKUs are already in rag_embeddings
    rag_col = db["rag_embeddings"]
    existing_ids = set(d["_id"] for d in rag_col.find({}, {"_id": 1}))

    to_sync = {k: v for k, v in sku_desc_map.items() if k not in existing_ids}
    print(f"Already in rag_embeddings: {len(existing_ids)} | To sync: {len(to_sync)}")

    if not to_sync:
        print(f"Already up to date! Total: {count_transcripts()}")
        return

    synced = 0
    errors = 0

    for i, (sku, doc) in enumerate(to_sync.items(), 1):
        try:
            desc  = doc["description"].strip()
            resp  = client.embeddings.create(model="text-embedding-3-small", input=desc)
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
                    "source":      "mongo_sync",
                    "saved_at":    datetime.datetime.now(datetime.UTC).isoformat(),
                    "length":      len(desc),
                },
                upsert=True,
            )
            synced += 1
            if synced % 10 == 0 or synced == 1:
                print(f"  [{synced}/{len(to_sync)}] Synced {sku}")
            time.sleep(0.05)  # small delay to avoid OpenAI rate limits
        except Exception as e:
            errors += 1
            print(f"  ERROR syncing {sku}: {e}")

    total = count_transcripts()
    print(f"\n=== Done! Synced: {synced} | Errors: {errors} | Total in rag_embeddings: {total} ===")


if __name__ == "__main__":
    main()
