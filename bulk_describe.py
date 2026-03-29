"""
bulk_describe.py
----------------
Run ONCE locally to generate GPT-4o descriptions for all products
that don't have one yet, and save them to MongoDB (catalog + rag_embeddings).

Embeddings are stored in MongoDB Atlas — they persist across Render deploys.
No ChromaDB or local disk needed.

Usage:
  python bulk_describe.py
"""

import os, io, time, base64, datetime, requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from openai import OpenAI
from PIL import Image as PILImage
from mongo_db import get_db

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
IMG_BASE       = "https://bgjewels.jewelscore.com/f/shortimage/"
SKU_IMG_BASE   = "https://sjadau.jewelscore.com/f/skuimage/"
DELAY          = 1.0   # seconds between products (avoids rate limits)

client = OpenAI(api_key=OPENAI_API_KEY)


def download_image(url):
    try:
        r = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://bgjewels.jewelscore.com/",
        }, timeout=15)
        r.raise_for_status()
        PILImage.open(io.BytesIO(r.content)).convert("RGB")
        return r.content
    except Exception:
        return None


def describe_image(image_bytes):
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": (
                "You are a jewellery expert. Describe this jewellery image in detail. "
                "Include: type of jewellery (necklace, ring, earring, bracelet, etc.), "
                "metal color (gold, silver, rose gold, etc.), "
                "stones or gems visible (diamond, ruby, emerald, etc.), "
                "design style (traditional, modern, polki, kundan, etc.), "
                "and any other notable features. "
                "Keep the description concise but informative (2-4 sentences)."
            )},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "low",
            }},
        ]}],
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def main():
    db      = get_db()
    rag_col = db["rag_embeddings"]

    no_desc = list(db["catalog"].find(
        {"$or": [
            {"description": {"$exists": False}},
            {"description": None},
            {"description": ""},
        ]},
        {"sku_name": 1, "picture1": 1, "item_name": 1,
         "style_name": 1, "metal_code": 1, "color_name": 1,
         "show_catalog_price": 1}
    ))

    total = len(no_desc)
    print(f"=== Bulk Describe ===")
    print(f"Products without description: {total}\n")

    if total == 0:
        print("All products already have descriptions!")
        print(f"Total in rag_embeddings: {rag_col.count_documents({})}")
        return

    described = skipped = errors = 0

    for i, doc in enumerate(no_desc, 1):
        sku = doc.get("sku_name", "").strip()
        pic = doc.get("picture1", "").strip()
        if not sku:
            skipped += 1
            continue

        # Try catalog image, then SKU image variants
        image_bytes = None
        for url in [
            (IMG_BASE + pic) if pic else None,
            SKU_IMG_BASE + sku + "_1.jpg",
            SKU_IMG_BASE + sku + ".jpg",
        ]:
            if url:
                image_bytes = download_image(url)
                if image_bytes:
                    break

        if not image_bytes:
            print(f"  [{i}/{total}] SKIP {sku} — no image")
            skipped += 1
            continue

        try:
            description = describe_image(image_bytes)
            if not description:
                raise ValueError("Empty description")

            # 1. Save description to MongoDB catalog/product_details/product_list
            for coll_name in ("catalog", "product_details", "product_list"):
                db[coll_name].update_many(
                    {"sku_name": sku},
                    {"$set": {"description": description}},
                )

            # 2. Embed + save to MongoDB rag_embeddings (persistent across deploys)
            emb_resp  = client.embeddings.create(model="text-embedding-3-small", input=description)
            embedding = emb_resp.data[0].embedding
            price = doc.get("show_catalog_price")

            rag_col.replace_one(
                {"_id": sku},
                {
                    "_id":         sku,
                    "sku_name":    sku,
                    "description": description,
                    "embedding":   embedding,
                    "item_name":   doc.get("item_name", ""),
                    "style":       doc.get("style_name", ""),
                    "metal":       doc.get("metal_code", ""),
                    "color":       doc.get("color_name", ""),
                    "price":       str(price) if price else "",
                    "picture1":    pic,
                    "image_url":   (IMG_BASE + pic) if pic else "",
                    "source":      "bulk_describe",
                    "saved_at":    datetime.datetime.now(datetime.UTC).isoformat(),
                    "length":      len(description),
                },
                upsert=True,
            )

            described += 1
            print(f"  [{i}/{total}] OK {sku}: {description[:70]}...")

        except Exception as e:
            errors += 1
            print(f"  [{i}/{total}] ERROR {sku}: {e}")

        time.sleep(DELAY)

    total_rag = rag_col.count_documents({})
    print(f"\n=== Done! Described={described} | Skipped={skipped} | Errors={errors} ===")
    print(f"Total in rag_embeddings (MongoDB): {total_rag}")
    print(f"\nNext steps:")
    print(f"  - Deploy to Render — embeddings are in MongoDB Atlas, no file push needed.")
    print(f"  - Run sync_chroma.py to add any remaining described SKUs to rag_embeddings.")


if __name__ == "__main__":
    main()
