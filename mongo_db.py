"""
mongo_db.py
-----------
MongoDB Atlas connection and helper functions.
Replaces MySQL for catalog, product_details, product_list, search_history.
"""
import os
import json
import datetime
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_BASE_DIR, ".env"))

MONGO_URI = os.getenv("MONGO_URI", "MONGO_URI_REMOVED?appName=Cluster0")
MONGO_DB  = os.getenv("MONGO_DATABASE", "jewellery")

_client = None
_db     = None


def get_db():
    global _client, _db
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=15000)
        _db = _client[MONGO_DB]
    return _db


def col(name: str):
    return get_db()[name]


def safe(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return v
