import logging
import os
import time

import chromadb

logger = logging.getLogger(__name__)

CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_RETRIES = int(os.getenv("CHROMA_RETRIES", "10"))
CHROMA_RETRY_DELAY = float(os.getenv("CHROMA_RETRY_DELAY", "2"))

_collection = None


def get_collection():
    global _collection
    if _collection is not None:
        return _collection

    last_error: Exception | None = None
    for attempt in range(1, CHROMA_RETRIES + 1):
        try:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            _collection = client.get_or_create_collection(name="movies")
            logger.info("Connected to ChromaDB at %s:%s", CHROMA_HOST, CHROMA_PORT)
            return _collection
        except Exception as exc:
            last_error = exc
            logger.warning(
                "ChromaDB connection attempt %s/%s failed: %s",
                attempt,
                CHROMA_RETRIES,
                exc,
            )
            if attempt < CHROMA_RETRIES:
                time.sleep(CHROMA_RETRY_DELAY)

    raise RuntimeError("Could not connect to ChromaDB") from last_error
