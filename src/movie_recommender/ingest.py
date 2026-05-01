import os
import logging
import time

import kagglehub
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_RETRIES = int(os.getenv("CHROMA_RETRIES", "10"))
CHROMA_RETRY_DELAY = float(os.getenv("CHROMA_RETRY_DELAY", "2"))
TMDB_LIMIT = int(os.getenv("TMDB_LIMIT", "5000"))

model = SentenceTransformer(MODEL_NAME)


def get_collection():
    last_error: Exception | None = None
    for attempt in range(1, CHROMA_RETRIES + 1):
        try:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            collection = client.get_or_create_collection(name="movies")
            logger.info("Connected to ChromaDB at %s:%s", CHROMA_HOST, CHROMA_PORT)
            return collection
        except Exception as exc:  # pragma: no cover - network/runtime dependent
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


def find_csv_path(dataset_path: str) -> str:
    for file_name in os.listdir(dataset_path):
        if "movie" in file_name.lower() and file_name.endswith(".csv"):
            return os.path.join(dataset_path, file_name)
    raise FileNotFoundError("Could not find a movie CSV file in the downloaded dataset")


def ingest_movies() -> None:
    logger.info("Downloading TMDB dataset")
    path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
    csv_path = find_csv_path(path)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["overview"]).head(TMDB_LIMIT)
    df["id_str"] = df["id"].astype(str)
    logger.info("Loaded %s rows", len(df))

    if df.empty:
        logger.warning("No rows with non-empty overview were found; skipping ingestion")
        return

    collection = get_collection()

    if collection.count() == 0:
        logger.info("Creating embeddings")
        embeds = model.encode(df["overview"].tolist())

        collection.add(
            embeddings=embeds.tolist(),
            documents=df["overview"].tolist(),
            metadatas=[
                {
                    "title": row["title"],
                    "genre": str(row.get("genres", "")),
                }
                for _, row in df.iterrows()
            ],
            ids=df["id_str"].tolist(),
        )
        logger.info("Ingested %s movies", collection.count())
    else:
        logger.info("Pre-existing %s movies", collection.count())


if __name__ == "__main__":
    ingest_movies()
