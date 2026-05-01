import os
import logging

import kagglehub
import pandas as pd
from sentence_transformers import SentenceTransformer

from movie_recommender.chroma import get_collection

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
TMDB_LIMIT = int(os.getenv("TMDB_LIMIT", "5000"))

logger.info("Loading sentence transformer model %r", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)
logger.info("Model ready")


def find_csv_path(dataset_path: str) -> str:
    for file_name in sorted(os.listdir(dataset_path)):
        if "movie" in file_name.lower() and file_name.endswith(".csv"):
            return os.path.join(dataset_path, file_name)
    raise FileNotFoundError(f"No movie CSV found in dataset at {dataset_path!r}")


def ingest_movies() -> None:
    logger.info("Downloading TMDB dataset")
    try:
        path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
    except Exception as exc:
        logger.error("Dataset download failed: %s", exc)
        raise

    try:
        csv_path = find_csv_path(path)
    except FileNotFoundError:
        logger.error("No movie CSV found in downloaded dataset at %r", path)
        raise

    logger.info("Loading dataset from %s", csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.error("Failed to read CSV at %r: %s", csv_path, exc)
        raise

    df = df.dropna(subset=["overview"]).head(TMDB_LIMIT)
    df["id_str"] = df["id"].astype(str)
    logger.info("Loaded %s rows", len(df))

    if df.empty:
        logger.warning("No rows with non-empty overview; skipping ingestion")
        return

    try:
        collection = get_collection()
    except RuntimeError:
        logger.error("ChromaDB unavailable; aborting ingestion")
        raise

    logger.info("Encoding %s overviews with model %r", len(df), MODEL_NAME)
    embeds = model.encode(df["overview"].tolist(), show_progress_bar=True)

    logger.info("Writing to ChromaDB")
    try:
        collection.upsert(
            embeddings=embeds.tolist(),
            documents=df["overview"].tolist(),
            metadatas=[
                {"title": r["title"], "genre": str(r.get("genres", ""))}
                for r in df.to_dict("records")
            ],
            ids=df["id_str"].tolist(),
        )
    except Exception as exc:
        logger.error("ChromaDB upsert failed after encoding %s movies: %s", len(df), exc)
        raise

    logger.info("Upserted %s movies", len(df))


if __name__ == "__main__":
    import sys
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    try:
        ingest_movies()
    except Exception:
        logger.exception("Ingestion failed")
        sys.exit(1)
