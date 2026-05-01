import chromadb
import gradio as gr
import logging
import os
import time

from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_RETRIES = int(os.getenv("CHROMA_RETRIES", "10"))
CHROMA_RETRY_DELAY = float(os.getenv("CHROMA_RETRY_DELAY", "2"))

model = SentenceTransformer(MODEL_NAME)
_collection = None


def _env_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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

def recommend(query: str, top_k: int = 5) -> str:
    if not query.strip():
        return "Please enter a movie description first."

    try:
        collection = get_collection()
        if collection.count() == 0:
            return "Run ingest first to populate the database"

        q_embed = model.encode([query])
        results = collection.query(
            query_embeddings=q_embed,
            n_results=top_k,
            include=["metadatas"],
        )
        metadatas = results.get("metadatas", [])
        if not metadatas or not metadatas[0]:
            return "No recommendations were returned. Run ingest first or try a different query."

        titles = [meta.get("title", "Unknown title") for meta in metadatas[0]]
        return "Recommendations:\n" + "\n".join(f"- {title}" for title in titles)
    except Exception:
        logger.exception("Recommendation lookup failed")
        return "Something went wrong while looking up recommendations. Please try again."

demo = gr.Interface(
    fn=recommend,
    inputs=[
        gr.Textbox(
            label="Description",
            placeholder="placeholder",
            lines=2
        ),
        gr.Slider(3, 10, value=5, step=1, label="Count")
    ],
    outputs=gr.Textbox(label="Movies", lines=8),
    title="Movie Recommendation System",
    description="Describe a movie to get recommendation",
)

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = _env_bool(os.getenv("GRADIO_SHARE"), default=False)

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )