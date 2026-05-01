import gradio as gr
import logging
import os

from sentence_transformers import SentenceTransformer

from movie_recommender.chroma import get_collection

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

logger.info("Loading sentence transformer model %r", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)
logger.info("Model ready")


def _env_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _recommend_titles(query: str, top_k: int = 5) -> list[str]:
    collection = get_collection()

    if collection.count() == 0:
        return []

    q_embed = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_embed,
        n_results=top_k,
        include=["metadatas"],
    )
    metadatas = results.get("metadatas", [])
    if not metadatas or not metadatas[0]:
        return []

    return [meta.get("title", "Unknown title") for meta in metadatas[0]]


def recommend(query: str, top_k: int = 5) -> str:
    if not query.strip():
        return "Please enter a movie description first."

    try:
        titles = _recommend_titles(query, top_k)
    except RuntimeError:
        logger.error("ChromaDB unreachable; cannot serve recommendation request")
        return "The recommendation service is currently unavailable. Please try again later."
    except Exception:
        logger.exception("Recommendation query failed for query %r", query)
        return (
            "Something went wrong while looking up recommendations. Please try again."
        )

    if not titles:
        return "No recommendations found. The database may be empty or the query too specific."

    return "Recommendations:\n" + "\n".join(f"- {title}" for title in titles)


demo = gr.Interface(
    fn=recommend,
    inputs=[
        gr.Textbox(label="Description", placeholder="placeholder", lines=2),
        gr.Slider(3, 10, value=5, step=1, label="Count"),
    ],
    outputs=gr.Textbox(label="Movies", lines=8),
    title="Movie Recommendation System",
    description="Describe a movie to get recommendation",
)

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = _env_bool(os.getenv("GRADIO_SHARE"), default=False)

    try:
        demo.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
        )
    except Exception:
        logger.exception("Failed to launch Gradio app")
        sys.exit(1)
