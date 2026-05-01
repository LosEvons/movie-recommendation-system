import logging
import numpy as np
from unittest.mock import MagicMock, patch

from movie_recommender.app import _recommend_titles, recommend


def _mock_embed_fn(values: list):
    """Returns a callable for model.embed that yields numpy arrays, one per input."""

    def _embed(texts, **kwargs):
        return (np.array(v) for v in values)

    return _embed


# recommend


def test_recommend_empty_query():
    assert recommend("") == "Please enter a movie description first."
    assert recommend("   ") == "Please enter a movie description first."


def test_recommend_connection_error(caplog):
    with (
        caplog.at_level(logging.ERROR, logger="movie_recommender.app"),
        patch(
            "movie_recommender.app.get_collection", side_effect=RuntimeError("refused")
        ),
    ):
        result = recommend("space adventure")

    assert "currently unavailable" in result
    assert any("ChromaDB unreachable" in msg for msg in caplog.messages)


def test_recommend_query_error(caplog):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 3
    mock_collection.query.side_effect = Exception("ChromaDB exploded")

    mock_model = MagicMock()
    mock_model.embed.side_effect = _mock_embed_fn([[0.1]])

    with (
        caplog.at_level(logging.ERROR, logger="movie_recommender.app"),
        patch("movie_recommender.app.get_collection", return_value=mock_collection),
        patch("movie_recommender.app.model", mock_model),
    ):
        result = recommend("action movie")

    assert "Something went wrong" in result
    assert any("Recommendation query failed" in msg for msg in caplog.messages)


# _recommend_titles


def test_recommend_titles_returns_list():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 3
    mock_collection.query.return_value = {
        "metadatas": [
            [
                {"title": "Interstellar", "genre": "Sci-Fi"},
                {"title": "Gravity", "genre": "Sci-Fi"},
                {"title": "The Martian", "genre": "Sci-Fi"},
            ]
        ]
    }

    mock_model = MagicMock()
    mock_model.embed.side_effect = _mock_embed_fn([[0.1, 0.2, 0.3]])

    with (
        patch("movie_recommender.app.get_collection", return_value=mock_collection),
        patch("movie_recommender.app.model", mock_model),
    ):
        titles = _recommend_titles("space adventure", top_k=3)

    assert titles == ["Interstellar", "Gravity", "The Martian"]


def test_recommend_titles_passes_top_k_to_query():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    mock_collection.query.return_value = {"metadatas": [[{"title": "A"}]]}

    mock_model = MagicMock()
    mock_model.embed.side_effect = _mock_embed_fn([[0.1]])

    with (
        patch("movie_recommender.app.get_collection", return_value=mock_collection),
        patch("movie_recommender.app.model", mock_model),
    ):
        _recommend_titles("thriller", top_k=7)

    mock_collection.query.assert_called_once()
    assert mock_collection.query.call_args.kwargs["n_results"] == 7


def test_recommend_titles_empty_collection_returns_empty():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    with patch("movie_recommender.app.get_collection", return_value=mock_collection):
        titles = _recommend_titles("space adventure")

    assert titles == []


def test_recommend_titles_no_results_returns_empty():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 3
    mock_collection.query.return_value = {"metadatas": [[]]}

    mock_model = MagicMock()
    mock_model.embed.side_effect = _mock_embed_fn([[0.1, 0.2, 0.3]])

    with (
        patch("movie_recommender.app.get_collection", return_value=mock_collection),
        patch("movie_recommender.app.model", mock_model),
    ):
        titles = _recommend_titles("something obscure")

    assert titles == []
