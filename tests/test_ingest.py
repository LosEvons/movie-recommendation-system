import logging
import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from movie_recommender.ingest import find_csv_path, ingest_movies


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2],
            "title": ["Movie A", "Movie B"],
            "overview": ["An adventure story.", "A romance story."],
            "genres": ["Action", "Romance"],
        }
    )


def _mock_embed(values: list) -> MagicMock:
    m = MagicMock()
    m.tolist.return_value = values
    return m


# find_csv_path


def test_find_csv_path_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "tmdb_5000_movies.csv")
        open(csv_file, "w").close()
        assert find_csv_path(tmpdir) == csv_file


def test_find_csv_path_ignores_non_movie_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "credits.csv"), "w").close()
        open(os.path.join(tmpdir, "tmdb_5000_movies.csv"), "w").close()
        result = find_csv_path(tmpdir)
        assert os.path.basename(result) == "tmdb_5000_movies.csv"


def test_find_csv_path_returns_first_alphabetically():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "movie_a.csv"), "w").close()
        open(os.path.join(tmpdir, "movie_b.csv"), "w").close()
        result = find_csv_path(tmpdir)
        assert os.path.basename(result) == "movie_a.csv"


def test_find_csv_path_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            find_csv_path(tmpdir)


# ingest_movies


def test_ingest_movies_upserts_all_rows():
    df = _sample_df()
    mock_collection = MagicMock()
    mock_embed = _mock_embed([[0.1] * 384, [0.2] * 384])

    with (
        patch(
            "movie_recommender.ingest.kagglehub.dataset_download",
            return_value="/fake/path",
        ),
        patch(
            "movie_recommender.ingest.find_csv_path", return_value="/fake/movies.csv"
        ),
        patch("movie_recommender.ingest.pd.read_csv", return_value=df),
        patch("movie_recommender.ingest.get_collection", return_value=mock_collection),
        patch("movie_recommender.ingest.model") as mock_model,
    ):
        mock_model.encode.return_value = mock_embed
        ingest_movies()

    mock_collection.upsert.assert_called_once()
    kwargs = mock_collection.upsert.call_args.kwargs
    assert kwargs["ids"] == ["1", "2"]
    assert kwargs["metadatas"][0] == {"title": "Movie A", "genre": "Action"}
    assert kwargs["metadatas"][1] == {"title": "Movie B", "genre": "Romance"}


def test_ingest_movies_skips_when_no_overviews():
    df = pd.DataFrame(columns=["id", "title", "overview", "genres"])

    with (
        patch(
            "movie_recommender.ingest.kagglehub.dataset_download",
            return_value="/fake/path",
        ),
        patch(
            "movie_recommender.ingest.find_csv_path", return_value="/fake/movies.csv"
        ),
        patch("movie_recommender.ingest.pd.read_csv", return_value=df),
        patch("movie_recommender.ingest.get_collection") as mock_get_collection,
    ):
        ingest_movies()

    mock_get_collection.assert_not_called()


def test_ingest_movies_drops_rows_without_overview():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Has Overview", "No Overview", "Also Has"],
            "overview": ["Some text.", None, "More text."],
            "genres": ["Action", "Drama", "Comedy"],
        }
    )
    mock_collection = MagicMock()
    mock_embed = _mock_embed([[0.1] * 384, [0.2] * 384])

    with (
        patch(
            "movie_recommender.ingest.kagglehub.dataset_download",
            return_value="/fake/path",
        ),
        patch(
            "movie_recommender.ingest.find_csv_path", return_value="/fake/movies.csv"
        ),
        patch("movie_recommender.ingest.pd.read_csv", return_value=df),
        patch("movie_recommender.ingest.get_collection", return_value=mock_collection),
        patch("movie_recommender.ingest.model") as mock_model,
    ):
        mock_model.encode.return_value = mock_embed
        ingest_movies()

    kwargs = mock_collection.upsert.call_args.kwargs
    assert "2" not in kwargs["ids"]
    assert len(kwargs["ids"]) == 2


def test_ingest_movies_raises_on_download_failure(caplog):
    with (
        caplog.at_level(logging.ERROR, logger="movie_recommender.ingest"),
        patch(
            "movie_recommender.ingest.kagglehub.dataset_download",
            side_effect=RuntimeError("network error"),
        ),
    ):
        with pytest.raises(RuntimeError, match="network error"):
            ingest_movies()

    assert any("Dataset download failed" in msg for msg in caplog.messages)


def test_ingest_movies_raises_on_csv_not_found():
    with (
        patch(
            "movie_recommender.ingest.kagglehub.dataset_download",
            return_value="/fake/path",
        ),
        patch(
            "movie_recommender.ingest.find_csv_path",
            side_effect=FileNotFoundError("no csv"),
        ),
    ):
        with pytest.raises(FileNotFoundError):
            ingest_movies()


def test_ingest_movies_raises_on_connection_failure(caplog):
    df = _sample_df()

    with (
        caplog.at_level(logging.ERROR, logger="movie_recommender.ingest"),
        patch(
            "movie_recommender.ingest.kagglehub.dataset_download",
            return_value="/fake/path",
        ),
        patch(
            "movie_recommender.ingest.find_csv_path", return_value="/fake/movies.csv"
        ),
        patch("movie_recommender.ingest.pd.read_csv", return_value=df),
        patch(
            "movie_recommender.ingest.get_collection",
            side_effect=RuntimeError("refused"),
        ),
    ):
        with pytest.raises(RuntimeError):
            ingest_movies()

    assert any("ChromaDB unavailable" in msg for msg in caplog.messages)


def test_ingest_movies_raises_on_upsert_failure(caplog):
    df = _sample_df()
    mock_collection = MagicMock()
    mock_collection.upsert.side_effect = Exception("ChromaDB write failed")
    mock_embed = _mock_embed([[0.1] * 384, [0.2] * 384])

    with (
        caplog.at_level(logging.ERROR, logger="movie_recommender.ingest"),
        patch(
            "movie_recommender.ingest.kagglehub.dataset_download",
            return_value="/fake/path",
        ),
        patch(
            "movie_recommender.ingest.find_csv_path", return_value="/fake/movies.csv"
        ),
        patch("movie_recommender.ingest.pd.read_csv", return_value=df),
        patch("movie_recommender.ingest.get_collection", return_value=mock_collection),
        patch("movie_recommender.ingest.model") as mock_model,
    ):
        mock_model.encode.return_value = mock_embed
        with pytest.raises(Exception, match="ChromaDB write failed"):
            ingest_movies()

    assert any("ChromaDB upsert failed" in msg for msg in caplog.messages)
