from unittest.mock import MagicMock, patch

import pytest

import movie_recommender.chroma as chroma_module
from movie_recommender.chroma import get_collection


@pytest.fixture(autouse=True)
def reset_collection_cache():
    """Wipe the module-level collection cache before and after each test."""
    chroma_module._collection = None
    yield
    chroma_module._collection = None


def _mock_client(collection: MagicMock | None = None) -> MagicMock:
    client = MagicMock()
    client.get_or_create_collection.return_value = collection or MagicMock()
    return client


# success paths


def test_get_collection_connects_and_returns_collection():
    mock_collection = MagicMock()
    client = _mock_client(mock_collection)

    with patch("movie_recommender.chroma.chromadb.HttpClient", return_value=client):
        result = get_collection()

    assert result is mock_collection
    client.get_or_create_collection.assert_called_once_with(name="movies")


def test_get_collection_caches_on_second_call():
    client = _mock_client()

    with patch("movie_recommender.chroma.chromadb.HttpClient", return_value=client):
        first = get_collection()
        second = get_collection()

    assert first is second
    # HttpClient and get_or_create_collection each called only once
    assert client.get_or_create_collection.call_count == 1


# retry paths


def test_get_collection_retries_on_transient_failure():
    mock_collection = MagicMock()
    client = MagicMock()
    client.get_or_create_collection.side_effect = [
        ConnectionError("timeout"),
        mock_collection,
    ]

    with (
        patch("movie_recommender.chroma.chromadb.HttpClient", return_value=client),
        patch("movie_recommender.chroma.time.sleep") as mock_sleep,
        patch.object(chroma_module, "CHROMA_RETRIES", 3),
    ):
        result = get_collection()

    assert result is mock_collection
    mock_sleep.assert_called_once()


def test_get_collection_sleeps_between_retries():
    client = MagicMock()
    client.get_or_create_collection.side_effect = [
        ConnectionError("timeout"),
        ConnectionError("timeout"),
        MagicMock(),
    ]

    with (
        patch("movie_recommender.chroma.chromadb.HttpClient", return_value=client),
        patch("movie_recommender.chroma.time.sleep") as mock_sleep,
        patch.object(chroma_module, "CHROMA_RETRIES", 3),
    ):
        get_collection()

    assert mock_sleep.call_count == 2


def test_get_collection_no_sleep_after_last_attempt():
    client = MagicMock()
    client.get_or_create_collection.side_effect = ConnectionError("refused")

    with (
        patch("movie_recommender.chroma.chromadb.HttpClient", return_value=client),
        patch("movie_recommender.chroma.time.sleep") as mock_sleep,
        patch.object(chroma_module, "CHROMA_RETRIES", 2),
    ):
        with pytest.raises(RuntimeError):
            get_collection()

    # sleep called once (after attempt 1), not after the final attempt
    assert mock_sleep.call_count == 1


def test_get_collection_raises_runtime_error_after_all_retries():
    client = MagicMock()
    client.get_or_create_collection.side_effect = ConnectionError("refused")

    with (
        patch("movie_recommender.chroma.chromadb.HttpClient", return_value=client),
        patch("movie_recommender.chroma.time.sleep"),
        patch.object(chroma_module, "CHROMA_RETRIES", 2),
    ):
        with pytest.raises(RuntimeError, match="Could not connect to ChromaDB"):
            get_collection()

    assert client.get_or_create_collection.call_count == 2


def test_get_collection_original_error_chained():
    original = ConnectionError("connection refused")
    client = MagicMock()
    client.get_or_create_collection.side_effect = original

    with (
        patch("movie_recommender.chroma.chromadb.HttpClient", return_value=client),
        patch("movie_recommender.chroma.time.sleep"),
        patch.object(chroma_module, "CHROMA_RETRIES", 1),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            get_collection()

    assert exc_info.value.__cause__ is original
