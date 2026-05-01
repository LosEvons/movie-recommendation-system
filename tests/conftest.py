from unittest.mock import MagicMock, patch

# Must patch before movie_recommender modules are imported, because they
# instantiate SentenceTransformer at module level.
_st_patcher = patch(
    "sentence_transformers.SentenceTransformer", return_value=MagicMock()
)
_st_patcher.start()
