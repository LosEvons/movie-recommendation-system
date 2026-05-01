from unittest.mock import MagicMock, patch

# Must patch before movie_recommender modules are imported, because they
# instantiate TextEmbedding at module level.
_st_patcher = patch("fastembed.TextEmbedding", return_value=MagicMock())
_st_patcher.start()
