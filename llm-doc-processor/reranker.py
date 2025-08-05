
"""
Re-ranker Module

This module uses a Cross-Encoder model to re-rank the results from the initial
retrieval step. Cross-Encoders are more accurate than standard sentence embeddings
for determining the relevance of a document chunk to a query, leading to a
significant improvement in answer quality.
"""

import logging
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from embedder import EmbeddedChunk
from query_parser import ParsedQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReRanker:
    """
    A re-ranking component that uses a Cross-Encoder model to improve the relevance
    of retrieved document chunks.
    """

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initializes the ReRanker.

        Args:
            model_name (str): The name of the Cross-Encoder model to use.
        """
        try:
            logger.info(f"Loading Cross-Encoder model: {model_name}")
            self.model = CrossEncoder(model_name)
            logger.info("Cross-Encoder model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model: {e}", exc_info=True)
            raise

    def rerank(self, parsed_query: ParsedQuery, chunks: List[Tuple[EmbeddedChunk, float]], top_n: int = 7) -> List[Tuple[EmbeddedChunk, float]]:
        """
        Re-ranks a list of retrieved chunks based on their relevance to the query.

        Args:
            parsed_query (ParsedQuery): The user's parsed query.
            chunks (List[Tuple[EmbeddedChunk, float]]): The list of chunks from the initial retrieval.
            top_n (int): The number of top chunks to return after re-ranking.

        Returns:
            A new list of chunks, sorted by their re-ranked relevance score.
        """
        if not chunks:
            return []

        logger.info(f"Re-ranking {len(chunks)} chunks for query: '{parsed_query.original_query}'")

        # Create pairs of [query, chunk_content] for the model
        query_chunk_pairs = [[parsed_query.original_query, chunk[0].chunk.content] for chunk in chunks]

        # Get the scores from the Cross-Encoder model
        scores = self.model.predict(query_chunk_pairs)

        # Add the new scores to the chunks
        for i in range(len(chunks)):
            # The new tuple will be (EmbeddedChunk, new_score)
            chunks[i] = (chunks[i][0], scores[i])

        # Sort the chunks by the new score in descending order
        chunks.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Re-ranking complete. Returning top {top_n} chunks.")
        return chunks[:top_n]
