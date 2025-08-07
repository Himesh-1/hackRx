"""
Re-ranker Module

This module uses a Cross-Encoder model to re-rank the results from the initial
retrieval step. Cross-Encoders are more accurate than standard sentence embeddings
for determining the relevance of a document chunk to a query, leading to a
significant improvement in answer quality.
"""

import logging
from typing import List, Tuple, Union
from sentence_transformers import CrossEncoder
from query_parser import ParsedQuery
from embedder import EmbeddedChunk
from chunker import Chunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReRanker:
    """
    A re-ranking component that uses a Cross-Encoder model to improve the relevance
    of retrieved document chunks.
    """

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'):
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

    def rerank(self, parsed_query: ParsedQuery, chunks_with_scores: List[Tuple[Union[EmbeddedChunk, Chunk], float]], top_n: int = 7) -> List[Tuple[Union[EmbeddedChunk, Chunk], float]]:
        """
        Re-ranks a list of retrieved chunks based on their relevance to the query.

        Args:
            parsed_query (ParsedQuery): The user's parsed query.
            chunks_with_scores (List[Tuple[Union[EmbeddedChunk, Chunk], float]]): 
                The list of chunks (with their initial scores) from the initial retrieval.
            top_n (int): The number of top chunks to return after re-ranking.

        Returns:
            A new list of chunks, sorted by their re-ranked relevance score.
        """
        if not chunks_with_scores:
            return []

        logger.info(f"Re-ranking {len(chunks_with_scores)} chunks for query: '{parsed_query.original_query}'")

        # Extract chunk content for the Cross-Encoder model
        chunk_contents = []
        for chunk_obj, _ in chunks_with_scores:
            if isinstance(chunk_obj, EmbeddedChunk):
                chunk_contents.append(chunk_obj.chunk.content)
            elif isinstance(chunk_obj, Chunk):
                chunk_contents.append(chunk_obj.content)
            else:
                logger.warning(f"Unsupported chunk type encountered during re-ranking: {type(chunk_obj)}")
                continue

        # Create pairs of [query, chunk_content] for the model
        query_chunk_pairs = [[parsed_query.original_query, content] for content in chunk_contents]

        # Get the scores from the Cross-Encoder model
        scores = self.model.predict(query_chunk_pairs)

        # Combine original chunk objects with their new scores
        reranked_chunks = []
        for i, (chunk_obj, original_score) in enumerate(chunks_with_scores):
            reranked_chunks.append((chunk_obj, scores[i]))

        # Sort the chunks by the new score in descending order
        reranked_chunks.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Re-ranking complete. Returning top {top_n} chunks.")
        return reranked_chunks[:top_n]
