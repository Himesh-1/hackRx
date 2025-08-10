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

    def rerank(self, parsed_queries: List[ParsedQuery], chunks_with_scores: List[List[Tuple[Union[EmbeddedChunk, Chunk], float]]], top_n: int) -> List[List[Tuple[Union[EmbeddedChunk, Chunk], float]]]:
        """
        Re-ranks a list of retrieved chunks for a list of queries based on their relevance to each query.

        Args:
            parsed_queries (List[ParsedQuery]): The user's parsed queries.
            chunks_with_scores (List[List[Tuple[Union[EmbeddedChunk, Chunk], float]]]): 
                A list of lists of chunks (with their initial scores) from the initial retrieval.
            top_n (int): The number of top chunks to return after re-ranking for each query.

        Returns:
            A new list of lists of chunks, sorted by their re-ranked relevance score.
        """
        if not chunks_with_scores or not any(chunks_with_scores):
            return [[] for _ in parsed_queries]

        logger.info(f"Re-ranking chunks for {len(parsed_queries)} queries.")

        query_chunk_pairs = []
        for i, pq in enumerate(parsed_queries):
            for chunk_obj, _ in chunks_with_scores[i]:
                if isinstance(chunk_obj, EmbeddedChunk):
                    content = chunk_obj.chunk.content
                elif isinstance(chunk_obj, Chunk):
                    content = chunk_obj.content
                else:
                    logger.warning(f"Unsupported chunk type encountered during re-ranking: {type(chunk_obj)}")
                    continue
                query_chunk_pairs.append([pq.original_query, content])

        if not query_chunk_pairs:
            return [[] for _ in parsed_queries]

        scores = self.model.predict(query_chunk_pairs)

        reranked_results = [[] for _ in parsed_queries]
        pair_index = 0
        for i, pq_chunks in enumerate(chunks_with_scores):
            for chunk_obj, _ in pq_chunks:
                reranked_results[i].append((chunk_obj, scores[pair_index]))
                pair_index += 1
            reranked_results[i].sort(key=lambda x: x[1], reverse=True)
            reranked_results[i] = reranked_results[i][:top_n]

        logger.info(f"Re-ranking complete. Returning top {top_n} chunks for each query.")
        return reranked_results
