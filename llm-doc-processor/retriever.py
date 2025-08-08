"""
Retriever Module
Retrieves relevant document chunks based on a query using a vector index.
"""

import faiss
import numpy as np
import logging
from typing import List, Tuple
from embedder import EmbeddedChunk
from query_parser import ParsedQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    """
    Retrieves relevant document chunks using a FAISS vector index.
    """

    def __init__(self, embedded_chunks: List[EmbeddedChunk]):
        """
        Initialize the Retriever.

        Args:
            embedded_chunks: A list of EmbeddedChunk objects to be indexed.
        """
        if not embedded_chunks:
            raise ValueError("Cannot initialize Retriever with an empty list of embedded chunks.")

        self.embedded_chunks = embedded_chunks
        self._build_index()

    def _build_index(self):
        """
        Builds the FAISS index from the embedded chunks.
        The index is an IndexFlatIP, suitable for inner product (cosine similarity) search.
        """
        logger.info("Building FAISS index...")
        embedding_dim = self.embedded_chunks[0].embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Using Inner Product for cosine similarity

        # Add embeddings to the index
        embeddings = np.array([chunk.embedding for chunk in self.embedded_chunks]).astype('float32')
        self.index.add(embeddings)

        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors.")

    def retrieve(self, parsed_query: ParsedQuery, top_k: int, query_embedding: np.ndarray = None) -> List[Tuple[EmbeddedChunk, float]]:
        """
        Retrieve the most relevant document chunks for a given parsed query using a pre-computed embedding.

        Args:
            parsed_query: The ParsedQuery object (used for logging).
            top_k: The number of top chunks to retrieve.
            query_embedding: The pre-computed embedding for the query.

        Returns:
            A list of tuples, where each tuple contains an EmbeddedChunk and its retrieval score.
        """
        logger.info(f"Retrieving top {top_k} chunks for query: '{parsed_query.original_query}'")

        if self.index is None:
            logger.error("CRITICAL: FAISS index is not built.")
            raise RuntimeError("FAISS index must be built before retrieval.")

        if query_embedding is None:
            logger.error("CRITICAL: No pre-computed query embedding provided to retrieve method.")
            raise ValueError("A pre-computed query embedding is required for retrieval.")
        
        query_embedding_np = np.array([query_embedding]).astype('float32')

        # Search the index
        distances, indices = self.index.search(query_embedding_np, top_k)

        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                retrieved_chunk = self.embedded_chunks[idx]
                score = distances[0][i]
                results.append((retrieved_chunk, score))
                logger.debug(f"Retrieved chunk {retrieved_chunk.chunk.chunk_id} with score {score}")

        logger.info(f"Retrieved {len(results)} relevant chunks.")
        return results