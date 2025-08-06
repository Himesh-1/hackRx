"""
Sparse Retriever Module
Implements a sparse retriever using the BM25 algorithm.
"""

import logging
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from chunker import Chunk
from query_parser import ParsedQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparseRetriever:
    """
    A sparse retriever that uses the BM25 algorithm to find relevant documents.
    """

    def __init__(self, chunks: List[Chunk]):
        """
        Initializes the SparseRetriever.

        Args:
            chunks (List[Chunk]): The list of chunks to be indexed.
        """
        if not chunks:
            raise ValueError("Cannot initialize SparseRetriever with an empty list of chunks.")

        self.chunks = chunks
        self.corpus = [chunk.content for chunk in chunks]
        self.tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, parsed_query: ParsedQuery, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """
        Retrieve the most relevant document chunks for a given parsed query.

        Args:
            parsed_query: The ParsedQuery object.
            top_k: The number of top chunks to retrieve.

        Returns:
            A list of tuples, where each tuple contains a Chunk and its retrieval score.
        """
        logger.info(f"Retrieving top {top_k} chunks for query: '{parsed_query.original_query}'")

        tokenized_query = parsed_query.enhanced_query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)

        top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]

        results = [(self.chunks[i], doc_scores[i]) for i in top_n_indices]

        logger.info(f"Retrieved {len(results)} relevant chunks.")
        return results
