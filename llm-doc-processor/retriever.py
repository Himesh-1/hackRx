
"""
Retriever Module
Retrieves relevant document chunks based on a query using a vector index.
"""

import faiss
import numpy as np
import logging
from typing import List, Tuple
from embedder import EmbeddedChunk, DocumentEmbedder
from query_parser import ParsedQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    """
    Retrieves relevant document chunks using a FAISS vector index.
    """

    def __init__(self, embedder: DocumentEmbedder, embedded_chunks: List[EmbeddedChunk]):
        """
        Initialize the Retriever.

        Args:
            embedder: An instance of DocumentEmbedder.
            embedded_chunks: A list of EmbeddedChunk objects to be indexed.
        """
        if not embedded_chunks:
            raise ValueError("Cannot initialize Retriever with an empty list of embedded chunks.")

        self.embedder = embedder
        self.embedded_chunks = embedded_chunks
        self.index = None
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

    def retrieve(self, parsed_query: ParsedQuery, top_k: int = 10) -> List[Tuple[EmbeddedChunk, float]]:
        """
        Retrieve the most relevant document chunks for a given parsed query.

        Args:
            parsed_query: The ParsedQuery object.
            top_k: The number of top chunks to retrieve.

        Returns:
            A list of tuples, where each tuple contains an EmbeddedChunk and its retrieval score.
        """
        logger.info(f"Retrieving top {top_k} chunks for query: '{parsed_query.original_query}'")

        # Use the enhanced query for retrieval
        query_embedding = self.embedder.embed_query(parsed_query.enhanced_query)
        query_embedding = np.array([query_embedding]).astype('float32')

        # Search the index
        distances, indices = self.index.search(query_embedding, top_k)

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

# Example usage (for testing purposes)
if __name__ == '__main__':
    from loader import DocumentLoader
    from chunker import DocumentChunker

    # This is a simplified example flow for testing the retriever
    # In a real application, these components would be managed by a central orchestrator.

    # 1. Load documents
    loader = DocumentLoader("docs/")
    documents = loader.load_all_documents()

    if not documents:
        logger.warning("No documents found in 'docs/' directory. Please add some for testing.")
    else:
        # 2. Chunk documents
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(documents)

        # 3. Embed chunks
        # Using a mock embedder for this example to avoid dependency on API keys or large models
        class MockEmbedder(DocumentEmbedder):
            def __init__(self, model_name: str = "mock-model", dim: int = 384):
                self.model_name = model_name
                self.embedding_dim = dim
            def embed_chunks(self, chunks: List) -> List[EmbeddedChunk]:
                # Generate random embeddings for testing
                return [
                    EmbeddedChunk(
                        chunk=c,
                        embedding=np.random.rand(self.embedding_dim).astype('float32'),
                        embedding_model=self.model_name,
                        embedding_dim=self.embedding_dim
                    ) for c in chunks
                ]
            def embed_query(self, query: str) -> np.ndarray:
                return np.random.rand(self.embedding_dim).astype('float32')

        embedder = MockEmbedder()
        embedded_chunks = embedder.embed_chunks(chunks)

        # 4. Initialize Retriever
        retriever = Retriever(embedder, embedded_chunks)

        # 5. Create a mock parsed query
        mock_parsed_query = ParsedQuery(
            original_query="Is knee surgery covered?",
            enhanced_query="knee surgery coverage policy claim",
            entities={'procedure': 'knee surgery'},
            intent='claim_inquiry',
            keywords=['knee', 'surgery', 'coverage'],
            confidence=0.9
        )

        # 6. Retrieve relevant chunks
        retrieved_results = retriever.retrieve(mock_parsed_query, top_k=3)

        print("Retrieval Test Results ---")
        print(f"Query: '{mock_parsed_query.original_query}'")
        print(f"Found {len(retrieved_results)} results:")

        for chunk, score in retrieved_results:
            print(f"Score: {score:.4f}")
            print(f"  Source: {chunk.chunk.source_document} (Chunk ID: {chunk.chunk.chunk_id})")
            print(f"  Content: {chunk.chunk.content[:150]}...")
