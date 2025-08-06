
import os
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np

from loader import DocumentLoader
from chunker import DocumentChunker
from embedder import DocumentEmbedder, EmbeddedChunk
from retriever import Retriever

logger = logging.getLogger(__name__)

INDEX_FILE = Path("data/indexed_docs.pkl")
EMBEDDING_CACHE_DIR = Path("data/embeddings")

def load_or_build_index(
    docs_path: str = "docs/",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    embedding_provider: str = "sentence_transformers",
    chunk_size: int = 512,
    chunking_strategy: str = "smart"
) -> Tuple[Retriever, List[EmbeddedChunk]]:
    """
    Loads the FAISS index and embedded chunks from a pickle file if it exists,
    otherwise builds them from scratch and saves them.
    """
    if INDEX_FILE.exists():
        try:
            with open(INDEX_FILE, 'rb') as f:
                data = pickle.load(f)
            embedded_chunks = data["embedded_chunks"]
            faiss_index_bytes = data["faiss_index_bytes"]
            
            # Reconstruct FAISS index
            index = faiss.deserialize_index(faiss_index_bytes)
            
            # Create a dummy embedder for the Retriever, as the model itself is lazy-loaded
            embedder = DocumentEmbedder(model_name=embedding_model_name, embedding_provider=embedding_provider)
            
            # Manually set the index on the retriever instance
            retriever = Retriever(embedder, embedded_chunks)
            retriever.index = index # Assign the loaded index
            
            logger.info(f"Loaded FAISS index with {retriever.index.ntotal} vectors and {len(embedded_chunks)} embedded chunks from {INDEX_FILE}")
            return retriever, embedded_chunks
        except Exception as e:
            logger.error(f"Failed to load index from {INDEX_FILE}: {e}. Rebuilding index.")
            # Fall through to rebuild if loading fails
    
    logger.info("Building FAISS index and embeddings from scratch...")
    loader = DocumentLoader(docs_path)
    documents = loader.load_all_documents()

    if not documents:
        logger.warning("No documents found to build index. Returning empty retriever.")
        # Create a dummy embedder for the Retriever, as the model itself is lazy-loaded
        embedder = DocumentEmbedder(model_name=embedding_model_name, embedding_provider=embedding_provider)
        return Retriever(embedder, []), [] # Return an empty retriever and chunks

    chunker = DocumentChunker(chunk_size=chunk_size, chunking_strategy=chunking_strategy)
    chunks = chunker.chunk_documents(documents)

    embedder = DocumentEmbedder(model_name=embedding_model_name, embedding_provider=embedding_provider)
    embedded_chunks = embedder.embed_chunks(chunks)

    if not embedded_chunks:
        logger.warning("No embedded chunks generated. Returning empty retriever.")
        return Retriever(embedder, []), []

    retriever = Retriever(embedder, embedded_chunks)
    retriever._build_index() # Ensure index is built before serialization
    
    # Serialize FAISS index
    faiss_index_bytes = faiss.serialize_index(retriever.index)

    # Save to pickle file
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump({
            "embedded_chunks": embedded_chunks,
            "faiss_index_bytes": faiss_index_bytes
        }, f)
    logger.info(f"Built and saved FAISS index with {retriever.index.ntotal} vectors and {len(embedded_chunks)} embedded chunks to {INDEX_FILE}")
    
    return retriever, embedded_chunks

if __name__ == "__main__":
    # Example usage:
    # This will build the index if it doesn't exist, or load it if it does.
    # You can then use the retriever for your application.
    retriever_instance, chunks_instance = load_or_build_index()
    print(f"Retriever ready with {len(chunks_instance)} chunks.")
    # You can add a simple test query here if needed
