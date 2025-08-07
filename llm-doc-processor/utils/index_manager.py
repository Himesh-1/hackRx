
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
from sparse_retriever import SparseRetriever
from chunker import Chunk # Import Chunk for sparse retriever

logger = logging.getLogger(__name__)

FAISS_INDEX_FILE = Path("data/faiss_index.bin")
EMBEDDED_CHUNKS_FILE = Path("data/embedded_chunks.pkl")
BM25_INDEX_FILE = Path("data/bm25_index.pkl")

def load_or_build_persistent_indices(
    docs_path: str = "docs/",
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers"),
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1600)),
    chunking_strategy: str = os.getenv("CHUNKING_STRATEGY", "hybrid")
) -> Tuple[Retriever, SparseRetriever, List[EmbeddedChunk], List[Chunk]]:
    """
    Loads FAISS and BM25 indices and embedded chunks from disk if they exist and are valid.
    Otherwise, builds them from scratch and saves them.
    """
    retriever = None
    sparse_retriever = None
    embedded_chunks = []
    chunks = []

    # Try loading existing indices
    if FAISS_INDEX_FILE.exists() and EMBEDDED_CHUNKS_FILE.exists() and BM25_INDEX_FILE.exists():
        try:
            logger.info("Attempting to load existing indices from disk...")
            
            # Load embedded chunks
            with open(EMBEDDED_CHUNKS_FILE, 'rb') as f:
                embedded_chunks = pickle.load(f)

            # Load FAISS index
            faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
            
            # Load BM25 index and corpus
            with open(BM25_INDEX_FILE, 'rb') as f:
                bm25_data = pickle.load(f)
                bm25_corpus = bm25_data["corpus"]
                bm25_tokenized_corpus = bm25_data["tokenized_corpus"]
                bm25_idf = bm25_data["idf"]
                bm25_avgdl = bm25_data["avgdl"]
                bm25_doc_len = bm25_data["doc_len"]
                
                # Reconstruct BM25Okapi object
                sparse_retriever_obj = SparseRetriever(bm25_corpus) # Initialize with corpus
                sparse_retriever_obj.tokenized_corpus = bm25_tokenized_corpus
                sparse_retriever_obj.bm25.idf = bm25_idf
                sparse_retriever_obj.bm25.avgdl = bm25_avgdl
                sparse_retriever_obj.bm25.doc_len = bm25_doc_len
                sparse_retriever = sparse_retriever_obj

            # Reconstruct Retriever
            retriever = Retriever(embedded_chunks)
            retriever.index = faiss_index # Assign the loaded FAISS index
            
            # Reconstruct original chunks for sparse retriever if needed later
            # This assumes a 1:1 mapping between embedded_chunks and original chunks
            chunks = [ec.chunk for ec in embedded_chunks]

            logger.info(f"Successfully loaded FAISS index with {retriever.index.ntotal} vectors and {len(embedded_chunks)} embedded chunks.")
            logger.info(f"Successfully loaded BM25 index with {len(bm25_corpus)} documents.")
            return retriever, sparse_retriever, embedded_chunks, chunks

        except Exception as e:
            logger.error(f"Failed to load persistent indices: {e}. Rebuilding from scratch.")
            # Clean up potentially corrupted files
            for f in [FAISS_INDEX_FILE, EMBEDDED_CHUNKS_FILE, BM25_INDEX_FILE]:
                if f.exists():
                    os.remove(f)

    logger.info("Building indices from scratch...")
    loader = DocumentLoader(docs_path)
    documents = loader.load_all_documents()

    if not documents:
        logger.warning("No documents found to build index. Returning empty retrievers.")
        return None, None, [], []

    chunker = DocumentChunker(chunk_size=chunk_size, chunking_strategy=chunking_strategy)
    chunks = chunker.chunk_documents(documents)

    if not chunks:
        logger.warning("No chunks generated. Returning empty retrievers.")
        return None, None, [], []

    embedder = DocumentEmbedder(model_name=embedding_model_name, embedding_provider=embedding_provider)
    embedded_chunks = embedder.embed_chunks(chunks)

    if not embedded_chunks:
        logger.warning("No embedded chunks generated. Returning empty retrievers.")
        return None, None, [], []

    # Build FAISS index
    retriever = Retriever(embedded_chunks)
    # FAISS index is built in Retriever's __init__

    # Build BM25 index
    sparse_retriever = SparseRetriever([c.content for c in chunks])

    # Save indices to disk
    try:
        faiss.write_index(retriever.index, str(FAISS_INDEX_FILE))
        with open(EMBEDDED_CHUNKS_FILE, 'wb') as f:
            pickle.dump(embedded_chunks, f)
        
        # Save BM25 components
        bm25_data = {
            "corpus": sparse_retriever.corpus,
            "tokenized_corpus": sparse_retriever.tokenized_corpus,
            "idf": sparse_retriever.bm25.idf,
            "avgdl": sparse_retriever.bm25.avgdl,
            "doc_len": sparse_retriever.bm25.doc_len,
        }
        with open(BM25_INDEX_FILE, 'wb') as f:
            pickle.dump(bm25_data, f)

        logger.info(f"Built and saved FAISS index with {retriever.index.ntotal} vectors to {FAISS_INDEX_FILE}")
        logger.info(f"Saved {len(embedded_chunks)} embedded chunks to {EMBEDDED_CHUNKS_FILE}")
        logger.info(f"Built and saved BM25 index to {BM25_INDEX_FILE}")

    except Exception as e:
        logger.error(f"Failed to save persistent indices: {e}")

    return retriever, sparse_retriever, embedded_chunks, chunks

if __name__ == "__main__":
    # Example usage:
    # This will build the index if it doesn't exist, or load it if it does.
    retriever_instance, sparse_retriever_instance, embedded_chunks_instance, chunks_instance = load_or_build_persistent_indices()
    if retriever_instance and sparse_retriever_instance:
        print(f"Retriever ready with {len(embedded_chunks_instance)} embedded chunks.")
        print(f"Sparse Retriever ready with {len(chunks_instance)} chunks.")
    else:
        print("Failed to initialize retrievers.")
