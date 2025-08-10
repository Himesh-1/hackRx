
import os
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import json

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from loader import DocumentLoader
from chunker import DocumentChunker
from embedder import DocumentEmbedder, EmbeddedChunk
from retriever import Retriever
from sparse_retriever import SparseRetriever
from chunker import Chunk # Import Chunk for sparse retriever
from build_index import build_persistent_indices

logger = logging.getLogger(__name__)

FAISS_INDEX_FILE = Path("data/faiss_index.bin")
EMBEDDED_CHUNKS_FILE = Path("data/embedded_chunks.pkl")
CHUNKS_FILE = Path("data/chunks.pkl")
BM25_INDEX_FILE = Path("data/bm25_index.pkl")
DOC_HASHES_FILE = Path("data/doc_hashes.json")

def load_or_build_persistent_indices() -> Tuple[Retriever, SparseRetriever, List[EmbeddedChunk], List[Chunk]]:
    """
    Loads FAISS and BM25 indices and embedded chunks from disk if they exist.
    Otherwise, builds them from scratch and saves them.
    """
    if not all([FAISS_INDEX_FILE.exists(), EMBEDDED_CHUNKS_FILE.exists(), CHUNKS_FILE.exists(), BM25_INDEX_FILE.exists(), DOC_HASHES_FILE.exists()]):
        logger.info("One or more persistent index files not found. Building from scratch...")
        build_persistent_indices()

    logger.info("Loading persistent indices from disk...")
    
    # Load FAISS index
    faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
    
    # Load embedded chunks
    with open(EMBEDDED_CHUNKS_FILE, 'rb') as f:
        embedded_chunks = pickle.load(f)
        
    # Load chunks
    with open(CHUNKS_FILE, 'rb') as f:
        chunks = pickle.load(f)

    # Load BM25 index
    with open(BM25_INDEX_FILE, 'rb') as f:
        bm25_index = pickle.load(f)

    # Create Retriever instance
    retriever = Retriever(embedded_chunks)
    retriever.index = faiss_index

    # Create SparseRetriever instance
    corpus = [chunk.content for chunk in chunks]
    sparse_retriever = SparseRetriever(corpus)
    sparse_retriever.bm25 = bm25_index


    logger.info(f"Successfully loaded FAISS index with {retriever.index.ntotal} vectors.")
    logger.info(f"Successfully loaded BM25 index.")
    
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
