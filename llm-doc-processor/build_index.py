
import os
import pickle
import hashlib
import logging
from pathlib import Path
import faiss
import numpy as np
from loader import DocumentLoader
from chunker import DocumentChunker, Chunk
from embedder import DocumentEmbedder, EmbeddedChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_persistent_indices():
    """
    Builds persistent indices for all documents in the docs/ directory.
    """
    logger.info("Starting to build persistent indices...")

    # 1. Load documents
    loader = DocumentLoader("docs/")
    docs = loader.load_all_documents()

    if not docs:
        logger.warning("No documents found in the docs/ directory. Exiting.")
        return

    # 2. Chunk documents
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(docs)

    # 3. Embed chunks
    embedder = DocumentEmbedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    # 4. Build FAISS index
    embedding_dim = embedded_chunks[0].embedding_dim
    index = faiss.IndexFlatIP(embedding_dim)
    embeddings = np.array([chunk.embedding for chunk in embedded_chunks]).astype('float32')
    index.add(embeddings)

    # 5. Build sparse retriever index (BM25)
    corpus = [chunk.content for chunk in chunks]
    from sparse_retriever import SparseRetriever
    sparse_retriever = SparseRetriever(corpus)


    # 6. Save indices and data
    data_dir = Path("data/")
    data_dir.mkdir(exist_ok=True)

    faiss.write_index(index, str(data_dir / "faiss_index.bin"))
    with open(data_dir / "embedded_chunks.pkl", "wb") as f:
        pickle.dump(embedded_chunks, f)
    with open(data_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(data_dir / "bm25_index.pkl", "wb") as f:
        pickle.dump(sparse_retriever.bm25, f)


    # 7. Compute and save document hashes
    doc_hashes = {}
    for doc in docs:
        with open(loader.docs_path / doc.filename, "rb") as f:
            doc_hashes[doc.filename] = hashlib.sha256(f.read()).hexdigest()

    with open(data_dir / "doc_hashes.json", "w") as f:
        import json
        json.dump(doc_hashes, f)

    logger.info("Persistent indices built successfully.")

if __name__ == "__main__":
    build_persistent_indices()
