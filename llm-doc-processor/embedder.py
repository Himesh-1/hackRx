
"""
Embedder Module
Converts text chunks into vector embeddings for semantic search
Supports multiple embedding models and caching
"""

import os
import json
import pickle
import hashlib
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from chunker import Chunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddedChunk:
    """Chunk with its vector embedding"""
    chunk: Chunk
    embedding: np.ndarray
    embedding_model: str
    embedding_dim: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "chunk": asdict(self.chunk),
            "embedding": self.embedding.tolist(),
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddedChunk':
        """Create from dictionary"""
        chunk_data = data["chunk"]
        chunk = Chunk(**chunk_data)
        
        return cls(
            chunk=chunk,
            embedding=np.array(data["embedding"]),
            embedding_model=data["embedding_model"],
            embedding_dim=data["embedding_dim"]
        )

class DocumentEmbedder:
    """
    Creates vector embeddings for text chunks using various embedding models
    """
    
    def __init__(self, 
                 model_name: str = os.getenv("EMBEDDING_MODEL", "models/embedding-001"),
                 embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "gemini"),
                 cache_dir: str = "data/embeddings/",
                 batch_size: int = 32):
        """
        Initialize embedder
        
        Args:
            model_name: Name of the embedding model
            embedding_provider: "sentence_transformers", "gemini"
            cache_dir: Directory to cache embeddings
            batch_size: Batch size for processing chunks
        """
        self.model_name = model_name
        self.embedding_provider = embedding_provider
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model (lazy loading)
        self.model = None
        
        # Cache for embeddings
        self._embedding_cache = {}
        self._load_cache()
    
    def _initialize_model(self):
        """
        Initializes the embedding model based on the configured provider.
        Supports 'sentence_transformers' and 'gemini'.
        Only loads the model if it hasn't been loaded yet.

        Raises:
            ValueError: If an unsupported embedding provider is specified.
            Exception: If there's an error during model initialization.
        """
        if self.model is not None:
            return # Model already loaded

        try:
            if self.embedding_provider == "sentence_transformers":
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
                
            elif self.embedding_provider == "gemini":
                api_key = os.getenv("GEMINI_API_KEYS", "").split(',')[0]
                if not api_key:
                    raise ValueError("GEMINI_API_KEYS environment variable not set or is empty.")
                genai.configure(api_key=api_key)
                self.model = "gemini"  # Placeholder
                logger.info(f"Initialized Gemini embeddings with model: {self.model_name}")
                
            else:
                raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        self._initialize_model() # Ensure model is loaded before use
        """
        Creates embeddings for a list of text chunks.
        It first checks for cached embeddings and processes all new chunks in a single batch.
        """
        embedded_chunks = []
        new_chunks = []
        
        # Check cache first
        for chunk in chunks:
            cache_key = self._get_cache_key(chunk.content)
            if cache_key in self._embedding_cache:
                cached_embedding = self._embedding_cache[cache_key]
                embedded_chunks.append(EmbeddedChunk(
                    chunk=chunk,
                    embedding=cached_embedding["embedding"],
                    embedding_model=cached_embedding["model"],
                    embedding_dim=cached_embedding["dim"]
                ))
            else:
                new_chunks.append(chunk)
        
        logger.info(f"Found {len(embedded_chunks)} cached embeddings, processing {len(new_chunks)} new chunks in a single batch.")
        
        # Process all new chunks in one go
        if new_chunks:
            batch_texts = [chunk.content for chunk in new_chunks]
            try:
                if self.embedding_provider == "sentence_transformers":
                    embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
                elif self.embedding_provider == "gemini":
                    embeddings = self._get_gemini_embeddings(batch_texts)
                
                # Normalize embeddings
                embeddings = normalize(embeddings)

                for chunk, embedding in zip(new_chunks, embeddings):
                    embedded_chunk = EmbeddedChunk(
                        chunk=chunk,
                        embedding=embedding,
                        embedding_model=self.model_name,
                        embedding_dim=len(embedding)
                    )
                    embedded_chunks.append(embedded_chunk)
                    # Cache the new embedding
                    cache_key = self._get_cache_key(chunk.content)
                    self._embedding_cache[cache_key] = {"embedding": embedding, "model": self.model_name, "dim": len(embedding)}

            except Exception as e:
                logger.error(f"Error processing batch of new chunks: {str(e)}")

        # Save updated cache
        self._save_cache()
        
        logger.info(f"Finished embedding process. Total embedded chunks: {len(embedded_chunks)}")
        return embedded_chunks
    
    def _get_gemini_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Retrieves embeddings for a list of texts using the Gemini API.

        Args:
            texts (List[str]): A list of strings to embed.

        Returns:
            np.ndarray: A NumPy array of embeddings.

        Raises:
            Exception: If there's an error during the Gemini API call.
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=texts,
                task_type="retrieval_document"
            )
            return np.array(result['embedding'])
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Creates an embedding for a given query string.

        Args:
            query (str): The query text to embed.

        Returns:
            np.ndarray: The query embedding as a NumPy array.

        Raises:
            Exception: If there's an error during query embedding.
        """
        # This method now simply calls the batch version for a single query.
        return self.embed_queries([query])[0]

    def embed_queries(self, queries: List[str]) -> List[np.ndarray]:
        """ 
        Creates embeddings for a list of query strings in a single batch.

        Args:
            queries (List[str]): The list of query texts to embed.

        Returns:
            List[np.ndarray]: A list of query embeddings as NumPy arrays.
        """
        self._initialize_model() # Ensure model is loaded before use
        logger.info(f"Embedding {len(queries)} queries in a single batch.")
        # Note: Caching is handled at the individual query level if needed, but batching is generally for performance.
        try:
            if self.embedding_provider == "sentence_transformers":
                embeddings = self.model.encode(queries, convert_to_numpy=True)
            elif self.embedding_provider == "gemini":
                embeddings = self._get_gemini_embeddings(queries)
            
            # Normalize embeddings
            embeddings = normalize(embeddings)

            return list(embeddings)

        except Exception as e:
            logger.error(f"Error creating query embeddings in batch: {str(e)}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generates a unique cache key for a given text and model.

        Args:
            text (str): The text content for which to generate the key.

        Returns:
            str: The generated cache key.
        """
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{self.model_name}_{text_hash}"
    
    def _load_cache(self):
        """
        Loads the embedding cache from disk.
        The cache is stored as a pickle file in the specified cache directory.
        """
        cache_file = self.cache_dir / f"embedding_cache_{self.model_name.replace('/', '_')}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self._embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {str(e)}")
                self._embedding_cache = {}
        else:
            self._embedding_cache = {}
    
    def _save_cache(self):
        """
        Saves the current embedding cache to disk.
        The cache is stored as a pickle file in the specified cache directory.
        """
        cache_file = self.cache_dir / f"embedding_cache_{self.model_name.replace('/', '_')}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self._embedding_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {str(e)}")
    
    def save_embeddings(self, embedded_chunks: List[EmbeddedChunk], filename: str = None):
        """
        Saves a list of embedded chunks to disk in JSON format.

        Args:
            embedded_chunks (List[EmbeddedChunk]): A list of EmbeddedChunk objects to save.
            filename (str, optional): The name of the file to save the embeddings to. If None, a default filename is generated.

        Raises:
            Exception: If there's an error during the saving process.
        """
        if not filename:
            filename = f"embeddings_{self.model_name.replace('/', '_')}.json"
        
        filepath = self.cache_dir / filename
        
        try:
            # Convert to serializable format
            data = {
                "metadata": {
                    "model_name": self.model_name,
                    "embedding_provider": self.embedding_provider,
                    "total_chunks": len(embedded_chunks),
                    "embedding_dim": embedded_chunks[0].embedding_dim if embedded_chunks else 0
                },
                "chunks": [chunk.to_dict() for chunk in embedded_chunks]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(embedded_chunks)} embeddings to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {str(e)}")
            raise
    
    def load_embeddings(self, filename: str = None) -> List[EmbeddedChunk]:
        """
        Load embedded chunks from disk
        
        Args:
            filename: Optional filename, otherwise auto-generated
            
        Returns:
            List of EmbeddedChunk objects
        """
        if not filename:
            filename = f"embeddings_{self.model_name.replace('/', '_')}.json"
        
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Embeddings file not found: {filepath}")
            return []
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            embedded_chunks = []
            for chunk_data in data["chunks"]:
                embedded_chunk = EmbeddedChunk.from_dict(chunk_data)
                embedded_chunks.append(embedded_chunk)
            
            logger.info(f"Loaded {len(embedded_chunks)} embeddings from {filepath}")
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {str(e)}")
            raise
    
    def get_embedding_stats(self, embedded_chunks: List[EmbeddedChunk]) -> Dict[str, Any]:
        """
        Get statistics about embeddings
        
        Args:
            embedded_chunks: List of EmbeddedChunk objects
            
        Returns:
            Dictionary with embedding statistics
        """
        if not embedded_chunks:
            return {"total_embeddings": 0}
        
        embeddings = np.array([chunk.embedding for chunk in embedded_chunks])
        
        # Calculate statistics
        stats = {
            "total_embeddings": len(embedded_chunks),
            "embedding_dim": embedded_chunks[0].embedding_dim,
            "model_name": embedded_chunks[0].embedding_model,
            "embedding_stats": {
                "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
                "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
                "min_value": float(np.min(embeddings)),
                "max_value": float(np.max(embeddings)),
                "mean_value": float(np.mean(embeddings))
            }
        }
        
        return stats
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                          metric: str = "cosine") -> float:
        """
        Compute similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric ("cosine", "euclidean", "dot")
            
        Returns:
            Similarity score
        """
        if metric == "cosine":
            return float(np.dot(embedding1, embedding2) / 
                        (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
        elif metric == "dot":
            return float(np.dot(embedding1, embedding2))
        elif metric == "euclidean":
            return float(-np.linalg.norm(embedding1 - embedding2))  # Negative for ranking
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def find_similar_chunks(self, query_embedding: np.ndarray, 
                           embedded_chunks: List[EmbeddedChunk],
                           top_k: int = 5,
                           similarity_metric: str = "cosine") -> List[tuple]:
        """
        Find most similar chunks to a query embedding
        
        Args:
            query_embedding: Query embedding
            embedded_chunks: List of EmbeddedChunk objects
            top_k: Number of top results to return
            similarity_metric: Similarity metric to use
            
        Returns:
            List of (EmbeddedChunk, similarity_score) tuples
        """
        if not embedded_chunks:
            return []
        
        similarities = []
        
        for embedded_chunk in embedded_chunks:
            similarity = self.compute_similarity(
                query_embedding, 
                embedded_chunk.embedding, 
                similarity_metric
            )
            similarities.append((embedded_chunk, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

# Example usage and testing
if __name__ == "__main__":
    from loader import DocumentLoader
    from chunker import DocumentChunker
    
    # Load and chunk documents
    loader = DocumentLoader("docs/")
    documents = loader.load_all_documents()
    
    chunker = DocumentChunker(chunk_size=512, chunking_strategy="smart")
    chunks = chunker.chunk_documents(documents)
    
    if not chunks:
        print("No chunks found. Please add documents to the docs/ directory.")
        exit()
    
    # Test different embedding models
    embedding_configs = [
        ("models/embedding-001", "gemini"),
        ("all-MiniLM-L6-v2", "sentence_transformers"),
    ]
    
    for model_name, provider in embedding_configs:
        print(f"\n=== Testing {model_name} ({provider}) ===")
        
        try:
            embedder = DocumentEmbedder(
                model_name=model_name,
                embedding_provider=provider,
                batch_size=16
            )
            
            # Create embeddings
            embedded_chunks = embedder.embed_chunks(chunks[:10])  # Test with first 10 chunks
            
            # Get statistics
            stats = embedder.get_embedding_stats(embedded_chunks)
            print(f"Created {stats['total_embeddings']} embeddings")
            print(f"Embedding dimension: {stats['embedding_dim']}")
            print(f"Mean norm: {stats['embedding_stats']['mean_norm']:.3f}")
            
            # Test query embedding
            test_query = "insurance coverage for surgery"
            query_embedding = embedder.embed_query(test_query)
            print(f"Query embedding shape: {query_embedding.shape}")
            
            # Find similar chunks
            similar_chunks = embedder.find_similar_chunks(
                query_embedding, embedded_chunks, top_k=5
            )
            
            print(f"\nTop 5 similar chunks for '{test_query}':")
            for i, (chunk, score) in enumerate(similar_chunks):
                print(f"{i+1}. Score: {score:.3f}")
                print(f"   Content: {chunk.chunk.content[:100]}...")
                print(f"   Source: {chunk.chunk.source_document}")
            
            # Save embeddings
            embedder.save_embeddings(embedded_chunks)
            
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
            continue
