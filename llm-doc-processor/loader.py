
"""
Document Loader Module
Handles loading and preprocessing of various document formats (PDF, DOCX, TXT)
"""

import os
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx
from urllib.parse import urlparse

from parser import DocumentParser, Document # Import Document and DocumentParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    Loads documents from various file formats and extracts text content
    """
    
    def __init__(self, docs_path: str = "docs/"):
        self.docs_path = Path(docs_path)
        self.supported_formats = {'.pdf', '.docx', '.txt', '.eml', '.msg'}
        self.download_dir = Path("data/downloaded_docs")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.parser = DocumentParser() # Initialize the parser
        
    def load_document(self, file_path: str) -> Document:
        """Load a single document and extract text content
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document object with extracted content and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        logger.info(f"Loading document: {file_path.name}")
        
        try:
            if file_ext == '.pdf':
                return self.parser.parse_pdf(file_path)
            elif file_ext == '.docx':
                return self.parser.parse_docx(file_path)
            elif file_ext == '.txt':
                return self.parser.parse_txt(file_path)
            elif file_ext == '.eml' or file_ext == '.msg':
                return self.parser.parse_email(file_path)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    import httpx
from urllib.parse import urlparse

from parser import DocumentParser, Document # Import Document and DocumentParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    Loads documents from various file formats and extracts text content
    """
    
    def __init__(self, docs_path: str = "docs/"):
        self.docs_path = Path(docs_path)
        self.supported_formats = {'.pdf', '.docx', '.txt', '.eml', '.msg'}
        self.download_dir = Path("data/downloaded_docs")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.parser = DocumentParser() # Initialize the parser
        
    def load_document(self, file_path: str) -> Document:
        """Load a single document and extract text content
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document object with extracted content and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        logger.info(f"Loading document: {file_path.name}")
        
        try:
            if file_ext == '.pdf':
                return self.parser.parse_pdf(file_path)
            elif file_ext == '.docx':
                return self.parser.parse_docx(file_path)
            elif file_ext == '.txt':
                return self.parser.parse_txt(file_path)
            elif file_ext == '.eml' or file_ext == '.msg':
                return self.parser.parse_email(file_path)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    async def load_document_from_url(self, url: str) -> Document:
        """
        Downloads a document from a URL and loads its content asynchronously.
        """
        logger.info(f"Attempting to download document from URL: {url}")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            # Determine filename from URL or headers
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = "downloaded_document"
            
            # Try to get filename from Content-Disposition header
            if 'Content-Disposition' in response.headers:
                cd = response.headers['Content-Disposition']
                fname_match = re.findall(r'filename="(.+)"', cd)
                if fname_match:
                    filename = fname_match[0]

            file_path = self.download_dir / filename
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Successfully downloaded {url} to {file_path}")
            return self.load_document(file_path)

        except httpx.RequestError as e:
            logger.error(f"Error downloading document from {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during URL document loading: {e}")
            raise

    def load_all_documents(self) -> List[Document]:
        """Load all supported documents from the docs directory
        
        Returns:
            List of Document objects
        """
        documents = []
        
        if not self.docs_path.exists():
            logger.warning(f"Documents directory not found: {self.docs_path}")
            return documents
        
        for file_path in self.docs_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    doc = self.load_document(file_path)
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {str(e)}")
                    continue
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about loaded documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {"total_docs": 0}
        
        stats = {
            "total_docs": len(documents),
            "file_types": {},
            "total_content_length": 0,
            "avg_content_length": 0
        }
        
        for doc in documents:
            # Count file types
            if doc.file_type in stats["file_types"]:
                stats["file_types"][doc.file_type] += 1
            else:
                stats["file_types"][doc.file_type] = 1
            
            # Calculate content lengths
            stats["total_content_length"] += len(doc.content)
        
        stats["avg_content_length"] = stats["total_content_length"] / len(documents)
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    loader = DocumentLoader("docs/")
    
    # Load all documents
    docs = loader.load_all_documents()
    
    # Print statistics
    stats = loader.get_document_stats(docs)
    print("Document Statistics:")
    print(f"Total documents: {stats['total_docs']}")
    print(f"File types: {stats['file_types']}")
    print(f"Average content length: {stats['avg_content_length']:.0f} characters")
    
    # Print first few characters of each document
    for doc in docs[:3]:  # Show first 3 documents
        print(f"\n--- {doc.filename} ({doc.file_type}) ---")
        print(doc.content[:200] + "..." if len(doc.content) > 200 else doc.content)


    def load_all_documents(self) -> List[Document]:
        """Load all supported documents from the docs directory
        
        Returns:
            List of Document objects
        """
        documents = []
        
        if not self.docs_path.exists():
            logger.warning(f"Documents directory not found: {self.docs_path}")
            return documents
        
        for file_path in self.docs_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    doc = self.load_document(file_path)
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {str(e)}")
                    continue
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about loaded documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {"total_docs": 0}
        
        stats = {
            "total_docs": len(documents),
            "file_types": {},
            "total_content_length": 0,
            "avg_content_length": 0
        }
        
        for doc in documents:
            # Count file types
            if doc.file_type in stats["file_types"]:
                stats["file_types"][doc.file_type] += 1
            else:
                stats["file_types"][doc.file_type] = 1
            
            # Calculate content lengths
            stats["total_content_length"] += len(doc.content)
        
        stats["avg_content_length"] = stats["total_content_length"] / len(documents)
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    loader = DocumentLoader("docs/")
    
    # Load all documents
    docs = loader.load_all_documents()
    
    # Print statistics
    stats = loader.get_document_stats(docs)
    print("Document Statistics:")
    print(f"Total documents: {stats['total_docs']}")
    print(f"File types: {stats['file_types']}")
    print(f"Average content length: {stats['avg_content_length']:.0f} characters")
    
    # Print first few characters of each document
    for doc in docs[:3]:  # Show first 3 documents
        print(f"\n--- {doc.filename} ({doc.file_type}) ---")
        print(doc.content[:200] + "..." if len(doc.content) > 200 else doc.content)
