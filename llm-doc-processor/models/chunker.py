"""
Text Chunker Module
Splits documents into smaller, semantically meaningful chunks for better retrieval
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .document_loader import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Text chunk with metadata"""
    content: str
    chunk_id: str
    source_document: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Optional[Dict[str, Any]] = None

class DocumentChunker:
    """
    Splits documents into smaller chunks for better semantic retrieval
    Supports multiple chunking strategies
    """
    
    def __init__(self, 
                 chunk_size: int = 512,
                 overlap_size: int = 50,
                 min_chunk_size: int = 100,
                 chunking_strategy: str = "smart"):
        """
        Initialize chunker with configuration
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            overlap_size: Overlap between adjacent chunks
            min_chunk_size: Minimum chunk size to avoid tiny fragments
            chunking_strategy: "fixed", "sentence", or "smart"
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.chunking_strategy = chunking_strategy
        
        # Patterns for smart chunking
        self.section_patterns = [
            r'\n\s*(?:SECTION|Section|章节)\s*\d+',
            r'\n\s*(?:ARTICLE|Article|条款)\s*\d+',
            r'\n\s*(?:CLAUSE|Clause|条件)\s*\d+',
            r'\n\s*\d+\.\s+[A-Z]',  # Numbered sections
            r'\n\s*[A-Z]+\.\s+[A-Z]',  # Letter sections
        ]
        
        self.sentence_endings = r'[.!?]+\s+'
        
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk a single document
        
        Args:
            document: Document object to chunk
            
        Returns:
            List of Chunk objects
        """
        if self.chunking_strategy == "fixed":
            return self._chunk_fixed_size(document)
        elif self.chunking_strategy == "sentence":
            return self._chunk_by_sentences(document)
        elif self.chunking_strategy == "smart":
            return self._chunk_smart(document)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunking_strategy}")
    
    def _chunk_fixed_size(self, document: Document) -> List[Chunk]:
        """Split document into fixed-size chunks with overlap"""
        chunks = []
        content = document.content
        
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            
            # Try to end at a word boundary
            if end < len(content):
                last_space = content.rfind(' ', start, end)
                if last_space > start + self.min_chunk_size:
                    end = last_space
            
            chunk_content = content[start:end].strip()
            
            if len(chunk_content) >= self.min_chunk_size:
                chunk = Chunk(
                    content=chunk_content,
                    chunk_id=f"{document.filename}_{chunk_index}",
                    source_document=document.filename,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "chunking_strategy": "fixed",
                        "original_file_type": document.file_type
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - self.overlap_size)
            
            # Prevent infinite loop
            if start >= len(content):
                break
        
        return chunks
    
    def _chunk_by_sentences(self, document: Document) -> List[Chunk]:
        """Split document into chunks based on sentence boundaries"""
        chunks = []
        content = document.content
        
        # Split by sentences
        sentences = re.split(self.sentence_endings, content)
        
        current_chunk = ""
        chunk_index = 0
        start_char = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk = Chunk(
                    content=current_chunk.strip(),
                    chunk_id=f"{document.filename}_{chunk_index}",
                    source_document=document.filename,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(current_chunk),
                    metadata={
                        "chunking_strategy": "sentence",
                        "original_file_type": document.file_type
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Handle overlap
                overlap_text = self._get_sentence_overlap(current_chunk)
                current_chunk = overlap_text + sentence
                start_char += len(current_chunk) - len(overlap_text)
            else:
                current_chunk += (" " if current_chunk else "") + sentence
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk = Chunk(
                content=current_chunk.strip(),
                chunk_id=f"{document.filename}_{chunk_index}",
                source_document=document.filename,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(current_chunk),
                metadata={
                    "chunking_strategy": "sentence",
                    "original_file_type": document.file_type
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_smart(self, document: Document) -> List[Chunk]:
        """
        Smart chunking that preserves semantic boundaries
        Tries to split at section headers, then paragraphs, then sentences
        """
        chunks = []
        content = document.content
        
        # Try to identify major sections first
        sections = self._identify_sections(content)
        
        if len(sections) > 1:
            # Process each section separately
            for i, (section_start, section_end, section_title) in enumerate(sections):
                section_content = content[section_start:section_end]
                section_chunks = self._chunk_section(
                    section_content, 
                    document, 
                    section_start, 
                    i, 
                    section_title
                )
                chunks.extend(section_chunks)
        else:
            # No clear sections, chunk by paragraphs
            chunks = self._chunk_by_paragraphs(document)
        
        return chunks
    
    def _identify_sections(self, content: str) -> List[tuple]:
        """Identify major sections in the document"""
        sections = []
        
        for pattern in self.section_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE))
            
            if matches:
                for i, match in enumerate(matches):
                    start = match.start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
                    title = match.group().strip()
                    sections.append((start, end, title))
                break  # Use first pattern that matches
        
        return sections
    
    def _chunk_section(self, section_content: str, document: Document, 
                       offset: int, section_idx: int, section_title: str) -> List[Chunk]:
        """Chunk a specific section"""
        chunks = []
        
        if len(section_content) <= self.chunk_size:
            # Section fits in one chunk
            chunk = Chunk(
                content=section_content.strip(),
                chunk_id=f"{document.filename}_s{section_idx}_0",
                source_document=document.filename,
                chunk_index=section_idx * 1000,  # Leave room for sub-chunks
                start_char=offset,
                end_char=offset + len(section_content),
                metadata={
                    "chunking_strategy": "smart_section",
                    "section_title": section_title,
                    "original_file_type": document.file_type
                }
            )
            chunks.append(chunk)
        else:
            # Split section into smaller chunks
            temp_doc = Document(
                content=section_content,
                filename=f"{document.filename}_section_{section_idx}",
                file_type=document.file_type
            )
            
            section_chunks = self._chunk_by_paragraphs(temp_doc)
            
            # Update metadata and indices
            for i, chunk in enumerate(section_chunks):
                chunk.chunk_id = f"{document.filename}_s{section_idx}_{i}"
                chunk.source_document = document.filename
                chunk.chunk_index = section_idx * 1000 + i
                chunk.start_char += offset
                chunk.end_char += offset
                chunk.metadata.update({
                    "section_title": section_title,
                    "chunking_strategy": "smart_section"
                })
            
            chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_by_paragraphs(self, document: Document) -> List[Chunk]:
        """Chunk document by paragraphs, combining small ones"""
        chunks = []
        content = document.content
        
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        start_char = 0
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk = Chunk(
                    content=current_chunk.strip(),
                    chunk_id=f"{document.filename}_{chunk_index}",
                    source_document=document.filename,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(current_chunk),
                    metadata={
                        "chunking_strategy": "smart_paragraph",
                        "original_file_type": document.file_type
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
                
                current_chunk = paragraph
                start_char += len(current_chunk)
            else:
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk = Chunk(
                content=current_chunk.strip(),
                chunk_id=f"{document.filename}_{chunk_index}",
                source_document=document.filename,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(current_chunk),
                metadata={
                    "chunking_strategy": "smart_paragraph",
                    "original_file_type": document.file_type
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_sentence_overlap(self, text: str) -> str:
        """Get last few sentences for overlap"""
        sentences = re.split(self.sentence_endings, text)
        overlap_sentences = sentences[-2:] if len(sentences) > 1 else sentences[-1:]
        return " ".join(overlap_sentences).strip()
    
    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {"total_chunks": 0}
        
        lengths = [len(chunk.content) for chunk in chunks]
        strategies = {}
        
        for chunk in chunks:
            strategy = chunk.metadata.get("chunking_strategy", "unknown")
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(lengths) / len(lengths),
            "min_chunk_length": min(lengths),
            "max_chunk_length": max(lengths),
            "strategies_used": strategies
        }

# Example usage and testing
if __name__ == "__main__":
    from document_loader import DocumentLoader
    
    # Load documents
    loader = DocumentLoader("docs/")
    documents = loader.load_all_documents()
    
    # Test different chunking strategies
    strategies = ["fixed", "sentence", "smart"]
    
    for strategy in strategies:
        print(f"\n=== Testing {strategy} chunking ===")
        
        chunker = DocumentChunker(
            chunk_size=512,
            overlap_size=50,
            chunking_strategy=strategy
        )
        
        chunks = chunker.chunk_documents(documents)
        stats = chunker.get_chunk_stats(chunks)
        
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Average chunk length: {stats['avg_chunk_length']:.0f}")
        print(f"Min/Max chunk length: {stats['min_chunk_length']}/{stats['max_chunk_length']}")
        
        # Show first few chunks
        for i, chunk in enumerate(chunks[:2]):
            print(f"\nChunk {i+1} ({chunk.chunk_id}):")
            print(chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content)