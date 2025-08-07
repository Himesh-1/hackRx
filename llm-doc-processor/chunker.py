
"""
Text Chunker Module
Splits documents into smaller, semantically meaningful chunks for better retrieval
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loader import Document

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
    Splits documents into smaller chunks for better semantic retrieval.
    Supports multiple chunking strategies.
    """
    
    def __init__(self,
                 chunk_size: int = 2240, # Approximately 560 tokens (changed from 1600 to 2240)
                 overlap_size: int = 400, # Approximately 25% overlap
                 min_chunk_size: int = 100,
                 chunking_strategy: str = "hybrid"):
        """
        Initialize chunker with configuration.
        
        Args:
            chunk_size: Target size for each chunk (in characters).
            overlap_size: Overlap between adjacent chunks.
            min_chunk_size: Minimum chunk size to avoid tiny fragments.
            chunking_strategy: "fixed", "sentence", "smart", or "hybrid".
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.chunking_strategy = chunking_strategy
        
        # Enhanced patterns for insurance policy document chunking
        self.section_patterns = [
            # Standard policy section headers
            r'\n\s*(?:SECTION|Section|ARTICLE|Article|CLAUSE|Clause|PART|Part)\s+[\d.]+\s*[-:.]?\s*([A-Z][A-Za-z\s]+)',
            # Insurance-specific sections
            r'\n\s*(?:COVERAGE|BENEFITS|EXCLUSIONS|WAITING PERIOD|ELIGIBILITY|CLAIMS)\s*[-:.]?\s*([A-Z][A-Za-z\s]+)',
            # Numbered clauses with titles
            r'\n\s*\d+(?:\.\d+)*\s+([A-Z][A-Za-z\s]+)',
            # Policy terms and conditions
            r'\n\s*(?:TERMS|CONDITIONS|DEFINITIONS|SCOPE)\s*[-:.]?\s*([A-Z][A-Za-z\s]+)',
            # Procedure specific sections
            r'\n\s*(?:SURGICAL|MEDICAL|EMERGENCY|DAYCARE)\s+(?:PROCEDURES|TREATMENTS|COVERAGE)\s*[-:.]?\s*([A-Z][A-Za-z\s]+)'
        ]
        
        # Enhanced sentence detection for policy documents
        self.sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        
        # Enhanced list pattern for policy terms
        self.list_item_pattern = r'\n\s*(?:\d+\.\d*|\d+\)|[a-z]\)|[-*•]|\([i|v|x]+\))\s+'
        
        # Enhanced insurance-specific metadata patterns
        self.metadata_patterns = {
            # Waiting periods and policy duration
            'waiting_period': r'(?:waiting\s+period|cooling\s+period)\s+of\s+(\d+)\s*(?:months|years)',
            'policy_duration': r'(?:insured|policy|coverage|covered|with)\s+(?:for|since|over)\s+(?:about|around|approximately)?\s*(\d+)\s*(?:months?|years?)',
            
            # Coverage and costs
            'coverage_limit': r'(?:coverage|limit|sum\s+insured|cost|expenses?|bill|amount)\s+(?:up\s+to|around|approximately|roughly)?\s*(?:Rs\.|INR|₹)\s*(\d+(?:,\d{3})*)',
            'treatment_cost': r'(?:costs?|charges?|expenses?|bill|amount|total)\s*(?:will|would|may|might|is|around|approximately|roughly)?\s*(?:be|come\s+to)?\s*(?:Rs\.|INR|₹)\s*(\d+(?:,\d{3})*)',
            
            # Medical procedure details
            'procedure_type': r'(?:surgery|operation|procedure|treatment)\s+(?:for|of|related\s+to)?\s*([a-zA-Z\s]+)',
            'procedure_method': r'(?:laparoscopic|open|minimally\s+invasive|robotic|arthroscopic)\s+(?:surgery|procedure|operation)(?:\s+for\s+([a-zA-Z\s]+))?',
            'medical_condition': r'(?:diagnosed\s+with|suffering\s+from|having|for|treating|treatment\s+for)\s+([a-zA-Z\s]+)',
            
            # Hospital and location
            'hospital_name': r'(?:at|in)\s+([A-Z][a-zA-Z\s]+(?:Hospital|Medical|Healthcare|Clinic))',
            'location': r'(?:in|at)\s+([A-Z][a-zA-Z]+(?:\s+City)?)\n',
            
            # Age limits and restrictions
            'age_limit': r'(?:age|aged?)\s*(?:limit|restriction)?\s*(?:between|from)?\s*(\d+)(?:\s*-\s*|\s+to\s+)?(\d+)?(?:\s*years?)?',
            
            # Claim type and status
            'claim_type': r'(?:cashless|reimbursement)\s+(?:claim|process|treatment)',
            'pre_existing': r'(?:no|any|having|with)\s+pre-?existing\s+conditions?',
            
            # Treatment timing
            'admission_date': r'(?:admitted|admission|scheduled)\s+(?:on|around|by)\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)',
            'treatment_duration': r'(?:for|about|around)\s+(\d+)\s*(?:days?|weeks?|nights?)'
        }

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Chunks a list of documents into smaller, semantically meaningful chunks.

        Args:
            documents (List[Document]): A list of Document objects to be chunked.

        Returns:
            List[Chunk]: A list of Chunk objects, representing the chunked content of the input documents.
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunks a single document based on the chosen strategy.

        Args:
            document (Document): The document to chunk.

        Returns:
            List[Chunk]: A list of Chunk objects generated from the document.
        """
        if self.chunking_strategy == "fixed":
            return self._chunk_fixed_size(document)
        elif self.chunking_strategy == "sentence":
            return self._chunk_by_sentences(document)
        elif self.chunking_strategy == "smart":
            return self._chunk_smart(document)
        elif self.chunking_strategy == "hybrid":
            return self._chunk_hybrid(document)
        else:
            logger.warning(f"Unknown chunking strategy '{self.chunking_strategy}'. Defaulting to 'hybrid'.")
            return self._chunk_hybrid(document)

    def _chunk_fixed_size(self, document: Document) -> List[Chunk]:
        """
        Splits the document into fixed-size chunks with a specified overlap.

        Args:
            document (Document): The document object to be chunked.

        Returns:
            List[Chunk]: A list of Chunk objects.
        """
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
        """
        Splits the document into chunks based on sentence boundaries.

        Args:
            document (Document): The document object to be chunked.

        Returns:
            List[Chunk]: A list of Chunk objects.
        """
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
        Performs smart chunking that preserves semantic boundaries.
        It attempts to split the document at section headers, then paragraphs, and finally sentences.

        Args:
            document (Document): The document object to be chunked.

        Returns:
            List[Chunk]: A list of Chunk objects, semantically chunked.
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

    def _chunk_hybrid(self, document: Document) -> List[Chunk]:
        """
        Implements an advanced chunking strategy optimized for insurance policy documents.
        It first splits by major sections, then by insurance-specific blocks if no standard sections are found.
        Further splits sections by paragraphs and lists, and extracts policy-related metadata.

        Args:
            document (Document): The document object to be chunked.

        Returns:
            List[Chunk]: A list of Chunk objects, chunked using the hybrid strategy.
        """
        text = document.content
        chunks = []
        
        # First, split by major sections
        sections = self._identify_sections(text)
        
        # If no standard sections found, try to identify insurance-specific blocks
        if not sections:
            insurance_blocks = self._identify_insurance_blocks(text)
            if insurance_blocks:
                sections = insurance_blocks
            else:
                sections = [(0, len(text), "Full Document")]
                
        # Process each section
        chunk_index = 0
        for start, end, title in sections:
            section_text = text[start:end]
            
            # Further split section by paragraphs and lists
            elements = re.split(r'(\n\s*\n)', section_text) # Split by blank lines
            
            current_chunk_content = ""
            current_chunk_start = start

            for i in range(0, len(elements), 2):
                element = elements[i]
                if not element.strip():
                    continue

                if len(current_chunk_content) + len(element) > self.chunk_size and current_chunk_content:
                    # Create chunk with insurance-specific metadata
                    metadata = {
                        "chunking_strategy": "hybrid",
                        "section_title": title,
                        "original_file_type": document.file_type,
                        "policy_metadata": {}
                    }
                    
                    # Extract insurance metadata from the chunk content
                    if any(term in current_chunk_content.lower() for term in ['waiting period', 'coverage', 'limit', 'age']):
                        # Extract waiting periods
                        waiting_matches = re.finditer(self.metadata_patterns['waiting_period'], current_chunk_content, re.IGNORECASE)
                        waiting_periods = [(int(m.group(1)), m.group(0)) for m in waiting_matches]
                        if waiting_periods:
                            metadata['policy_metadata']['waiting_periods'] = waiting_periods
                        
                        # Extract coverage limits
                        coverage_matches = re.finditer(self.metadata_patterns['coverage_limit'], current_chunk_content, re.IGNORECASE)
                        coverage_limits = [(int(m.group(1).replace(',', '')), m.group(0)) for m in coverage_matches]
                        if coverage_limits:
                            metadata['policy_metadata']['coverage_limits'] = coverage_limits
                        
                        # Extract age limits
                        age_matches = re.finditer(self.metadata_patterns['age_limit'], current_chunk_content, re.IGNORECASE)
                        age_limits = [(m.group(1), m.group(2) if m.group(2) else None, m.group(0)) for m in age_matches]
                        if age_limits:
                            metadata['policy_metadata']['age_limits'] = age_limits
                    
                    chunks.append(Chunk(
                        content=f"{title}. {current_chunk_content.strip()}",
                        chunk_id=f"{document.filename}_{chunk_index}",
                        source_document=document.filename,
                        chunk_index=chunk_index,
                        start_char=current_chunk_start,
                        end_char=current_chunk_start + len(current_chunk_content),
                        metadata=metadata
                    ))
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_sentence_overlap(current_chunk_content)
                    current_chunk_content = overlap_text + "\n\n" + element
                    current_chunk_start += len(current_chunk_content) - len(overlap_text) - len(element) - 4
                else:
                    if not current_chunk_content:
                        current_chunk_start = start + section_text.find(element)
                    current_chunk_content += element + "\n\n"

            # Add the last remaining chunk if it exists
            if current_chunk_content.strip() and len(current_chunk_content.strip()) >= self.min_chunk_size:
                metadata = {
                    "chunking_strategy": "hybrid",
                    "section_title": title,
                    "original_file_type": document.file_type,
                    "policy_metadata": {}
                }
                chunks.append(Chunk(
                    content=f"{title}. {current_chunk_content.strip()}",
                    chunk_id=f"{document.filename}_{chunk_index}",
                    source_document=document.filename,
                    chunk_index=chunk_index,
                    start_char=current_chunk_start,
                    end_char=current_chunk_start + len(current_chunk_content),
                    metadata=metadata
                ))
        
        return chunks

    def _identify_insurance_blocks(self, content: str) -> List[tuple]:
        """Identify insurance-specific content blocks"""
        blocks = []
        
        # Enhanced insurance document patterns
        block_patterns = [
            # Coverage and Benefits
            (r'(?i)coverage\s+details?', 'Coverage Details'),
            (r'(?i)benefits?\s+covered', 'Covered Benefits'),
            (r'(?i)sum\s+insured', 'Sum Insured'),
            
            # Medical Procedures
            (r'(?i)surgical\s+procedures?', 'Surgical Procedures'),
            (r'(?i)laparoscopic\s+procedures?', 'Laparoscopic Procedures'),
            (r'(?i)(?:appendix|appendicitis)\s+(?:surgery|treatment)', 'Appendicitis Treatment'),
            (r'(?i)planned\s+surgeries?', 'Planned Surgeries'),
            (r'(?i)daycare\s+procedures?', 'Daycare Procedures'),
            
            # Claims Process
            (r'(?i)claims?\s+(?:process|procedure)', 'Claims Process'),
            (r'(?i)(?:set\s+up\s+)?cashless\s+claims?', 'Cashless Claims'),
            (r'(?i)pre-auth\w*\s+(?:process|procedure)', 'Pre-Authorization Process'),
            (r'(?i)(?:document|papers?|forms?)\s+(?:requirements?|checklist|needed|required)', 'Required Documents'),
            
            # Conditions and Restrictions
            (r'(?i)waiting\s+periods?', 'Waiting Periods'),
            (r'(?i)exclusions?', 'Exclusions'),
            (r'(?i)pre-existing\s+conditions?', 'Pre-existing Conditions'),
            
            # Network Hospitals
            (r'(?i)network\s+hospitals?', 'Network Hospitals'),
            (r'(?i)empanell?ed\s+hospitals?', 'Empanelled Hospitals'),
            
            # Specific Treatments
            (r'(?i)emergency\s+treatment', 'Emergency Treatment'),
            (r'(?i)planned\s+hospitalization', 'Planned Hospitalization'),
            (r'(?i)treatment\s+guidelines?', 'Treatment Guidelines')
        ]
        
        last_end = 0
        for pattern, title in block_patterns:
            matches = list(re.finditer(pattern, content))
            for match in matches:
                start = match.start()
                # Find the next section start or use document end
                next_starts = [m.start() for m in matches if m.start() > start]
                end = min(next_starts) if next_starts else len(content)
                if start > last_end:  # Avoid overlapping blocks
                    blocks.append((start, end, title))
                    last_end = end
        
        return sorted(blocks, key=lambda x: x[0])

    def _identify_sections(self, content: str) -> List[tuple]:
        """
        Identifies major sections in the document based on predefined patterns.

        Args:
            content (str): The text content of the document.

        Returns:
            List[tuple]: A list of tuples, each containing (start_char, end_char, title) of the identified section.
        """
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
        """
        Chunks a specific section with enhanced insurance policy understanding.

        Args:
            section_content (str): The content of the section to chunk.
            document (Document): The original document object.
            offset (int): The starting character offset of the section within the original document.
            section_idx (int): The index of the section.
            section_title (str): The title of the section.

        Returns:
            List[Chunk]: A list of chunks generated from the section.
        """
        chunks = []
        
        if len(section_content) <= self.chunk_size:
            # Extract insurance-specific metadata
            metadata = {
                "chunking_strategy": "smart_section",
                "section_title": section_title,
                "original_file_type": document.file_type,
                "policy_metadata": {}
            }
            
            # Extract waiting periods
            waiting_matches = re.finditer(self.metadata_patterns['waiting_period'], section_content, re.IGNORECASE)
            waiting_periods = [(int(m.group(1)), m.group(0)) for m in waiting_matches]
            if waiting_periods:
                metadata['policy_metadata']['waiting_periods'] = waiting_periods
            
            # Extract coverage limits
            coverage_matches = re.finditer(self.metadata_patterns['coverage_limit'], section_content, re.IGNORECASE)
            coverage_limits = [(int(m.group(1).replace(',', '')), m.group(0)) for m in coverage_matches]
            if coverage_limits:
                metadata['policy_metadata']['coverage_limits'] = coverage_limits
            
            # Extract age limits
            age_matches = re.finditer(self.metadata_patterns['age_limit'], section_content, re.IGNORECASE)
            age_limits = [(m.group(1), m.group(2) if m.group(2) else None, m.group(0)) for m in age_matches]
            if age_limits:
                metadata['policy_metadata']['age_limits'] = age_limits
            
            # Extract procedure types
            proc_matches = re.finditer(self.metadata_patterns['procedure_type'], section_content, re.IGNORECASE)
            procedure_types = [m.group(0) for m in proc_matches]
            if procedure_types:
                metadata['policy_metadata']['procedure_types'] = procedure_types
            
            chunk = Chunk(
                content=f"{section_title}. {section_content.strip()}",
                chunk_id=f"{document.filename}_s{section_idx}_0",
                source_document=document.filename,
                chunk_index=section_idx * 1000,  # Leave room for sub-chunks
                start_char=offset,
                end_char=offset + len(section_content),
                metadata=metadata
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
        """
        Chunks the document by paragraphs, combining smaller ones to meet minimum chunk size.

        Args:
            document (Document): The document object to be chunked.

        Returns:
            List[Chunk]: A list of Chunk objects, chunked by paragraphs.
        """
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
    strategies = ["fixed", "sentence", "smart", "hybrid"]
    
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
