from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from .base import DocumentProcessor, DocumentMetadata
from .text_utils import process_text

class TextProcessor(DocumentProcessor):
    def extract_full_text(self, file_path: str) -> tuple[str, DocumentMetadata]:
        """
        Extract full text from text file without chunking.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            tuple: (full_text, metadata)
        """
        # Read text content
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
        
        # Create metadata
        metadata = self._create_metadata(file_path)
        
        return text_content, metadata

    def process(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> tuple[List[str], DocumentMetadata]:
        """
        Process a text document and extract its content and metadata.
        
        Args:
            file_path: Path to the text file
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
            
        Returns:
            tuple: (processed_text_chunks, metadata)
        """
        text_content, metadata = self.extract_full_text(file_path)
        
        # Process text and split into chunks
        processed_chunks = process_text(text_content, chunk_size=chunk_size)
        
        return processed_chunks, metadata
    
    def _create_metadata(self, file_path: str) -> DocumentMetadata:
        """Create metadata for text file."""
        path = Path(file_path)
        
        # Basic metadata from file
        additional_metadata = {
            'file_size': path.stat().st_size,
            'file_extension': path.suffix,
            'last_modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        }
        
        return DocumentMetadata(
            source_path=file_path,
            doc_type='text',
            title=path.stem,  # Use filename as title
            additional_metadata=additional_metadata
        ) 