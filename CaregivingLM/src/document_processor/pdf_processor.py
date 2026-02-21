from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import PyPDF2

from .base import DocumentProcessor, DocumentMetadata
from .text_utils import process_text

class PDFProcessor(DocumentProcessor):
    def extract_full_text(self, file_path: str) -> tuple[str, DocumentMetadata]:
        """
        Extract full text from PDF without chunking.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            tuple: (full_text, metadata)
        """
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            text_content = []
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())
            
            # Join all pages
            full_text = '\n'.join(text_content)
            
            # Extract metadata
            metadata = self._extract_metadata(pdf_reader.metadata, file_path)
            
            return full_text, metadata

    def process(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> tuple[List[str], DocumentMetadata]:
        """
        Process a PDF document and extract its text content and metadata.
        
        Args:
            file_path: Path to the PDF file
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
            
        Returns:
            tuple: (processed_text_chunks, metadata)
        """
        full_text, metadata = self.extract_full_text(file_path)
        
        # Process text and split into chunks
        processed_chunks = process_text(full_text, chunk_size=chunk_size)
        
        return processed_chunks, metadata
    
    def _extract_metadata(
        self,
        pdf_metadata: Dict[str, Any],
        file_path: str
    ) -> DocumentMetadata:
        """Extract metadata from PDF document."""
        # Convert PDF metadata to our format
        title = pdf_metadata.get('/Title')
        author = pdf_metadata.get('/Author')
        
        # Try to extract date from PDF metadata
        date = None
        if '/CreationDate' in pdf_metadata:
            try:
                date_str = pdf_metadata['/CreationDate']
                # PDF dates are in format: D:YYYYMMDDHHmmSS
                if date_str.startswith('D:'):
                    date_str = date_str[2:]
                    date = datetime.strptime(date_str[:14], '%Y%m%d%H%M%S')
            except (ValueError, TypeError):
                pass
        
        # Additional metadata from PDF
        additional_metadata = {
            'creator': pdf_metadata.get('/Creator'),
            'producer': pdf_metadata.get('/Producer'),
            'subject': pdf_metadata.get('/Subject'),
            'keywords': pdf_metadata.get('/Keywords'),
        }
        
        return DocumentMetadata(
            source_path=file_path,
            doc_type='pdf',
            title=title,
            author=author,
            date=date,
            additional_metadata=additional_metadata
        ) 