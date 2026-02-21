from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib
import json

class DocumentMetadata:
    def __init__(
        self,
        source_path: str,
        doc_type: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        date: Optional[datetime] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ):
        self.source_path = source_path
        self.doc_type = doc_type
        self.title = title
        self.author = author
        self.date = date
        self.additional_metadata = additional_metadata or {}
        self.processed_date = datetime.now()
        self.document_id = self._generate_document_id()

    def _generate_document_id(self) -> str:
        """Generate a unique document ID based on metadata."""
        content = f"{self.source_path}{self.processed_date.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "doc_type": self.doc_type,
            "title": self.title,
            "author": self.author,
            "date": self.date.isoformat() if self.date else None,
            "processed_date": self.processed_date.isoformat(),
            "additional_metadata": self.additional_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        """Create metadata instance from dictionary."""
        return cls(
            source_path=data["source_path"],
            doc_type=data["doc_type"],
            title=data["title"],
            author=data["author"],
            date=datetime.fromisoformat(data["date"]) if data["date"] else None,
            additional_metadata=data["additional_metadata"]
        )

class DocumentProcessor(ABC):
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def process(self, file_path: str) -> tuple[List[str], DocumentMetadata]:
        """
        Process a document and return its text chunks and metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            tuple: (processed_text_chunks, metadata)
        """
        pass

    def save_full_text(self, full_text: str, metadata: DocumentMetadata) -> str:
        """
        Save full extracted text before chunking.
        
        Args:
            full_text: Complete extracted text
            metadata: Document metadata
            
        Returns:
            str: Path to the saved full text file
        """
        # Save full text
        output_path = self.output_dir / f"{metadata.document_id}.txt"
        output_path.write_text(full_text)
        
        # Save metadata
        metadata_path = self.output_dir / f"{metadata.document_id}_metadata.json"
        metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2))
        
        return str(output_path)

    def save_processed_document(self, chunks: List[str], metadata: DocumentMetadata) -> str:
        """
        Save processed document chunks and its metadata.
        
        Args:
            chunks: List of processed text chunks
            metadata: Document metadata
            
        Returns:
            str: Path to the saved processed document
        """
        # Save text chunks
        output_path = self.output_dir / f"{metadata.document_id}.txt"
        output_path.write_text('\n\n'.join(chunks))
        
        # Save metadata
        metadata_path = self.output_dir / f"{metadata.document_id}_metadata.json"
        metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2))
        
        return str(output_path) 