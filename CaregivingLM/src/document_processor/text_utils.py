import re
import unicodedata
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from ..config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
import logging

logger = logging.getLogger(__name__)

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace and line breaks in text.
    - Removes single newlines not between paragraphs
    - Preserves paragraph breaks
    """
    # Replace single newlines with spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Normalize multiple spaces
    text = re.sub(r' +', ' ', text)
    # Normalize multiple newlines to double newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def fix_hyphenated_words(text: str) -> str:
    """Fix hyphenated words across line breaks."""
    # Join hyphenated words
    text = re.sub(r'-\n([a-z])', r'\1', text)
    # Also handle cases where the hyphen might be followed by a space
    text = re.sub(r'-\s*\n([a-z])', r'\1', text)
    return text

def remove_page_numbers(text: str) -> str:
    """Remove page numbers and common page artifacts."""
    # Remove "Page X of Y" patterns
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    # Remove standalone page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Remove form feed characters
    text = text.replace('\f', '')
    return text

def clean_unicode(text: str) -> str:
    """Normalize unicode characters and clean encoding."""
    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char == '\n')
    return text

def remove_headers_footers(text: str, min_frequency: int = 2) -> str:
    """
    Remove repeating headers and footers.
    Uses frequency analysis to identify and remove lines that appear too frequently.
    """
    lines = text.split('\n')
    line_freq: Dict[str, int] = {}
    
    # Count line frequencies
    for line in lines:
        line = line.strip()
        if line:
            line_freq[line] = line_freq.get(line, 0) + 1
    
    # Filter out frequent lines
    filtered_lines = [
        line for line in lines
        if not line.strip() or line_freq.get(line.strip(), 0) < min_frequency
    ]
    
    return '\n'.join(filtered_lines)

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def split_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Args:
        text: Text to split
        chunk_size: Target size of each chunk in tokens
        chunk_overlap: Overlap between chunks in tokens
        
    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name="cl100k_base",
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    return splitter.split_text(text)

def validate_chunk_size(chunk: str, max_tokens: int = 8000) -> bool:
    """
    Validate that a chunk doesn't exceed the maximum token limit.
    
    Args:
        chunk: Text chunk to validate
        max_tokens: Maximum allowed tokens (default 8000 for safety)
        
    Returns:
        True if chunk is within limits, False otherwise
    """
    token_count = count_tokens(chunk)
    return token_count <= max_tokens

def split_large_chunks(
    chunks: List[str],
    max_tokens: int = 8000,
    chunk_size: int = 1000
) -> List[str]:
    """
    Split any chunks that exceed the maximum token limit.
    
    Args:
        chunks: List of text chunks
        max_tokens: Maximum allowed tokens per chunk
        chunk_size: Target size for new chunks
        
    Returns:
        List of chunks, with large chunks split into smaller ones
    """
    result_chunks = []
    
    for chunk in chunks:
        if validate_chunk_size(chunk, max_tokens):
            result_chunks.append(chunk)
        else:
            # Split the large chunk
            logger.warning(f"Chunk exceeds {max_tokens} tokens ({count_tokens(chunk)}), splitting...")
            sub_chunks = split_into_chunks(chunk, chunk_size=chunk_size)
            result_chunks.extend(sub_chunks)
    
    return result_chunks

def process_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """
    Apply all text processing steps and return chunks.
    
    Args:
        text: Raw text to process
        chunk_size: Target size of each chunk in tokens
        
    Returns:
        List of processed text chunks
    """
    # Apply all cleaning steps
    text = normalize_whitespace(text)
    text = fix_hyphenated_words(text)
    text = remove_page_numbers(text)
    text = clean_unicode(text)
    text = remove_headers_footers(text)
    
    # Split into chunks
    chunks = split_into_chunks(text, chunk_size=chunk_size)
    
    # Validate and split any chunks that are too large
    chunks = split_large_chunks(chunks, max_tokens=8000, chunk_size=chunk_size)
    
    return chunks 