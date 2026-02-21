#!/usr/bin/env python3
"""
Enhanced document processing pipeline with three stages:
1. Extract full text to data/extracted/
2. Proofread corrected text to data/corrected/ (manual step using text_proofreader.py)
3. Chunk corrected text to data/processed/ and add to vector store
"""

import argparse
from pathlib import Path
import json
import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.document_processor.pdf_processor import PDFProcessor
from src.document_processor.text_processor import TextProcessor
from src.vector_store.chroma_store import ChromaVectorStore
from src.document_processor.text_utils import count_tokens, process_text

def load_config(config_path: str) -> dict:
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def stage1_extract_full_text(config: dict, debug: bool = False) -> None:
    """
    Stage 1: Extract full text from documents to data/extracted/
    """
    print("=== STAGE 1: EXTRACTING FULL TEXT ===")
    
    # Get document directories from config
    documents_dir = Path(config.get("documents_dir", "data/corpus"))
    extracted_dir = Path("data/extracted")
    extracted_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processors with output directory
    pdf_processor = PDFProcessor(output_dir=str(extracted_dir))
    text_processor = TextProcessor(output_dir=str(extracted_dir))
    
    if debug:
        print(f"Processing documents from: {documents_dir}")
        print(f"Saving extracted text to: {extracted_dir}")
    
    # Process all documents
    for file_path in documents_dir.glob("**/*"):
        if file_path.is_file():
            if debug:
                print(f"Extracting: {file_path}")
            
            # Select processor based on file extension
            if file_path.suffix.lower() == '.pdf':
                processor = pdf_processor
            elif file_path.suffix.lower() in ['.txt', '.md']:
                processor = text_processor
            else:
                if debug:
                    print(f"Skipping unsupported file type: {file_path}")
                continue
            
            try:
                # Extract full text without chunking
                full_text, metadata = processor.extract_full_text(str(file_path))
                
                if debug:
                    print(f"Extracted {len(full_text)} characters")
                    print(f"Document ID: {metadata.document_id}")
                
                # Save full text
                processor.save_full_text(full_text, metadata)
                
                if debug:
                    print("Saved to extracted directory")
                
            except Exception as e:
                print(f"Error extracting {file_path}: {str(e)}")
                continue
    
    print("Stage 1 complete: Full text extraction finished\n")

def stage2_proofread_instructions() -> None:
    """
    Stage 2: Instructions for proofreading text
    """
    print("=== STAGE 2: PROOFREADING TEXT ===")
    print("Run the following command to proofread extracted text:")
    print("python scripts/text_proofreader.py data/extracted --output-dir data/corrected")
    print("After proofreading is complete, run this script with --stage 3\n")

def stage3_chunk_and_vectorize(config: dict, debug: bool = False) -> None:
    """
    Stage 3: Chunk corrected text and add to vector store
    """
    print("=== STAGE 3: CHUNKING AND VECTORIZING ===")
    
    corrected_dir = Path("data/corrected")
    processed_dir = Path(config.get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    if not corrected_dir.exists():
        print(f"Error: Corrected directory {corrected_dir} does not exist")
        print("Please run stage 2 (proofreading) first")
        return
    
    # Get chunk size configuration
    chunk_size = config.get("chunk_size", 1000)
    chunk_overlap = config.get("chunk_overlap", 200)
    
    # Initialize vector store
    vector_store = ChromaVectorStore(
        persist_directory=config["vector_store_dir"],
        collection_name=config["collection_name"]
    )
    
    # Initialize text processor for chunking
    text_processor = TextProcessor(output_dir=str(processed_dir))
    
    if debug:
        print(f"Processing corrected files from: {corrected_dir}")
        print(f"Saving chunked files to: {processed_dir}")
        print(f"Vector store directory: {config['vector_store_dir']}")
        print(f"Collection name: {config['collection_name']}")
        print(f"Chunk size: {chunk_size}")
        print(f"Chunk overlap: {chunk_overlap}")
    
    # Process all corrected text files
    corrected_files = list(corrected_dir.glob("corrected_*.txt"))
    print(f"Found {len(corrected_files)} corrected files to process")
    
    for file_path in corrected_files:
        if debug:
            print(f"\nProcessing: {file_path}")
        
        try:
            # Load the corrected text
            with open(file_path, 'r', encoding='utf-8') as f:
                corrected_text = f.read()
            
            # Load original metadata (find corresponding metadata file)
            original_id = file_path.stem.replace("corrected_", "")
            metadata_files = list(corrected_dir.glob(f"{original_id}_metadata.json"))
            
            if not metadata_files:
                # Look in extracted directory
                metadata_files = list(Path("data/extracted").glob(f"{original_id}_metadata.json"))
            
            if metadata_files:
                with open(metadata_files[0], 'r') as f:
                    metadata_dict = json.load(f)
                # Update the document ID to reflect it's now corrected
                metadata_dict['document_id'] = f"corrected_{metadata_dict['document_id']}"
                from src.document_processor.base import DocumentMetadata
                metadata = DocumentMetadata.from_dict(metadata_dict)
            else:
                print(f"Warning: No metadata found for {file_path}, creating basic metadata")
                from src.document_processor.base import DocumentMetadata
                metadata = DocumentMetadata(
                    source_path=str(file_path),
                    doc_type='corrected_text',
                    title=file_path.stem
                )
            
            # Process text and split into chunks
            processed_chunks = process_text(corrected_text, chunk_size=chunk_size)
            
            if debug:
                total_tokens = sum(count_tokens(chunk) for chunk in processed_chunks)
                print(f"Generated {len(processed_chunks)} chunks")
                print(f"Total tokens: {total_tokens:,}")
                print(f"Average tokens per chunk: {total_tokens // len(processed_chunks) if processed_chunks else 0}")
            
            # Save processed document
            text_processor.save_processed_document(processed_chunks, metadata)
            
            # Convert metadata to dictionary and flatten nested structures
            meta_dict = metadata.to_dict()
            flattened_meta = {}
            
            def flatten_dict(d, prefix=''):
                for k, v in d.items():
                    key = f"{prefix}{k}" if prefix else k
                    if isinstance(v, dict):
                        flatten_dict(v, f"{key}_")
                    else:
                        # Convert non-primitive values to strings
                        if not isinstance(v, (str, int, float, bool, type(None))):
                            v = str(v)
                        flattened_meta[key] = v
            
            flatten_dict(meta_dict)
            
            # Add to vector store
            chunk_ids = [f"{flattened_meta['document_id']}_chunk_{i}" for i in range(len(processed_chunks))]
            
            # Use smaller batch size if document is large
            batch_size = 50 if len(processed_chunks) > 100 else 100
            vector_store.add_documents(processed_chunks, [flattened_meta] * len(processed_chunks), chunk_ids, batch_size=batch_size)
            
            if debug:
                print("Added to vector store")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    print("\nStage 3 complete: Chunking and vectorization finished")
    if debug:
        print("\nVector store contents:")
        vector_store.list_collections()

def process_documents(config: dict, debug: bool = False) -> None:
    """
    Legacy function that runs all stages for backward compatibility.
    """
    print("Running all stages of document processing...")
    stage1_extract_full_text(config, debug)
    stage2_proofread_instructions()
    print("NOTE: Complete stage 2 proofreading manually, then run with --stage 3")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Multi-stage document processing pipeline")
    parser.add_argument("--config", default="config/config.json", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], 
                       help="Which stage to run: 1=extract, 2=proofread info, 3=chunk. If not specified, runs stage 1 only.")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.stage is None:
        # Default behavior: run stage 1 only
        stage1_extract_full_text(config, args.debug)
    elif args.stage == 1:
        stage1_extract_full_text(config, args.debug)
    elif args.stage == 2:
        stage2_proofread_instructions()
    elif args.stage == 3:
        stage3_chunk_and_vectorize(config, args.debug)
    else:
        # Legacy mode - run all stages
        process_documents(config, args.debug)

if __name__ == "__main__":
    main() 