#!/usr/bin/env python3
"""
Text Proofreader Script

Processes extracted text files chunk by chunk to correct OCR errors, typos, and formatting issues
while preserving the original structure and content. Based on qwen_inference.py.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
from pathlib import Path
from typing import List, Optional
import time


def load_text_file(file_path: str) -> str:
    """Load text content from file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def save_text_file(content: str, file_path: str) -> None:
    """Save corrected text to file"""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def chunk_text(text: str, chunk_size: int = 2000) -> List[str]:
    """
    Split text into simple non-overlapping chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Number of characters per chunk
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this isn't the last chunk, try to break at a natural boundary
        if end < len(text):
            # Look for paragraph breaks first
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break > start:
                end = paragraph_break + 2  # Include the paragraph break
            else:
                # Look for sentence endings
                sentence_break = text.rfind('. ', start, end)
                if sentence_break > start:
                    end = sentence_break + 2  # Include the period and space
                else:
                    # Look for any line break
                    line_break = text.rfind('\n', start, end)
                    if line_break > start:
                        end = line_break + 1  # Include the line break
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move to next chunk with no overlap
        start = end
    
    return chunks


def load_prompt_context(file_path: str) -> str:
    """Load prompt context from a text file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


def process_text_chunk(model, tokenizer, prompt_context: str, text_chunk: str, 
                      debug: bool = False) -> str:
    """Process a single chunk of text through the model for correction"""
    
    prompt = prompt_context + "\n\n" + text_chunk
    
    if debug:
        print(f"\n{'='*50}")
        print("DEBUG: Processing chunk:")
        print(f"{'='*50}")
        print(f"Chunk length: {len(text_chunk)} characters")
        print(f"First 200 chars: {text_chunk[:200]}...")
        print(f"{'='*50}")
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=8192,  # Smaller than article filter since we want preservation
        do_sample=False,      # Deterministic generation for consistent corrections
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    # Parse thinking content (same as demo script)
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)  # </think>
    except ValueError:
        index = 0
    
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    corrected_text = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    if debug:
        print(f"Model thinking: {thinking_content[:200]}...")
        print(f"Corrected text length: {len(corrected_text)} characters")
        print(f"First 200 chars of corrected: {corrected_text[:200]}...")
    
    return corrected_text


def write_chunk_to_file(output_file, corrected_chunk: str) -> None:
    """
    Write a corrected chunk to the output file.
    
    Args:
        output_file: Open file handle for writing
        corrected_chunk: The corrected text chunk
    """
    output_file.write(corrected_chunk)
    output_file.flush()  # Ensure it's written immediately


def process_file(input_path: str, output_path: str, model, tokenizer, 
                prompt_context: str, chunk_size: int = 10000,
                debug: bool = False) -> None:
    """Process a single text file with streaming output"""
    
    print(f"Processing: {input_path}")
    print(f"Streaming output to: {output_path}")
    
    # Load the text
    text = load_text_file(input_path)
    print(f"Loaded text: {len(text)} characters")
    
    # Split into chunks
    chunks = chunk_text(text, chunk_size)
    print(f"Split into {len(chunks)} chunks")
    
    # Open output file for streaming
    with open(output_path, 'w', encoding='utf-8') as output_file:
        # Process each chunk and write immediately
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}... ", end="", flush=True)
            
            try:
                corrected_chunk = process_text_chunk(
                    model, tokenizer, prompt_context, chunk, debug
                )
                
                # Write chunk to file immediately
                write_chunk_to_file(output_file, corrected_chunk)
                
                print(f"✓ Written to file")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                print(f"Using original chunk...")
                
                # Write original chunk on error
                write_chunk_to_file(output_file, chunk)
    
    # Calculate final file size
    final_size = Path(output_path).stat().st_size
    print(f"Processing complete!")
    print(f"Original: {len(text)} chars, Final file: {final_size} bytes")
    print(f"You can examine the results at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Proofread and correct extracted text files")
    parser.add_argument("input_path", help="Input file or directory path")
    parser.add_argument("--output-dir", default="data/corrected", 
                       help="Output directory for corrected files")
    parser.add_argument("--chunk-size", type=int, default=8000,
                       help="Size of text chunks in characters for internal processing")
    parser.add_argument("--prompt-file", default="src/prompts/text_correction.txt",
                       help="Prompt context file")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B",
                       help="Model name to use")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    parser.add_argument("--file-pattern", default="*.txt",
                       help="File pattern to match (when input is directory)")
    
    args = parser.parse_args()
    
    # Validate paths
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        return 1
    
    prompt_file = Path(args.prompt_file)
    if not prompt_file.exists():
        print(f"Error: Prompt file {prompt_file} does not exist")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto"
    )
    
    print(f"Loading prompt context from: {prompt_file}")
    prompt_context = load_prompt_context(str(prompt_file))
    
    # Determine files to process
    files_to_process = []
    if input_path.is_file():
        files_to_process.append(input_path)
    else:
        # Process all matching files in directory
        files_to_process = list(input_path.glob(args.file_pattern))
        if not files_to_process:
            print(f"No files matching pattern '{args.file_pattern}' found in {input_path}")
            return 1
    
    print(f"Found {len(files_to_process)} files to process")
    
    # Process each file
    for file_path in files_to_process:
        try:
            # Generate output filename
            output_filename = f"corrected_{file_path.name}"
            output_path = output_dir / output_filename
            
            start_time = time.time()
            process_file(
                str(file_path), str(output_path), model, tokenizer,
                prompt_context, args.chunk_size, args.debug
            )
            elapsed = time.time() - start_time
            print(f"Completed {file_path.name} in {elapsed:.1f} seconds\n")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}\n")
    
    print("Text proofreading complete!")
    return 0


if __name__ == "__main__":
    exit(main())