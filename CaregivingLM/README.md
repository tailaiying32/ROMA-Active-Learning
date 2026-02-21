# CaregivingLM

A language model system for caregiving and occupational therapy applications, featuring advanced document processing, text proofreading, vector storage, and intelligent querying capabilities.

## Features

- **Three-Stage Document Processing**: Extract → Proofread → Chunk pipeline for high-quality text
- **AI-Powered Text Proofreading**: Automatic OCR error correction and typo fixing
- **Article Filtering**: LM-based relevance assessment for caregiving knowledge bases
- **Vector Storage**: Chroma-based vector store with OpenAI embeddings  
- **RAG Pipeline**: Retrieval-Augmented Generation for context-aware responses
- **Multiple Query Modes**: Support for general queries, functional assessments, and robot adaptation
- **Streaming Processing**: Real-time text correction with immediate file output

## Document Processing Pipeline

The system uses a three-stage pipeline to ensure high-quality text processing:

### Stage 1: Text Extraction
Extract full text from documents without chunking to preserve complete structure.

```bash
# Extract all documents to data/extracted/
python scripts/process_documents.py --stage 1 --debug

# Or use default behavior (stage 1 only)
python scripts/process_documents.py --debug
```

### Stage 2: Text Proofreading
AI-powered correction of OCR errors, typos, and formatting issues while preserving original structure.

```bash
# Proofread all extracted files
python scripts/text_proofreader.py data/extracted --output-dir data/corrected

# Process single file with custom chunk size
python scripts/text_proofreader.py data/extracted/document.txt --chunk-size 15000

# Debug mode to see processing details
python scripts/text_proofreader.py data/extracted/document.txt --debug
```

**Proofreading Features:**
- **Streaming output**: Text is written to corrected files as it's processed
- **Real-time monitoring**: Watch progress with ✓/✗ indicators per chunk
- **Error correction**: Fixes OCR errors, typos, punctuation, and garbled characters
- **Structure preservation**: Maintains original formatting, line breaks, and layout
- **Memory efficient**: No need to store entire documents in memory
- **Early examination**: View results while processing continues

### Stage 3: Chunking and Vectorization
Process corrected text into chunks and add to vector store.

```bash
# Chunk corrected files and add to vector store
python scripts/process_documents.py --stage 3 --debug
```

### Article Filtering (Optional)
Filter research articles for relevance to caregiving and rehabilitation.

```bash
# Filter articles using LM assessment
python scripts/chunked_article_filter.py articles.jsonl --chunk-size 10 --output-relevant relevant.jsonl --output-irrelevant irrelevant.jsonl
```

## Query Modes

The system supports three distinct query modes:

### 1. Default Mode
General question-answering mode for caregiving and occupational therapy queries.

```bash
python scripts/query.py "What are the key principles of stroke rehabilitation?" --mode default
```

### 2. Functionality Mode
Specialized mode for generating structured functional assessments based on medical conditions.

```bash
python scripts/query.py "Patient with left hemiplegia following ischemic stroke affecting the right middle cerebral artery, 3 months post-stroke" --mode functionality
```

The functionality mode generates structured JSON output including:
- Joint range of motion assessments
- Muscle strength evaluations (0-4 scale)
- Motor control and coordination analysis
- Tone and spasticity assessment
- Compensatory movement patterns
- Expected functional abilities and limitations
- Clinical confidence notes and source references

### 3. Robot Mode
Advanced mode for generating robot adaptation guidelines based on functional assessments and specific ADL tasks.

```bash
python scripts/query.py "Generate robot adaptation guidelines for dressing task" --mode robot --functionality_cache data/cache/interaction_20231201_143022.json --task_name dressing
```

The robot mode requires:
- `--functionality_cache`: Path to a previous functionality assessment cache file
- `--task_name`: Name of the ADL task (dressing, feeding, hygiene, toileting, transferring)

The robot mode generates structured JSON output including:
- Movement speed and trajectory specifications
- Maximum force limits for different body regions
- Joint angle constraints based on user limitations
- Grasp assistance strategies
- Sequence of actions for task completion
- Postural support requirements
- Communication prompts for user guidance
- Safety notes and personalization recommendations

## Output Files

All modes automatically save responses to text files in the `outputs/` directory:
- `default_response_YYYYMMDD_HHMMSS.txt` - General query responses
- `functionality_response_YYYYMMDD_HHMMSS.txt` - Structured functional assessments
- `robot_response_YYYYMMDD_HHMMSS.txt` - Robot adaptation guidelines

## Usage Examples

### Basic Query
```bash
python scripts/query.py "How does occupational therapy help with ADL training?"
```

### Functionality Assessment
```bash
python scripts/query.py "Patient with C6 spinal cord injury, complete tetraplegia" --mode functionality
```

### Robot Adaptation (Two-Step Process)
```bash
# Step 1: Generate functionality assessment
python scripts/query.py "Patient with left hemiplegia following ischemic stroke affecting the right middle cerebral artery, 3 months post-stroke" --mode functionality

# Step 2: Generate robot adaptation guidelines (use cache file from step 1)
python scripts/query.py "Generate robot adaptation guidelines for dressing task" --mode robot --functionality_cache data/cache/interaction_20231201_143022.json --task_name dressing
```

### Available Tasks for Robot Mode
- `dressing` - Upper and lower body dressing tasks
- `feeding` - Self-feeding and utensil manipulation
- `hygiene` - Personal hygiene and grooming
- `toileting` - Toileting and bathroom transfers
- `transferring` - Bed, chair, and surface transfers

### Debug Mode
```bash
python scripts/query.py "What are the best practices for wheelchair transfers?" --debug
```

### Test All Modes
```bash
# Test default and functionality modes
python scripts/test_modes.py

# Test robot mode
python scripts/test_robot_mode.py
```

## Configuration

Edit `config/config.json` to customize:
- Chunk sizes for document processing
- Vector store settings
- OpenAI model selection
- Cache and output directories

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

3. Process documents (three-stage pipeline):
```bash
# Stage 1: Extract full text
python scripts/process_documents.py --stage 1 --debug

# Stage 2: Proofread extracted text
python scripts/text_proofreader.py data/extracted --output-dir data/corrected

# Stage 3: Chunk and vectorize corrected text
python scripts/process_documents.py --stage 3 --debug
```

4. Query the system:
```bash
python scripts/query.py "Your question here"
```

## Quick Start (Legacy Single-Stage)

For backward compatibility, you can still process documents in a single step without proofreading:

```bash
# Extract and chunk directly (skips proofreading)
python scripts/process_documents.py --debug
```

## Directory Structure

The processing pipeline creates the following directory structure:

```
data/
├── corpus/           # Original documents (PDFs, text files)
├── extracted/        # Stage 1: Full extracted text files
├── corrected/        # Stage 2: Proofreaded text files  
├── processed/        # Stage 3: Final chunked text files
├── vector_store/     # Chroma vector database
├── cache/           # Query cache files
└── outputs/         # Query response files
```

**Key Files:**
- `data/extracted/*.txt` - Raw extracted text (may contain OCR errors)
- `data/corrected/corrected_*.txt` - AI-corrected text (ready for chunking)
- `data/processed/*.txt` - Final chunked text (used for vector store)

## Configuration Files

- `config/config.json` - Main configuration (chunk sizes, paths, model settings)
- `src/prompts/text_correction.txt` - Proofreading prompt template
- `src/prompts/article_filter.txt` - Article filtering prompt template
- `.env` - Environment variables (OpenAI API key)

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions, including token limit errors and document processing problems.