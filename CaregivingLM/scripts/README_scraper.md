# Article Scraper

A two-step article scraping and PDF download tool using the `paperscraper` package. This tool searches multiple academic databases (PubMed, arXiv, bioRxiv, medRxiv) for articles based on keywords and downloads their PDFs.

## Features

- **Two-step process**: Separate search and download phases for better control
- **Multiple databases**: Searches PubMed, arXiv, bioRxiv, and medRxiv
- **Configurable limits**: Set maximum results per keyword and maximum PDFs to download
- **Comprehensive logging**: Detailed logs for debugging and monitoring
- **Flexible configuration**: JSON-based configuration file
- **Easy-to-use wrapper**: Simplified interface for common operations
- **Dump management**: Automatic setup and checking of bioRxiv/medRxiv dumps
- **Smart shuffling**: Randomly shuffles articles across keywords for even distribution

## Installation

1. Install the required packages:
```bash
pip install paperscraper
```

2. Make sure you have the scraper scripts in your `scripts/` directory:
   - `article_scraper.py` - Main scraper class
   - `run_scraper.py` - Simplified wrapper script

## Quick Start

### 1. Prepare Keywords

Edit `config/keywords.txt` with your search terms (one per line):
```
caregiving
caregiver burden
family caregiver
dementia care
```

### 2. Configure Settings

Edit `config/scraper_config.json` to adjust settings:
```json
{
  "output_directory": "scraped_articles",
  "max_pdfs_to_download": 25,
  "databases": ["pubmed", "arxiv", "biorxiv", "medrxiv"],
  "log_level": "INFO",
  "shuffle_seed": null
}
```

**New: Shuffling Configuration**
- `shuffle_seed`: Set to a number for reproducible shuffling, or `null` for random shuffling
- When set to a number, the same seed will always produce the same article order
- When set to `null`, articles are randomly shuffled each time

### 3. Setup Dumps (Required for bioRxiv/medRxiv)

**Important**: bioRxiv and medRxiv require downloading large dump files first:

```bash
cd scripts
python run_scraper.py --setup-dumps
```

This will download:
- **medRxiv**: ~35 MB (takes ~30 minutes)
- **bioRxiv**: ~350 MB (takes ~1 hour)

**Note**: You only need to run this once. The dumps are stored locally and reused.

### 4. Run the Scraper

#### Option A: Run both steps together
```bash
cd scripts
python run_scraper.py
```

#### Option B: Run steps separately (recommended for large searches)

**Step 1: Search only**
```bash
cd scripts
python run_scraper.py --search-only
```

**Step 2: Download PDFs (after reviewing results)**
```bash
cd scripts
python run_scraper.py --download-only --metadata-file scraped_articles/metadata/search_results_YYYYMMDD_HHMMSS.json
```

## Smart Shuffling Feature

The scraper now includes intelligent shuffling that ensures articles are evenly distributed across all keywords. This prevents the issue where all articles from the first keyword appear at the beginning of the list.

### How it works:

1. **Keyword Tracking**: Each article is tagged with the keyword it was found with
2. **Even Distribution**: Articles are distributed evenly across keywords
3. **Random Shuffling**: Articles within each keyword are shuffled, then the final list is shuffled
4. **Reproducible**: Optional seed parameter for consistent results

### Example:

If you have 3 keywords and want to download 300 PDFs:
- **Before**: All 300 might come from the first keyword
- **After**: ~100 articles from each keyword, randomly mixed

### Configuration:

```json
{
  "shuffle_seed": 42  // Fixed seed for reproducible results
}
```

or

```json
{
  "shuffle_seed": null  // Random shuffling each time
}
```

### Command Line Usage:

```bash
# Use fixed seed for reproducible results
python article_scraper.py --keywords config/keywords.txt --max-pdfs 300 --shuffle-seed 42

# Use random shuffling
python article_scraper.py --keywords config/keywords.txt --max-pdfs 300
```

## Usage Examples

### Basic Usage
```bash
# Check if dumps are available
python run_scraper.py --check-dumps

# Setup dumps (first time only)
python run_scraper.py --setup-dumps

# Run with default settings
python run_scraper.py

# Use custom keywords file
python run_scraper.py --keywords my_keywords.txt

# Use custom config file
python run_scraper.py --config my_config.json
```

### Advanced Usage
```bash
# Search only (Step 1)
python run_scraper.py --search-only

# Download only (Step 2) - requires metadata file from Step 1
python run_scraper.py --download-only --metadata-file path/to/metadata.json

# Use the main scraper directly with more options
python article_scraper.py --keywords config/keywords.txt --max-pdfs 50

# Use with shuffling seed for reproducible results
python article_scraper.py --keywords config/keywords.txt --max-pdfs 300 --shuffle-seed 42
```

## Testing the Shuffling

You can test the shuffling functionality with the included test script:

```bash
cd scripts
python test_shuffling.py
```

This will create mock data and demonstrate how articles are distributed across keywords with different settings.

## Output Structure

The scraper creates the following directory structure:
```
scraped_articles/
├── metadata/
│   ├── search_results_YYYYMMDD_HHMMSS.json
│   ├── pubmed_caregiving_YYYYMMDD_HHMMSS.jsonl
│   ├── arxiv_caregiving_YYYYMMDD_HHMMSS.jsonl
│   ├── biorxiv_caregiving_YYYYMMDD_HHMMSS.jsonl
│   ├── medrxiv_caregiving_YYYYMMDD_HHMMSS.jsonl
│   └── combined_for_pdf_download.jsonl
├── pdfs/
│   ├── 10.1000_abc123.pdf
│   ├── 1234.5678.pdf
│   └── ...
├── pubmed/
├── arxiv/
├── biorxiv/
└── medrxiv/
```

## Configuration Options

### scraper_config.json
- `output_directory`: Where to save results (default: "scraped_articles")
- `max_pdfs_to_download`: Maximum PDFs to download total (default: 25)
- `databases`: List of databases to search (default: all four)
- `log_level`: Logging level (default: "INFO")
- `shuffle_seed`: Random seed for reproducible shuffling (default: null for random)

### Command Line Options

#### run_scraper.py
- `--keywords, -k`: Path to keywords file
- `--config, -c`: Path to config file
- `--search-only, -s`: Run only search step
- `--download-only, -d`: Run only download step
- `--metadata-file, -m`: Metadata file for download step
- `--setup-dumps`: Setup bioRxiv and medRxiv dumps
- `--check-dumps`: Check if dumps exist

#### article_scraper.py
- `--keywords, -k`: Path to keywords file (required)
- `--output-dir, -o`: Output directory
- `--max-pdfs, -p`: Maximum PDFs to download
- `--step`: Which step to run (search/download/both)
- `--metadata-file, -m`: Metadata file for download step
- `--setup-dumps`: Setup bioRxiv and medRxiv dumps

## Database-Specific Requirements

### PubMed & arXiv
- **No setup required** - works immediately
- Uses direct API access

### bioRxiv & medRxiv
- **Requires dump setup** - run `--setup-dumps` first
- Downloads large files (~350MB total)
- Takes significant time (1-1.5 hours total)
- Dumps are stored locally and reused

## Two-Step Process Benefits

1. **Control**: Review search results before downloading PDFs
2. **Resource Management**: Avoid downloading too many files at once
3. **Debugging**: Easier to troubleshoot issues
4. **Flexibility**: Can modify search parameters and re-run without re-downloading

## Tips for Large Searches

1. **Start small**: Use a few keywords and low limits first
2. **Review results**: Check the metadata files before downloading
3. **Adjust limits**: Increase `max_pdfs_to_download` gradually
4. **Monitor disk space**: PDFs can take significant storage
5. **Use separate steps**: Run search first, then download with appropriate limits
6. **Setup dumps once**: bioRxiv/medRxiv dumps only need to be downloaded once

## Troubleshooting

### Common Issues

1. **No results found**: Check your keywords and internet connection
2. **PDF download fails**: Some articles may not have open-access PDFs
3. **Rate limiting**: The scraper includes delays, but you may need to adjust
4. **Disk space**: Monitor available storage when downloading many PDFs
5. **bioRxiv/medRxiv not working**: Run `--setup-dumps` first

### Dump Issues

```bash
# Check if dumps exist
python run_scraper.py --check-dumps

# Setup dumps if missing
python run_scraper.py --setup-dumps
```

### Logs

Check `article_scraper.log` for detailed information about the scraping process.

## Database-Specific Notes

- **PubMed**: Uses direct API, requires DOI for PDF download
- **arXiv**: Uses direct API, requires arXiv ID for PDF download
- **bioRxiv/medRxiv**: Requires local dumps, uses DOI for PDF download
- **Open Access**: Only open-access PDFs can be downloaded

## Performance Notes

- **PubMed**: Fast, no setup required
- **arXiv**: Fast, no setup required
- **bioRxiv**: Requires dump (~350MB, ~1 hour download)
- **medRxiv**: Requires dump (~35MB, ~30 minutes download)

## License

This tool uses the `paperscraper` package. Please respect the terms of service of the databases being accessed. 