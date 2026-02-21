#!/usr/bin/env python3
"""
Simplified wrapper for the article scraper.
Loads configuration and provides easy-to-use functions.
"""

import json
import sys
from pathlib import Path
from article_scraper import ArticleScraper
import logging

def load_config(config_file: str = "config/scraper_config.json") -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found. Using defaults.")
        return {
            "output_directory": "scraped_articles",
            "max_pdfs_to_download": 25,
            "databases": ["pubmed", "arxiv", "biorxiv", "medrxiv"],
            "log_level": "INFO",
            "shuffle_seed": None,
            "skip_existing": False
        }

def setup_dumps():
    """Setup the required dumps for bioRxiv and medRxiv."""
    print("=== SETTING UP DUMPS ===")
    print("This will download large files:")
    print("- medRxiv: ~35 MB (takes ~30 minutes)")
    print("- bioRxiv: ~350 MB (takes ~1 hour)")
    print()
    
    response = input("Do you want to continue? (y/N): ")
    if response.lower() != 'y':
        print("Dump setup cancelled.")
        return
    
    scraper = ArticleScraper()
    scraper.setup_dumps()

def run_search_only(keywords_file: str = "config/keywords.txt", config_file: str = "config/scraper_config.json", skip_existing: bool = None):
    """Run only the search step (Step 1)."""
    config = load_config(config_file)
    
    # Use parameter if provided, otherwise use config value
    if skip_existing is None:
        skip_existing = config.get('skip_existing', False)
    
    print("=== RUNNING SEARCH ONLY (STEP 1) ===")
    print(f"Keywords file: {keywords_file}")
    print(f"Output directory: {config['output_directory']}")
    print(f"Skip existing keywords: {skip_existing}")
    print()
    
    pdf_path = config.get('pdf_download_path')
    scraper = ArticleScraper(config['output_directory'], pdf_path)
    
    # Configure threading
    max_workers = config.get('max_download_workers', 8)
    scraper.max_download_workers = max_workers
    keywords = scraper.load_keywords(keywords_file)
    
    if not keywords:
        print("No keywords found. Please check your keywords file.")
        return None
    
    results = scraper.search_articles(keywords, config.get('databases', ['pubmed', 'arxiv', 'biorxiv', 'medrxiv']), skip_existing)
    metadata_file = scraper.save_metadata(results, keywords)
    
    print(f"\nSearch complete! Metadata saved to: {metadata_file}")
    print("You can now run the download step separately using:")
    print(f"python scripts/run_scraper.py --download-only --metadata-file {metadata_file}")
    
    return metadata_file


def run_download_only(metadata_file: str, config_file: str = "config/scraper_config.json", skip_attempted: bool = True):
    """Run only the download step (Step 2)."""
    config = load_config(config_file)
    
    print("=== RUNNING DOWNLOAD ONLY (STEP 2) ===")
    print(f"Metadata file: {metadata_file}")
    print(f"Max PDFs to download: {config['max_pdfs_to_download']}")
    print(f"Shuffle seed: {config.get('shuffle_seed', 'None (random)')}")
    print(f"Skip already attempted: {skip_attempted}")
    if config.get('pdf_download_path'):
        print(f"PDF download path: {config['pdf_download_path']}")
    print()
    
    pdf_path = config.get('pdf_download_path')
    scraper = ArticleScraper(config['output_directory'], pdf_path)
    
    # Configure threading
    max_workers = config.get('max_download_workers', 8)
    scraper.max_download_workers = max_workers
    shuffle_seed = config.get('shuffle_seed')
    scraper.download_pdfs(metadata_file, config['max_pdfs_to_download'], shuffle_seed, skip_attempted)
    
    print("\nDownload complete!")

def run_download_from_search_results(search_results_file: str, config_file: str = "config/scraper_config.json", skip_attempted: bool = True):
    """Download PDFs directly from search results file without filtering."""
    config = load_config(config_file)
    
    print("=== DOWNLOADING DIRECTLY FROM SEARCH RESULTS ===")
    print(f"Search results file: {search_results_file}")
    print(f"Max PDFs to download: {config['max_pdfs_to_download']}")
    print(f"Shuffle seed: {config.get('shuffle_seed', 'None (random)')}")
    print(f"Skip already attempted: {skip_attempted}")
    if config.get('pdf_download_path'):
        print(f"PDF download path: {config['pdf_download_path']}")
    print("⚠️  Note: This will download without filtering - all articles with DOI/arXiv ID will be attempted")
    print()
    
    # Confirm with user
    response = input("Continue with unfiltered download? (y/N): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    pdf_path = config.get('pdf_download_path')
    scraper = ArticleScraper(config['output_directory'], pdf_path)
    
    # Configure threading
    max_workers = config.get('max_download_workers', 8)
    scraper.max_download_workers = max_workers
    shuffle_seed = config.get('shuffle_seed')
    
    # Use the search results file directly as metadata file
    scraper.download_pdfs(search_results_file, config['max_pdfs_to_download'], shuffle_seed, skip_attempted)
    
    print("\nDirect download complete!")

def run_both_steps(keywords_file: str = "config/keywords.txt", config_file: str = "config/scraper_config.json", skip_existing: bool = None):
    """Run both search and download steps."""
    config = load_config(config_file)
    
    # Use parameter if provided, otherwise use config value
    if skip_existing is None:
        skip_existing = config.get('skip_existing', False)
    
    print("=== RUNNING BOTH STEPS ===")
    print(f"Keywords file: {keywords_file}")
    print(f"Output directory: {config['output_directory']}")
    print(f"Databases: {config.get('databases', ['pubmed', 'arxiv', 'biorxiv', 'medrxiv'])}")
    print(f"Max PDFs to download: {config['max_pdfs_to_download']}")
    print(f"Shuffle seed: {config.get('shuffle_seed', 'None (random)')}")
    print(f"Skip existing keywords: {skip_existing}")
    print()
    
    pdf_path = config.get('pdf_download_path')
    scraper = ArticleScraper(config['output_directory'], pdf_path)
    
    # Configure threading
    max_workers = config.get('max_download_workers', 8)
    scraper.max_download_workers = max_workers
    keywords = scraper.load_keywords(keywords_file)
    
    if not keywords:
        print("No keywords found. Please check your keywords file.")
        return
    
    # Step 1: Search
    print("Step 1: Searching articles...")
    results = scraper.search_articles(keywords, config.get('databases', ['pubmed', 'arxiv', 'biorxiv', 'medrxiv']), skip_existing)
    metadata_file = scraper.save_metadata(results, keywords)
    
    # Step 2: Download
    print("\nStep 2: Downloading PDFs...")
    shuffle_seed = config.get('shuffle_seed')
    scraper.download_pdfs(str(metadata_file), config['max_pdfs_to_download'], shuffle_seed, True)
    
    print("\nBoth steps complete!")

def check_dumps():
    """Check if bioRxiv and medRxiv dumps exist."""
    dump_path = Path("server_dumps")
    if not dump_path.exists():
        print("❌ server_dumps directory not found")
        return False
    
    biorxiv_dumps = list(dump_path.glob("biorxiv_*.jsonl"))
    medrxiv_dumps = list(dump_path.glob("medrxiv_*.jsonl"))
    
    print("=== DUMP STATUS ===")
    if biorxiv_dumps:
        latest_biorxiv = max(biorxiv_dumps, key=lambda x: x.stat().st_mtime)
        print(f"✓ bioRxiv dump found: {latest_biorxiv.name}")
    else:
        print("❌ bioRxiv dump not found")
    
    if medrxiv_dumps:
        latest_medrxiv = max(medrxiv_dumps, key=lambda x: x.stat().st_mtime)
        print(f"✓ medRxiv dump found: {latest_medrxiv.name}")
    else:
        print("❌ medRxiv dump not found")
    
    return len(biorxiv_dumps) > 0 and len(medrxiv_dumps) > 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Easy-to-use article scraper wrapper')
    parser.add_argument('--keywords', '-k', default='config/keywords.txt', 
                       help='Path to keywords file (default: config/keywords.txt)')
    parser.add_argument('--config', '-c', default='config/scraper_config.json',
                       help='Path to config file (default: config/scraper_config.json)')
    parser.add_argument('--search-only', '-s', action='store_true',
                       help='Run only the search step')
    parser.add_argument('--download-only', '-d', action='store_true',
                       help='Run only the download step')
    parser.add_argument('--download-unfiltered', action='store_true',
                       help='Download PDFs directly from search results without filtering')
    parser.add_argument('--metadata-file', '-m', 
                       help='Metadata file for download step (required with --download-only)')
    parser.add_argument('--search-results-file', '-r',
                       help='Search results file for unfiltered download (required with --download-unfiltered)')
    parser.add_argument('--setup-dumps', action='store_true',
                       help='Setup bioRxiv and medRxiv dumps')
    parser.add_argument('--check-dumps', action='store_true',
                       help='Check if bioRxiv and medRxiv dumps exist')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip keywords that have already been scraped')
    parser.add_argument('--retry-failed', action='store_true',
                       help='Retry previously failed downloads (default: skip all attempted articles)')
    parser.add_argument('--disable-resume', action='store_true',
                       help='Disable resume functionality completely (for debugging)')
    
    args = parser.parse_args()
    
    if args.setup_dumps:
        setup_dumps()
    elif args.check_dumps:
        check_dumps()
    elif args.download_only:
        if not args.metadata_file:
            print("Error: --metadata-file is required when using --download-only")
            sys.exit(1)
        skip_attempted = not args.retry_failed and not args.disable_resume
        run_download_only(args.metadata_file, args.config, skip_attempted)
    elif args.download_unfiltered:
        if not args.search_results_file:
            print("Error: --search-results-file is required when using --download-unfiltered")
            sys.exit(1)
        skip_attempted = not args.retry_failed and not args.disable_resume
        run_download_from_search_results(args.search_results_file, args.config, skip_attempted)
    elif args.search_only:
        run_search_only(args.keywords, args.config, args.skip_existing)
    else:
        run_both_steps(args.keywords, args.config, args.skip_existing)

if __name__ == "__main__":
    main() 