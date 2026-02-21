#!/usr/bin/env python3
"""
Article Scraper using paperscraper package
A two-step approach to scrape and download article PDFs from multiple databases.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set
import logging
from datetime import datetime
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil
import hashlib
import io
import contextlib

# paperscraper imports - using the correct API
from paperscraper.pubmed import get_and_dump_pubmed_papers
from paperscraper.arxiv import get_and_dump_arxiv_papers
from paperscraper.xrxiv.xrxiv_query import XRXivQuery
from paperscraper.pdf import save_pdf_from_dump
from paperscraper.load_dumps import QUERY_FN_DICT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('article_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from paperscraper
logging.getLogger('paperscraper').setLevel(logging.CRITICAL)
logging.getLogger('paperscraper.pdf').setLevel(logging.CRITICAL)
logging.getLogger('paperscraper.pdf.fallbacks').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.CRITICAL)

class ArticleScraper:
    """Main class for scraping articles and downloading PDFs."""
    
    def __init__(self, output_dir: str = "scraped_articles", pdf_path: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different databases
        self.db_dirs = {
            'pubmed': self.output_dir / 'pubmed',
            'arxiv': self.output_dir / 'arxiv', 
            'biorxiv': self.output_dir / 'biorxiv',
            'medrxiv': self.output_dir / 'medrxiv'
        }
        
        for db_dir in self.db_dirs.values():
            db_dir.mkdir(exist_ok=True)
            
        # Create metadata directory
        self.metadata_dir = self.output_dir / 'metadata'
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Set download directory - use custom path if provided, otherwise default
        if pdf_path:
            self.download_dir = Path(pdf_path)
            self.download_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using custom download path: {self.download_dir}")
        else:
            self.download_dir = self.output_dir / 'downloads'
            self.download_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different file types
        self.pdfs_dir = self.download_dir / 'pdfs'
        self.xmls_dir = self.download_dir / 'xmls'
        self.pdfs_dir.mkdir(exist_ok=True)
        self.xmls_dir.mkdir(exist_ok=True)
        logger.info(f"Created subdirectories: {self.pdfs_dir} and {self.xmls_dir}")
        
        logger.info(f"Initialized ArticleScraper with output directory: {self.output_dir}")
    
    def get_article_id(self, article: Dict) -> str:
        """Generate a unique identifier for an article based on DOI or PMID."""
        if article.get('doi'):
            return f"doi_{hashlib.md5(article['doi'].encode()).hexdigest()[:12]}"
        elif article.get('pmid'):
            return f"pmid_{article['pmid']}"
        elif article.get('arxiv_id'):
            return f"arxiv_{hashlib.md5(article['arxiv_id'].encode()).hexdigest()[:12]}"
        else:
            # Fallback: use title hash
            title = article.get('title', 'unknown')
            return f"title_{hashlib.md5(title.encode()).hexdigest()[:12]}"

    def load_download_tracking(self, tracking_file: Path) -> Dict:
        """Load download tracking data."""
        if not tracking_file.exists():
            return {
                "metadata": {
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "attempted_articles": {}
            }
        
        try:
            with open(tracking_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading download tracking file {tracking_file}: {e}")
            return {
                "metadata": {
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "attempted_articles": {}
            }

    def save_download_tracking(self, tracking_file: Path, tracking_data: Dict):
        """Save download tracking data."""
        tracking_data["metadata"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tracking_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(tracking_file, 'w', encoding='utf-8') as f:
            json.dump(tracking_data, f, indent=2, ensure_ascii=False)

    def get_downloaded_articles(self) -> Set[str]:
        """Get set of already downloaded articles by scanning the PDF directory."""
        downloaded = set()
        
        # Scan for PDF and XML files in both directories
        for search_dir in [self.pdfs_dir, self.xmls_dir]:
            if search_dir.exists():
                for file_path in search_dir.rglob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.xml']:
                        # Extract identifier from filename (usually DOI-based)
                        filename = file_path.stem
                        # Common patterns: 10.1016_j.example.2024.123456
                        if filename.startswith('10.'):
                            # Convert back to DOI format and create ID
                            doi = filename.replace('_', '/')
                            downloaded.add(f"doi_{hashlib.md5(doi.encode()).hexdigest()[:12]}")
                        # Also add the raw filename for direct comparison
                        downloaded.add(filename)
        
        return downloaded


    def record_attempted_articles(self, articles: List[Dict], tracking_file: Path, batch_name: str = "download"):
        """Record all attempted articles (both successful and failed) in the tracking system."""
        tracking_data = self.load_download_tracking(tracking_file)
        
        # Get the set of files that actually exist after download
        downloaded_files = self.get_downloaded_articles()
        
        newly_attempted = 0
        successful_downloads = 0
        
        for article in articles:
            article_id = self.get_article_id(article)
            
            # Skip if already recorded
            if article_id in tracking_data["attempted_articles"]:
                continue
            
            # Check if the file was actually downloaded by looking for it in the filesystem
            file_exists = False
            if article.get('doi'):
                doi_filename = article['doi'].replace('/', '_')
                if doi_filename in downloaded_files:
                    file_exists = True
            
            # Also check by article_id
            if article_id in downloaded_files:
                file_exists = True
            
            # Record the attempt regardless of success/failure
            tracking_data["attempted_articles"][article_id] = {
                "title": article.get('title', ''),
                "doi": article.get('doi', ''),
                "pmid": article.get('pmid', ''),
                "arxiv_id": article.get('arxiv_id', ''),
                "attempted_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "batch_name": batch_name,
                "source": article.get('source', 'unknown'),
                "download_successful": file_exists,
                "download_status": "success" if file_exists else "failed"
            }
            newly_attempted += 1
            if file_exists:
                successful_downloads += 1
        
        if newly_attempted > 0:
            self.save_download_tracking(tracking_file, tracking_data)
            logger.info(f"Recorded {newly_attempted} attempted articles ({successful_downloads} successful, {newly_attempted - successful_downloads} failed)")
        
        return {"attempted": newly_attempted, "successful": successful_downloads}

    def _is_article_attempted(self, article: Dict, tracking_file: Path) -> bool:
        """Check if a single article has already been attempted."""
        # Use cached data if available, otherwise load
        if not hasattr(self, '_cached_tracking_data') or not hasattr(self, '_cached_downloaded_files'):
            self._cached_tracking_data = self.load_download_tracking(tracking_file)
            self._cached_downloaded_files = self.get_downloaded_articles()
        
        article_id = self.get_article_id(article)
        
        # Check tracking file
        if article_id in self._cached_tracking_data["attempted_articles"]:
            return True
        
        # Check filesystem
        if article_id in self._cached_downloaded_files:
            return True
        
        # Additional check using DOI filename pattern
        if article.get('doi'):
            doi_filename = article['doi'].replace('/', '_')
            if doi_filename in self._cached_downloaded_files:
                return True
        
        return False

    def load_keywords(self, keywords_file: str) -> List[str]:
        """Load keywords from a text file."""
        try:
            with open(keywords_file, 'r', encoding='utf-8') as f:
                keywords = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(keywords)} keywords from {keywords_file}")
            return keywords
        except FileNotFoundError:
            logger.error(f"Keywords file {keywords_file} not found!")
            return []
    
    def get_already_scraped_keywords(self, databases: List[str]) -> set:
        """Get a set of keywords that have already been scraped."""
        scraped_keywords = set()
        
        # Look for existing metadata files
        for db in databases:
            pattern = f"{db}_*.jsonl"
            existing_files = list(self.metadata_dir.glob(pattern))
            
            for file_path in existing_files:
                # Extract keyword from filename
                # Format: pubmed_keyword_name_timestamp.jsonl
                filename = file_path.stem
                parts = filename.split('_')
                if len(parts) >= 3:
                    # The timestamp is always the last part (format: YYYYMMDD_HHMMSS)
                    # Find the timestamp part by looking for the pattern (8 digits followed by 6 digits)
                    timestamp_found = False
                    for i in range(len(parts) - 1, -1, -1):
                        if len(parts[i]) == 8 and parts[i].isdigit():
                            # Found timestamp part (YYYYMMDD), keyword is everything before it
                            keyword_parts = parts[1:i]  # Skip database name and timestamp
                            keyword = ' '.join(keyword_parts).replace('_', ' ')
                            scraped_keywords.add(keyword)
                            timestamp_found = True
                            break
                    
                    # Fallback: if no timestamp pattern found, assume last part is timestamp
                    if not timestamp_found:
                        keyword_parts = parts[1:-1]  # Skip database name and last part
                        keyword = ' '.join(keyword_parts).replace('_', ' ')
                        scraped_keywords.add(keyword)
        
        return scraped_keywords
    
    def filter_keywords(self, keywords: List[str], databases: List[str], skip_existing: bool = False) -> List[str]:
        """Filter keywords to skip already scraped ones if requested."""
        if not skip_existing:
            return keywords
        
        scraped_keywords = self.get_already_scraped_keywords(databases)
        
        if scraped_keywords:
            logger.info(f"Found {len(scraped_keywords)} already scraped keywords:")
            for keyword in sorted(scraped_keywords):
                logger.info(f"  - {keyword}")
        
        filtered_keywords = [k for k in keywords if k not in scraped_keywords]

        if len(filtered_keywords) < len(keywords):
            skipped_count = len(keywords) - len(filtered_keywords)
            logger.info(f"Skipping {skipped_count} already scraped keywords")
            logger.info(f"Will process {len(filtered_keywords)} new keywords")
        else:
            logger.info("No previously scraped keywords found, processing all keywords")
        
        return filtered_keywords
    
    def _prepare_query_for_paperscraper(self, keywords: List[str]) -> List[List[str]]:
        """Convert keywords list to paperscraper query format."""
        # paperscraper expects queries as lists of lists for AND operations
        # Each inner list represents OR operations
        return [[keyword] for keyword in keywords]
    
    def search_articles(self, keywords: List[str], databases: List[str] = None, skip_existing: bool = False) -> Dict[str, List[Dict]]:
        """Step 1: Search for articles across specified databases."""
        # Default to all databases if none specified
        if databases is None:
            databases = ['pubmed', 'arxiv', 'biorxiv', 'medrxiv']
        
        # Filter keywords to skip already scraped ones if requested
        keywords = self.filter_keywords(keywords, databases, skip_existing)
        
        if not keywords:
            logger.info("No keywords to process (all have been scraped already)")
            return {db: [] for db in databases}
        
        # Initialize results dict with only the specified databases
        all_results = {db: [] for db in databases}
        
        # Convert keywords to paperscraper format
        queries = self._prepare_query_for_paperscraper(keywords)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, keyword in enumerate(keywords):
            logger.info(f"Searching for keyword ({i+1}/{len(keywords)}): {keyword}")
            
            # Create query for this keyword
            query = [[keyword]]
            
            # Search PubMed if specified
            if 'pubmed' in databases:
                try:
                    logger.info(f"Searching PubMed for: {keyword}")
                    pubmed_file = self.metadata_dir / f"pubmed_{keyword.replace(' ', '_')}_{timestamp}.jsonl"
                    get_and_dump_pubmed_papers(query, output_filepath=str(pubmed_file))
                    
                    # Load results
                    pubmed_results = self._load_jsonl_results(str(pubmed_file))
                    # Add keyword information to each article
                    for article in pubmed_results:
                        article['keyword'] = keyword
                    all_results['pubmed'].extend(pubmed_results)
                    logger.info(f"Found {len(pubmed_results)} PubMed articles for '{keyword}'")
                    
                except Exception as e:
                    logger.error(f"Error searching PubMed for '{keyword}': {e}")
            
            # Search arXiv if specified
            if 'arxiv' in databases:
                try:
                    logger.info(f"Searching arXiv for: {keyword}")
                    arxiv_file = self.metadata_dir / f"arxiv_{keyword.replace(' ', '_')}_{timestamp}.jsonl"
                    get_and_dump_arxiv_papers(query, output_filepath=str(arxiv_file))
                    
                    # Load results
                    arxiv_results = self._load_jsonl_results(str(arxiv_file))
                    # Add keyword information to each article
                    for article in arxiv_results:
                        article['keyword'] = keyword
                    all_results['arxiv'].extend(arxiv_results)
                    logger.info(f"Found {len(arxiv_results)} arXiv articles for '{keyword}'")
                    
                except Exception as e:
                    logger.error(f"Error searching arXiv for '{keyword}': {e}")
            
            # Search bioRxiv if specified
            if 'biorxiv' in databases:
                try:
                    logger.info(f"Searching bioRxiv for: {keyword}")
                    biorxiv_file = self.metadata_dir / f"biorxiv_{keyword.replace(' ', '_')}_{timestamp}.jsonl"
                    
                    # Check if bioRxiv dump exists
                    dump_path = Path("server_dumps")
                    biorxiv_dumps = list(dump_path.glob("biorxiv_*.jsonl"))
                    
                    if biorxiv_dumps:
                        # Use the most recent dump
                        latest_dump = max(biorxiv_dumps, key=lambda x: x.stat().st_mtime)
                        querier = XRXivQuery(str(latest_dump))
                        querier.search_keywords(query, output_filepath=str(biorxiv_file))
                        
                        # Load results
                        biorxiv_results = self._load_jsonl_results(str(biorxiv_file))
                        # Add keyword information to each article
                        for article in biorxiv_results:
                            article['keyword'] = keyword
                        all_results['biorxiv'].extend(biorxiv_results)
                        logger.info(f"Found {len(biorxiv_results)} bioRxiv articles for '{keyword}'")
                    else:
                        logger.warning("bioRxiv dump not found. Run 'from paperscraper.get_dumps import biorxiv; biorxiv()' first.")
                    
                except Exception as e:
                    logger.error(f"Error searching bioRxiv for '{keyword}': {e}")
            
            # Search medRxiv if specified
            if 'medrxiv' in databases:
                try:
                    logger.info(f"Searching medRxiv for: {keyword}")
                    medrxiv_file = self.metadata_dir / f"medrxiv_{keyword.replace(' ', '_')}_{timestamp}.jsonl"
                    
                    # Check if medRxiv dump exists
                    dump_path = Path("server_dumps")
                    medrxiv_dumps = list(dump_path.glob("medrxiv_*.jsonl"))
                    
                    if medrxiv_dumps:
                        # Use the most recent dump
                        latest_dump = max(medrxiv_dumps, key=lambda x: x.stat().st_mtime)
                        querier = XRXivQuery(str(latest_dump))
                        querier.search_keywords(query, output_filepath=str(medrxiv_file))
                        
                        # Load results
                        medrxiv_results = self._load_jsonl_results(str(medrxiv_file))
                        # Add keyword information to each article
                        for article in medrxiv_results:
                            article['keyword'] = keyword
                        all_results['medrxiv'].extend(medrxiv_results)
                        logger.info(f"Found {len(medrxiv_results)} medRxiv articles for '{keyword}'")
                    else:
                        logger.warning("medRxiv dump not found. Run 'from paperscraper.get_dumps import medrxiv; medrxiv()' first.")
                    
                except Exception as e:
                    logger.error(f"Error searching medRxiv for '{keyword}': {e}")
        
        return all_results
    
    def _load_jsonl_results(self, filepath: str) -> List[Dict]:
        """Load results from a JSONL file."""
        results = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        except FileNotFoundError:
            logger.warning(f"Results file {filepath} not found")
        except Exception as e:
            logger.error(f"Error loading results from {filepath}: {e}")
        
        return results
    
    def save_metadata(self, results: Dict[str, List[Dict]], keywords: List[str]):
        """Save search results metadata to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save overall results
        overall_file = self.metadata_dir / f"search_results_{timestamp}.json"
        with open(overall_file, 'w', encoding='utf-8') as f:
            json.dump({
                'keywords': keywords,
                'timestamp': timestamp,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        # Save individual database results
        for db_name, db_results in results.items():
            if db_results:
                db_file = self.metadata_dir / f"{db_name}_results_{timestamp}.json"
                with open(db_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'database': db_name,
                        'timestamp': timestamp,
                        'count': len(db_results),
                        'articles': db_results
                    }, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved {len(db_results)} {db_name} articles to {db_file}")
        
        logger.info(f"Saved all metadata to {self.metadata_dir}")
        return overall_file
    
    def download_pdfs(self, metadata_file: str, max_pdfs: int = 50, shuffle_seed: int = None, skip_attempted: bool = True):
        """Step 2: Download PDFs for articles found in step 1."""
        # Set random seed for reproducible shuffling
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            logger.info(f"Using shuffle seed: {shuffle_seed}")
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata file {metadata_file}: {e}")
            return
        
        results = data['results']
        keywords = data.get('keywords', [])
        
        # Set up download tracking
        tracking_file = self.metadata_dir / "download_tracking.json"
        batch_name = Path(metadata_file).stem
        
        # Collect articles by keyword, filtering out attempted ones FIRST if skip_attempted is True
        articles_by_keyword = {}
        total_collected = 0
        total_filtered = 0
        
        for db_name, db_results in results.items():
            for article in db_results:
                # Only include articles with DOI or arXiv ID
                if article.get('doi') or article.get('arxiv_id'):
                    total_collected += 1
                    
                    # Filter out already attempted articles BEFORE shuffling/selection
                    if skip_attempted:
                        if self._is_article_attempted(article, tracking_file):
                            total_filtered += 1
                            continue
                    
                    keyword = article.get('keyword', 'unknown')
                    if keyword not in articles_by_keyword:
                        articles_by_keyword[keyword] = []
                    articles_by_keyword[keyword].append(article)
        
        # Log distribution after filtering
        logger.info("=== ARTICLE DISTRIBUTION BY KEYWORD (AFTER FILTERING) ===")
        total_available = sum(len(articles) for articles in articles_by_keyword.values())
        logger.info(f"Total collected articles: {total_collected}")
        if skip_attempted:
            logger.info(f"Filtered out {total_filtered} already attempted articles")
        logger.info(f"Available for download: {total_available}")
        for keyword, articles in articles_by_keyword.items():
            logger.info(f"Keyword '{keyword}': {len(articles)} articles")
        
        if total_available == 0:
            logger.info("No articles available for download after filtering.")
            return
        
        # Create evenly distributed and shuffled list from filtered articles
        articles_to_download = self._create_shuffled_list(articles_by_keyword, max_pdfs)
        
        logger.info(f"Selected {len(articles_to_download)} articles for download (max: {max_pdfs})")
        
        logger.info(f"Prepared {len(articles_to_download)} articles for PDF download")
        
        # Create a temporary combined JSONL file for paperscraper
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
                for article in articles_to_download:
                    temp_file.write(json.dumps(article) + '\n')
                temp_filename = temp_file.name
            
            logger.info("Starting downloads...")
            # Download articles one by one to have control over file placement
            self._download_articles_individually(articles_to_download)
            
            # Record all attempted articles (both successful and failed)
            if skip_attempted:
                attempt_results = self.record_attempted_articles(articles_to_download, tracking_file, batch_name)
                logger.info(f"Download session complete. Attempted: {attempt_results['attempted']}, Successful: {attempt_results['successful']}")
            
        except Exception as e:
            logger.error(f"Error during download: {e}")
            # In case of major error, still try to analyze what we might have
            self._analyze_download_results_with_counts(articles_to_download, 0, 0)
            
            # Still record the attempts even if there was an error
            if skip_attempted:
                self.record_attempted_articles(articles_to_download, tracking_file, batch_name)
                
        finally:
            # Clean up temporary file
            try:
                import os
                os.unlink(temp_filename)
            except:
                pass
    
    def _create_shuffled_list(self, articles_by_keyword: Dict[str, List[Dict]], max_pdfs: int) -> List[Dict]:
        """Create a shuffled list of articles with even distribution across keywords."""
        if not articles_by_keyword:
            return []
        
        # Calculate how many articles to take from each keyword
        num_keywords = len(articles_by_keyword)
        articles_per_keyword = max_pdfs // num_keywords
        remaining_articles = max_pdfs % num_keywords
        
        logger.info(f"Distributing {max_pdfs} articles across {num_keywords} keywords")
        logger.info(f"Base articles per keyword: {articles_per_keyword}")
        logger.info(f"Extra articles to distribute: {remaining_articles}")
        
        shuffled_articles = []
        
        # Convert to list for random selection
        keyword_list = list(articles_by_keyword.keys())
        
        # Shuffle the keyword order for extra articles
        random.shuffle(keyword_list)
        
        for i, keyword in enumerate(keyword_list):
            articles = articles_by_keyword[keyword]
            
            # Determine how many articles to take from this keyword
            articles_to_take = articles_per_keyword
            if i < remaining_articles:
                articles_to_take += 1
            
            # Shuffle articles for this keyword
            random.shuffle(articles)
            
            # Take the required number of articles
            selected_articles = articles[:articles_to_take]
            
            logger.info(f"Keyword '{keyword}': taking {len(selected_articles)} articles (from {len(articles)} available)")
            shuffled_articles.extend(selected_articles)
        
        # Final shuffle of the entire list
        random.shuffle(shuffled_articles)
        
        # Log final distribution
        logger.info("=== FINAL DISTRIBUTION ===")
        keyword_counts = {}
        for article in shuffled_articles:
            keyword = article.get('keyword', 'unknown')
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        for keyword, count in keyword_counts.items():
            logger.info(f"Keyword '{keyword}': {count} articles")
        
        return shuffled_articles
    
    def _get_directory_size(self, directory: Path) -> int:
        """Get the total size of a directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except (OSError, FileNotFoundError):
            pass
        return total_size
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def _check_disk_usage_and_pause(self, completed_count: int, total_articles: int, last_size_check: dict) -> bool:
        """Check disk usage and pause if needed. Returns True to continue, False to stop."""
        # Check size every 10 downloads or when reaching 50GB boundaries
        should_check = (completed_count % 10 == 0) or (completed_count == total_articles)
        
        if not should_check and last_size_check.get('last_pause_gb', 0) == last_size_check.get('current_gb', 0):
            return True
        
        # Calculate total size of download directory
        total_size = self._get_directory_size(self.download_dir)
        size_gb = total_size / (1024**3)
        
        # Update tracking
        last_size_check['current_size'] = total_size
        last_size_check['current_gb'] = size_gb
        
        # Calculate completion percentage
        completion_pct = (completed_count / total_articles * 100) if total_articles > 0 else 0
        
        # Check if we've crossed a 50GB boundary
        current_50gb_mark = int(size_gb // 50)
        last_50gb_mark = last_size_check.get('last_pause_gb', 0) // 50
        
        if current_50gb_mark > last_50gb_mark and size_gb >= 50:
            # Pause and ask user
            print(f"\n" + "="*80)
            print(f"🚨 DISK SPACE WARNING")
            print(f"="*80)
            print(f"📁 Download directory: {self.download_dir}")
            print(f"💾 Current size: {self._format_size(total_size)} ({size_gb:.2f} GB)")
            print(f"📊 Progress: {completed_count}/{total_articles} articles ({completion_pct:.1f}%)")
            print(f"🎯 Reached {current_50gb_mark * 50} GB milestone")
            
            if size_gb >= 100:  # Show disk space info for larger sizes
                try:
                    disk_usage = shutil.disk_usage(self.download_dir)
                    free_gb = disk_usage.free / (1024**3)
                    total_disk_gb = disk_usage.total / (1024**3)
                    used_pct = ((total_disk_gb - free_gb) / total_disk_gb) * 100
                    print(f"💽 Disk space: {free_gb:.1f} GB free / {total_disk_gb:.1f} GB total ({used_pct:.1f}% used)")
                except:
                    pass
            
            print(f"\nDownload will continue unless you choose to stop.")
            print("="*80)
            
            response = input("Continue downloading? (Y/n): ").strip().lower()
            if response in ['n', 'no']:
                print("Download stopped by user.")
                return False
            
            # Update last pause mark
            last_size_check['last_pause_gb'] = current_50gb_mark * 50
            print("Continuing download...\n")
        
        return True
    
    def _download_single_article(self, article: Dict, article_index: int, suppress_output: bool = True) -> Dict:
        """Download a single article and return result status."""
        title = article.get('title', 'Unknown Title')[:50] + "..." if len(article.get('title', '')) > 50 else article.get('title', 'Unknown Title')
        doi = article.get('doi')
        arxiv_id = article.get('arxiv_id')
        
        result = {
            'index': article_index,
            'title': title,
            'doi': doi,
            'arxiv_id': arxiv_id,
            'success': False,
            'file_type': None,
            'error': None
        }
        
        try:
            # Create temporary file for this single article
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
                temp_file.write(json.dumps(article) + '\n')
                single_article_file = temp_file.name
            
            # Try downloading - suppress all output
            try:
                # Comprehensive output suppression for paperscraper
                if suppress_output:
                    import sys
                    import os
                    from contextlib import redirect_stdout, redirect_stderr
                    import io
                    
                    # Suppress all possible output streams
                    devnull = open(os.devnull, 'w')
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    
                    try:
                        # Redirect to devnull to catch any progress bars
                        sys.stdout = devnull
                        sys.stderr = devnull
                        
                        # Also use context managers as backup
                        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                            save_pdf_from_dump(
                                single_article_file,
                                pdf_path=str(self.pdfs_dir),
                                key_to_save='doi',
                                api_keys="API_KEYS.txt"
                            )
                    finally:
                        # Restore original streams
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                        devnull.close()
                else:
                    save_pdf_from_dump(
                        single_article_file,
                        pdf_path=str(self.pdfs_dir),
                        key_to_save='doi',
                        api_keys="API_KEYS.txt"
                    )
                
                # Check what was actually downloaded
                if doi:
                    pdf_filename = doi.replace('/', '_').replace(':', '_') + '.pdf'
                    xml_filename = doi.replace('/', '_').replace(':', '_') + '.xml'
                    
                    pdf_path = self.pdfs_dir / pdf_filename
                    xml_path = self.pdfs_dir / xml_filename
                    
                    if pdf_path.exists():
                        result['success'] = True
                        result['file_type'] = 'pdf'
                    elif xml_path.exists():
                        # Move XML to correct directory
                        xml_path.rename(self.xmls_dir / xml_filename)
                        result['success'] = True
                        result['file_type'] = 'xml'
                    else:
                        result['error'] = "File not found after download attempt"
                
                elif arxiv_id:
                    arxiv_filename = arxiv_id + '.pdf'
                    arxiv_path = self.pdfs_dir / arxiv_filename
                    
                    if arxiv_path.exists():
                        result['success'] = True
                        result['file_type'] = 'pdf'
                    else:
                        result['error'] = "File not found after download attempt"
                
            
            except Exception as download_error:
                result['error'] = str(download_error)
            
            # Clean up temp file
            try:
                os.unlink(single_article_file)
            except:
                pass
                
        except Exception as e:
            result['error'] = f"Setup error: {str(e)}"
        
        return result

    def _download_articles_multithreaded(self, articles_to_download: List[Dict], max_workers: int = 8):
        """Download articles using multiple threads with ONE shared progress bar."""
        pdf_count = 0
        xml_count = 0
        failed_count = 0
        completed_count = 0
        total_articles = len(articles_to_download)
        
        # Thread-safe counters and progress bar
        lock = threading.Lock()
        
        # Disk space monitoring
        last_size_check = {'last_pause_gb': 0, 'current_gb': 0, 'current_size': 0}
        download_stopped = False
        
        # Set environment variables to disable progress bars in subprocesses
        original_tqdm_disable = os.environ.get('TQDM_DISABLE', '')
        original_progress_disable = os.environ.get('PROGRESS_DISABLE', '')
        
        os.environ['TQDM_DISABLE'] = '1'
        os.environ['PROGRESS_DISABLE'] = '1'
        
        # Check initial disk space
        initial_size = self._get_directory_size(self.download_dir)
        print(f"Starting download of {total_articles} articles using {max_workers} threads...")
        print(f"Initial download directory size: {self._format_size(initial_size)}")
        print("Will pause every 50 GB for user confirmation.\n")
        
        # Create ONE progress bar for all threads
        pbar = tqdm(total=total_articles, desc="Downloading", unit="article", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        
        try:
            # Submit all download tasks
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_article = {
                    executor.submit(self._download_single_article, article, i, True): (article, i) 
                    for i, article in enumerate(articles_to_download, 1)
                }
                
                # Process completed tasks as they finish
                for future in as_completed(future_to_article):
                    if download_stopped:
                        break
                        
                    article, article_index = future_to_article[future]
                    
                    try:
                        result = future.result()
                        
                        with lock:
                            completed_count += 1
                            
                            if result['success']:
                                if result['file_type'] == 'pdf':
                                    pdf_count += 1
                                elif result['file_type'] == 'xml':
                                    xml_count += 1
                            else:
                                failed_count += 1
                            
                            # Update the shared progress bar
                            total_success = pdf_count + xml_count
                            success_rate = (total_success / completed_count * 100) if completed_count > 0 else 0
                            
                            # Add current size to progress bar
                            current_size_mb = last_size_check.get('current_size', 0) / (1024**2)
                            
                            pbar.set_postfix({
                                'PDFs': pdf_count,
                                'XMLs': xml_count,
                                'Failed': failed_count,
                                'Success': f'{success_rate:.1f}%',
                                'Size': f'{current_size_mb:.0f}MB'
                            })
                            pbar.update(1)
                            
                            # Check disk usage periodically (releases lock temporarily)
                            temp_completed = completed_count
                        
                        # Check disk usage outside the lock to avoid blocking other threads
                        if not self._check_disk_usage_and_pause(temp_completed, total_articles, last_size_check):
                            with lock:
                                download_stopped = True
                            break
                    
                    except Exception as e:
                        with lock:
                            completed_count += 1
                            failed_count += 1
                            
                            total_success = pdf_count + xml_count
                            success_rate = (total_success / completed_count * 100) if completed_count > 0 else 0
                            
                            current_size_mb = last_size_check.get('current_size', 0) / (1024**2)
                            
                            pbar.set_postfix({
                                'PDFs': pdf_count,
                                'XMLs': xml_count,
                                'Failed': failed_count,
                                'Success': f'{success_rate:.1f}%',
                                'Size': f'{current_size_mb:.0f}MB'
                            })
                            pbar.update(1)
        
        finally:
            pbar.close()
            # Restore original environment variables
            if original_tqdm_disable:
                os.environ['TQDM_DISABLE'] = original_tqdm_disable
            else:
                os.environ.pop('TQDM_DISABLE', None)
            
            if original_progress_disable:
                os.environ['PROGRESS_DISABLE'] = original_progress_disable
            else:
                os.environ.pop('PROGRESS_DISABLE', None)
        
        # Final summary
        total_successful = pdf_count + xml_count
        success_rate = (total_successful / completed_count * 100) if completed_count > 0 else 0
        
        # Calculate final directory size
        final_size = self._get_directory_size(self.download_dir)
        final_size_gb = final_size / (1024**3)
        
        if download_stopped:
            print(f"\n⏹️  Download stopped by user!")
            print(f"📊 Results: {total_successful}/{completed_count} completed articles successful ({success_rate:.1f}%)")
            print(f"   📊 Progress: {completed_count}/{total_articles} total articles ({completed_count/total_articles*100:.1f}% complete)")
        else:
            print(f"\n✅ Download complete!")
            print(f"📊 Results: {total_successful}/{total_articles} successful ({success_rate:.1f}%)")
        
        print(f"   📄 PDFs: {pdf_count}")
        print(f"   📄 XMLs: {xml_count}")
        print(f"   ❌ Failed: {failed_count}")
        print(f"   💾 Final size: {self._format_size(final_size)} ({final_size_gb:.2f} GB)")
        print(f"   📁 Saved to: PDFs → {self.pdfs_dir}")
        print(f"   📁 Saved to: XMLs → {self.xmls_dir}")
        
        # Update the analyze method to use our actual counts
        self._analyze_download_results_with_counts(articles_to_download, pdf_count, xml_count)

    def _download_articles_individually(self, articles_to_download: List[Dict]):
        """Download articles with configurable threading."""
        # Check if multi-threading is configured
        max_workers = getattr(self, 'max_download_workers', 8)  # Default to 8 threads
        
        # Temporarily suppress logger to reduce noise during downloads
        original_level = logger.level
        logger.setLevel(logging.WARNING)
        
        try:
            if max_workers > 1:
                self._download_articles_multithreaded(articles_to_download, max_workers)
            else:
                self._download_articles_singlethreaded(articles_to_download)
        finally:
            # Restore original logging level
            logger.setLevel(original_level)
    
    def _download_articles_singlethreaded(self, articles_to_download: List[Dict]):
        """Single-threaded download with clean progress bar."""
        pdf_count = 0
        xml_count = 0
        failed_count = 0
        total_articles = len(articles_to_download)
        
        # Disk space monitoring
        last_size_check = {'last_pause_gb': 0, 'current_gb': 0, 'current_size': 0}
        
        # Check initial disk space
        initial_size = self._get_directory_size(self.download_dir)
        print(f"Starting single-threaded download of {total_articles} articles...")
        print(f"Initial download directory size: {self._format_size(initial_size)}")
        print("Will pause every 50 GB for user confirmation.\n")
        
        # Create progress bar
        with tqdm(total=total_articles, desc="Downloading", unit="article") as pbar:
            for i, article in enumerate(articles_to_download, 1):
                result = self._download_single_article(article, i)
                
                if result['success']:
                    if result['file_type'] == 'pdf':
                        pdf_count += 1
                    elif result['file_type'] == 'xml':
                        xml_count += 1
                else:
                    failed_count += 1
                
                # Update size tracking
                if i % 10 == 0 or i == total_articles:  # Check every 10 downloads
                    current_size = self._get_directory_size(self.download_dir)
                    last_size_check['current_size'] = current_size
                    last_size_check['current_gb'] = current_size / (1024**3)
                
                current_size_mb = last_size_check.get('current_size', 0) / (1024**2)
                
                # Update progress bar
                pbar.set_postfix({
                    'PDFs': pdf_count,
                    'XMLs': xml_count,
                    'Failed': failed_count,
                    'Success': f'{((pdf_count + xml_count) / i * 100):.1f}%',
                    'Size': f'{current_size_mb:.0f}MB'
                })
                pbar.update(1)
                
                # Check disk usage and potentially pause
                if not self._check_disk_usage_and_pause(i, total_articles, last_size_check):
                    print("Download stopped by user.")
                    break
        
        # Final summary
        total_successful = pdf_count + xml_count
        completed_count = pdf_count + xml_count + failed_count
        success_rate = (total_successful / completed_count * 100) if completed_count > 0 else 0
        
        # Calculate final directory size
        final_size = self._get_directory_size(self.download_dir)
        final_size_gb = final_size / (1024**3)
        
        if completed_count < total_articles:
            print(f"\n⏹️  Download stopped by user!")
            print(f"📊 Results: {total_successful}/{completed_count} completed articles successful ({success_rate:.1f}%)")
            print(f"   📊 Progress: {completed_count}/{total_articles} total articles ({completed_count/total_articles*100:.1f}% complete)")
        else:
            print(f"\n✅ Download complete!")
            print(f"📊 Results: {total_successful}/{total_articles} successful ({success_rate:.1f}%)")
        
        print(f"   📄 PDFs: {pdf_count}")
        print(f"   📄 XMLs: {xml_count}")
        print(f"   ❌ Failed: {failed_count}")
        print(f"   💾 Final size: {self._format_size(final_size)} ({final_size_gb:.2f} GB)")
        print(f"   📁 Saved to: PDFs → {self.pdfs_dir}")
        print(f"   📁 Saved to: XMLs → {self.xmls_dir}")
        
        # Update the analyze method to use our actual counts
        self._analyze_download_results_with_counts(articles_to_download, pdf_count, xml_count)
    
    def _analyze_download_results_with_counts(self, articles_to_download: List[Dict], pdf_count: int, xml_count: int):
        """Analyze download results with actual counts."""
        logger.info("=== DOWNLOAD ANALYSIS ===")
        
        total_downloaded = pdf_count + xml_count
        total_attempted = len(articles_to_download)
        
        logger.info(f"Total articles attempted: {total_attempted}")
        logger.info(f"Successfully downloaded: {total_downloaded}")
        logger.info(f"  - PDFs: {pdf_count}")
        logger.info(f"  - XMLs: {xml_count}")
        logger.info(f"Failed downloads: {total_attempted - total_downloaded}")
        
        if total_attempted > 0:
            success_rate = (total_downloaded / total_attempted) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        # Save detailed results to a log file
        self._save_download_summary_with_counts(articles_to_download, pdf_count, xml_count)
    
    def _save_download_summary_with_counts(self, articles: List[Dict], pdf_count: int, xml_count: int):
        """Save a detailed download summary to a log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.metadata_dir / f"download_summary_{timestamp}.json"
        
        total_downloaded = pdf_count + xml_count
        
        summary_data = {
            'timestamp': timestamp,
            'total_articles': len(articles),
            'total_downloaded': total_downloaded,
            'pdfs_downloaded': pdf_count,
            'xmls_downloaded': xml_count,
            'failed_downloads': len(articles) - total_downloaded,
            'success_rate': f"{(total_downloaded / len(articles) * 100):.1f}%" if articles else "0%",
            'download_directories': {
                'pdfs': str(self.pdfs_dir),
                'xmls': str(self.xmls_dir)
            }
        }
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Download summary saved to: {summary_file}")
        except Exception as e:
            logger.error(f"Error saving download summary: {e}")
    
    def _analyze_download_results(self, articles_to_download: List[Dict]):
        """Analyze the results of PDF downloads and log detailed information."""
        logger.info("=== PDF DOWNLOAD ANALYSIS ===")
        
        successful_count = 0
        failed_count = 0
        no_identifier_count = 0
        missing_pdf_count = 0
        
        for article in articles_to_download:
            title = article.get('title', 'Unknown Title')[:100] + "..." if len(article.get('title', '')) > 100 else article.get('title', 'Unknown Title')
            doi = article.get('doi')
            arxiv_id = article.get('arxiv_id')
            
            # Check if PDF was downloaded
            pdf_found = False
            if doi:
                # Check for DOI-based PDF
                doi_filename = doi.replace('/', '_').replace(':', '_') + '.pdf'
                pdf_path = self.pdfs_dir / doi_filename
                if pdf_path.exists():
                    pdf_found = True
                    successful_count += 1
                    logger.info(f"✓ Downloaded: {title} (DOI: {doi})")
                else:
                    failed_count += 1
                    logger.warning(f"✗ Failed to download: {title} (DOI: {doi}) - PDF not found")
                    missing_pdf_count += 1
            
            elif arxiv_id:
                # Check for arXiv-based PDF
                arxiv_filename = arxiv_id + '.pdf'
                pdf_path = self.pdfs_dir / arxiv_filename
                if pdf_path.exists():
                    pdf_found = True
                    successful_count += 1
                    logger.info(f"✓ Downloaded: {title} (arXiv: {arxiv_id})")
                else:
                    failed_count += 1
                    logger.warning(f"✗ Failed to download: {title} (arXiv: {arxiv_id}) - PDF not found")
                    missing_pdf_count += 1
            
            else:
                no_identifier_count += 1
                logger.warning(f"✗ Skipped: {title} - No DOI or arXiv ID")
        
        # Summary
        logger.info("=== DOWNLOAD SUMMARY ===")
        logger.info(f"Total articles processed: {len(articles_to_download)}")
        logger.info(f"Successfully downloaded: {successful_count}")
        logger.info(f"Failed to download: {failed_count}")
        logger.info(f"Skipped (no identifier): {no_identifier_count}")
        
        if failed_count > 0:
            logger.info("=== FAILURE REASONS ===")
            logger.info(f"PDFs not found/downloaded: {missing_pdf_count}")
            logger.info("Common reasons for failures:")
            logger.info("- Article is not open access")
            logger.info("- PDF requires institutional access")
            logger.info("- Article is behind a paywall")
            logger.info("- DOI/arXiv ID is invalid or outdated")
            logger.info("- Network connectivity issues")
            logger.info("- Rate limiting by the source")
        
        # Save detailed results to a log file
        self._save_download_summary(articles_to_download, successful_count, failed_count, no_identifier_count)
    
    def _save_download_summary(self, articles: List[Dict], successful: int, failed: int, skipped: int):
        """Save a detailed download summary to a log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.metadata_dir / f"pdf_download_summary_{timestamp}.json"
        
        summary_data = {
            'timestamp': timestamp,
            'total_articles': len(articles),
            'successful_downloads': successful,
            'failed_downloads': failed,
            'skipped_articles': skipped,
            'success_rate': f"{(successful / len(articles) * 100):.1f}%" if articles else "0%",
            'articles': []
        }
        
        for article in articles:
            title = article.get('title', 'Unknown Title')
            doi = article.get('doi')
            arxiv_id = article.get('arxiv_id')
            
            # Check if PDF exists
            pdf_exists = False
            if doi:
                doi_filename = doi.replace('/', '_').replace(':', '_') + '.pdf'
                pdf_path = self.pdfs_dir / doi_filename
                pdf_exists = pdf_path.exists()
            elif arxiv_id:
                arxiv_filename = arxiv_id + '.pdf'
                pdf_path = self.pdfs_dir / arxiv_filename
                pdf_exists = pdf_path.exists()
            
            article_summary = {
                'title': title,
                'doi': doi,
                'arxiv_id': arxiv_id,
                'database': article.get('database', 'unknown'),
                'pdf_downloaded': pdf_exists,
                'failure_reason': self._get_failure_reason(article, pdf_exists)
            }
            summary_data['articles'].append(article_summary)
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Download summary saved to: {summary_file}")
        except Exception as e:
            logger.error(f"Error saving download summary: {e}")
    
    def _get_failure_reason(self, article: Dict, pdf_exists: bool) -> str:
        """Determine the reason for download failure."""
        if pdf_exists:
            return "Success"
        
        doi = article.get('doi')
        arxiv_id = article.get('arxiv_id')
        
        if not doi and not arxiv_id:
            return "No DOI or arXiv ID available"
        
        if doi:
            return "PDF not accessible (likely not open access or requires institutional access)"
        elif arxiv_id:
            return "PDF not accessible (likely not open access or requires institutional access)"
        
        return "Unknown reason"
    
    def setup_dumps(self):
        """Setup the required dumps for bioRxiv and medRxiv."""
        logger.info("Setting up dumps for bioRxiv and medRxiv...")
        
        try:
            from paperscraper.get_dumps import biorxiv, medrxiv
            
            logger.info("Downloading medRxiv dump (takes ~30min)...")
            medrxiv()
            
            logger.info("Downloading bioRxiv dump (takes ~1h)...")
            biorxiv()
            
            logger.info("Dumps setup complete! Please restart your Python interpreter.")
            
        except Exception as e:
            logger.error(f"Error setting up dumps: {e}")

def main():
    parser = argparse.ArgumentParser(description='Scrape articles and download PDFs using paperscraper')
    parser.add_argument('--keywords', '-k', required=True, help='Path to keywords file')
    parser.add_argument('--output-dir', '-o', default='scraped_articles', help='Output directory')
    parser.add_argument('--databases', '-d', nargs='+', 
                       default=['pubmed', 'arxiv', 'biorxiv', 'medrxiv'],
                       choices=['pubmed', 'arxiv', 'biorxiv', 'medrxiv'],
                       help='Databases to search (default: all)')
    parser.add_argument('--max-pdfs', '-p', type=int, default=50, help='Maximum number of PDFs to download')
    parser.add_argument('--step', '-s', choices=['search', 'download', 'both'], default='both',
                       help='Which step to run: search (step 1), download (step 2), or both')
    parser.add_argument('--metadata-file', '-m', help='Metadata file for download step (required if step=download)')
    parser.add_argument('--setup-dumps', action='store_true', help='Setup bioRxiv and medRxiv dumps')
    parser.add_argument('--shuffle-seed', type=int, help='Random seed for reproducible shuffling of articles')
    parser.add_argument('--skip-existing', action='store_true', 
                       help='Skip keywords that have already been scraped')
    
    args = parser.parse_args()
    
    scraper = ArticleScraper(args.output_dir)
    
    if args.setup_dumps:
        scraper.setup_dumps()
        return
    
    if args.step in ['search', 'both']:
        logger.info("=== STEP 1: SEARCHING ARTICLES ===")
        logger.info(f"Searching databases: {args.databases}")
        keywords = scraper.load_keywords(args.keywords)
        if not keywords:
            logger.error("No keywords loaded. Exiting.")
            return
        
        results = scraper.search_articles(keywords, args.databases, args.skip_existing)
        metadata_file = scraper.save_metadata(results, keywords)
        logger.info(f"Search complete. Metadata saved to: {metadata_file}")
    
    if args.step in ['download', 'both']:
        logger.info("=== STEP 2: DOWNLOADING PDFS ===")
        if args.step == 'download' and not args.metadata_file:
            logger.error("--metadata-file is required when step=download")
            return
        
        metadata_file = args.metadata_file if args.step == 'download' else metadata_file
        scraper.download_pdfs(metadata_file, args.max_pdfs, args.shuffle_seed)
    
    logger.info("Article scraping process complete!")

if __name__ == "__main__":
    main() 