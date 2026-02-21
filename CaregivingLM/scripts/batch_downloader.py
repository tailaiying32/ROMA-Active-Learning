#!/usr/bin/env python3
"""
Batch Downloader for Filtered Articles

Automatically processes relevant article batches and downloads PDFs/XMLs.
Tracks download status and can resume interrupted downloads.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Set
import time
import hashlib
from run_scraper import load_config
from article_scraper import ArticleScraper


def find_filtered_directories(base_path: Path) -> List[Path]:
    """Find all *_filtered directories in the given path."""
    filtered_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.endswith('_filtered'):
            filtered_dirs.append(item)
    return sorted(filtered_dirs)


def get_article_id(article: Dict) -> str:
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


def load_download_tracking(tracking_file: Path) -> Dict:
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
        print(f"Error loading download tracking file {tracking_file}: {e}")
        return {
            "metadata": {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "attempted_articles": {}
        }


def save_download_tracking(tracking_file: Path, tracking_data: Dict):
    """Save download tracking data."""
    tracking_data["metadata"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tracking_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(tracking_data, f, indent=2, ensure_ascii=False)


def get_downloaded_articles(pdf_dir: Path) -> Set[str]:
    """Get set of already downloaded articles by scanning the PDF directory."""
    downloaded = set()
    
    if not pdf_dir.exists():
        return downloaded
    
    # Scan for PDF and XML files
    for file_path in pdf_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.xml']:
            # Extract identifier from filename (usually DOI-based)
            filename = file_path.stem
            # Common patterns: 10.1016_j.example.2024.123456
            if filename.startswith('10.'):
                downloaded.add(f"doi_{hashlib.md5(filename.replace('_', '/').encode()).hexdigest()[:12]}")
            # Handle other patterns as needed
            downloaded.add(filename)
    
    return downloaded


def filter_unattempted_articles(articles: List[Dict], download_tracking_file: Path, pdf_dir: Path) -> List[Dict]:
    """Filter out articles that have already been attempted (either successfully downloaded or failed)."""
    tracking_data = load_download_tracking(download_tracking_file)
    attempted_from_tracking = set(tracking_data["attempted_articles"].keys())
    downloaded_from_filesystem = get_downloaded_articles(pdf_dir)
    
    # Combine both sets - we skip anything that was attempted OR exists on filesystem
    all_attempted = attempted_from_tracking | downloaded_from_filesystem
    
    unattempted = []
    skipped_count = 0
    
    for article in articles:
        article_id = get_article_id(article)
        
        # Check if article has already been attempted
        if article_id in all_attempted:
            skipped_count += 1
            continue
            
        # Additional check using DOI filename pattern
        if article.get('doi'):
            doi_filename = article['doi'].replace('/', '_')
            if doi_filename in downloaded_from_filesystem:
                skipped_count += 1
                continue
        
        unattempted.append(article)
    
    print(f"   📋 Found {len(articles)} total articles, {skipped_count} already attempted, {len(unattempted)} remaining")
    
    return unattempted


def record_attempted_articles(articles: List[Dict], download_tracking_file: Path, batch_id: str, pdf_dir: Path):
    """Record all attempted articles (both successful and failed) in the tracking system."""
    tracking_data = load_download_tracking(download_tracking_file)
    
    # Get the set of files that actually exist after download
    downloaded_files = get_downloaded_articles(pdf_dir)
    
    newly_attempted = 0
    successful_downloads = 0
    
    for article in articles:
        article_id = get_article_id(article)
        
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
            "batch_id": batch_id,
            "source": article.get('source', 'unknown'),
            "download_successful": file_exists,
            "download_status": "success" if file_exists else "failed"
        }
        newly_attempted += 1
        if file_exists:
            successful_downloads += 1
    
    if newly_attempted > 0:
        save_download_tracking(download_tracking_file, tracking_data)
        print(f"   📥 Recorded {newly_attempted} attempted articles ({successful_downloads} successful, {newly_attempted - successful_downloads} failed)")
    
    return {"attempted": newly_attempted, "successful": successful_downloads}


def load_batch_tracking(tracking_file: Path) -> Dict:
    """Load batch tracking data."""
    if not tracking_file.exists():
        return None
    
    try:
        with open(tracking_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading tracking file {tracking_file}: {e}")
        return None


def update_batch_status(tracking_file: Path, batch_id: str, status: str, details: Dict = None):
    """Update the download status of a specific batch."""
    tracking_data = load_batch_tracking(tracking_file)
    if not tracking_data:
        return
    
    if batch_id in tracking_data["batches"]:
        tracking_data["batches"][batch_id]["download_status"] = status
        tracking_data["batches"][batch_id]["download_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        if details:
            tracking_data["batches"][batch_id].update(details)
        
        # Save updated tracking data
        with open(tracking_file, 'w', encoding='utf-8') as f:
            json.dump(tracking_data, f, indent=2, ensure_ascii=False)


def get_pending_batches(tracking_data: Dict) -> List[str]:
    """Get list of batch IDs that are pending download."""
    pending = []
    for batch_id, batch_info in tracking_data["batches"].items():
        if batch_info["download_status"] == "pending":
            pending.append(batch_id)
    return sorted(pending)


def create_temp_search_results_file(batch_articles: List[Dict], temp_dir: Path, batch_id: str) -> Path:
    """Create a temporary search_results format file for the downloader."""
    temp_file = temp_dir / f"temp_search_results_{batch_id}.json"
    
    # Convert batch articles to search_results format
    search_results_data = {
        "keywords": ["filtered_batch"],
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "results": {
            "pubmed": batch_articles  # Put all articles under pubmed for simplicity
        }
    }
    
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(search_results_data, f, indent=2, ensure_ascii=False)
    
    return temp_file


def process_batch(batch_file: Path, batch_id: str, tracking_file: Path, scraper: ArticleScraper, temp_dir: Path, config: Dict) -> Dict:
    """Process a single batch for download."""
    print(f"\n📦 Processing {batch_id}: {batch_file.name}")
    
    # Load batch articles
    try:
        batch_articles = []
        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    batch_articles.append(json.loads(line))
        
        print(f"   Loaded {len(batch_articles)} articles")
        
        # Filter articles that have DOI or arXiv ID (downloadable)
        downloadable_articles = [
            article for article in batch_articles 
            if article.get('doi') or article.get('arxiv_id')
        ]
        
        if not downloadable_articles:
            print("   ⚠️  No downloadable articles (no DOI or arXiv ID)")
            return {"status": "skipped", "reason": "no_downloadable_articles"}
        
        print(f"   {len(downloadable_articles)} articles have DOI/arXiv ID")
        
        # Set up download tracking
        download_tracking_file = batch_file.parent.parent / "tracking" / "download_status.json"
        pdf_dir = Path(config.get('pdf_download_path', 'scraped_articles/pdfs'))
        
        # Filter out already attempted articles
        unattempted_articles = filter_unattempted_articles(
            downloadable_articles, download_tracking_file, pdf_dir
        )
        
        if not unattempted_articles:
            print("   ✅ All articles in this batch have already been attempted")
            return {
                "status": "skipped",
                "reason": "all_already_attempted",
                "total_articles": len(batch_articles),
                "downloadable_articles": len(downloadable_articles),
                "already_attempted": len(downloadable_articles)
            }
        
        # Create temporary search results file with only unattempted articles
        temp_search_file = create_temp_search_results_file(unattempted_articles, temp_dir, batch_id)
        
        # Download using existing scraper
        print(f"   🔄 Starting download for {len(unattempted_articles)} new articles...")
        scraper.download_pdfs(str(temp_search_file), len(unattempted_articles))
        
        # Record all attempted articles (both successful and failed)
        attempt_results = record_attempted_articles(
            unattempted_articles, download_tracking_file, batch_id, pdf_dir
        )
        
        # Clean up temp file
        temp_search_file.unlink()
        
        return {
            "status": "completed",
            "total_articles": len(batch_articles),
            "downloadable_articles": len(downloadable_articles),
            "already_attempted": len(downloadable_articles) - len(unattempted_articles),
            "newly_attempted": attempt_results["attempted"],
            "successful_downloads": attempt_results["successful"],
            "failed_downloads": attempt_results["attempted"] - attempt_results["successful"]
        }
        
    except Exception as e:
        print(f"   ❌ Error processing batch: {e}")
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Download PDFs from filtered article batches")
    parser.add_argument("--filtered-dir", help="Specific filtered directory to process")
    parser.add_argument("--batch-id", help="Specific batch ID to process (e.g., batch_001)")
    parser.add_argument("--config", default="config/scraper_config.json",
                       help="Path to scraper config file")
    parser.add_argument("--list-pending", action="store_true",
                       help="List all pending batches and exit")
    parser.add_argument("--resume", action="store_true",
                       help="Resume downloading all pending batches")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    pdf_path = config.get('pdf_download_path')
    scraper = ArticleScraper(config['output_directory'], pdf_path)
    
    # Create temp directory for processing
    temp_dir = Path("temp_download_processing")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        if args.list_pending:
            # List all pending batches across all filtered directories
            base_path = Path("scraped_articles/metadata")
            filtered_dirs = find_filtered_directories(base_path)
            
            if not filtered_dirs:
                print("No filtered directories found.")
                return 0
            
            total_pending = 0
            for filtered_dir in filtered_dirs:
                tracking_file = filtered_dir / "tracking" / "batch_status.json"
                tracking_data = load_batch_tracking(tracking_file)
                
                if tracking_data:
                    pending_batches = get_pending_batches(tracking_data)
                    if pending_batches:
                        print(f"\n📁 {filtered_dir.name}:")
                        print(f"   Tracking file: {tracking_file}")
                        print(f"   Pending batches: {len(pending_batches)}")
                        for batch_id in pending_batches:
                            batch_info = tracking_data["batches"][batch_id]
                            print(f"     • {batch_id}: {batch_info['filename']} ({batch_info['article_count']} articles)")
                        total_pending += len(pending_batches)
            
            print(f"\nTotal pending batches: {total_pending}")
            return 0
        
        elif args.filtered_dir:
            # Process specific filtered directory
            filtered_dir = Path(args.filtered_dir)
            if not filtered_dir.exists():
                print(f"Error: Directory {filtered_dir} does not exist")
                return 1
            
            tracking_file = filtered_dir / "tracking" / "batch_status.json"
            tracking_data = load_batch_tracking(tracking_file)
            
            if not tracking_data:
                print(f"Error: No tracking data found in {tracking_file}")
                return 1
            
            relevant_batches_dir = filtered_dir / "relevant_batches"
            
            if args.batch_id:
                # Process specific batch
                if args.batch_id not in tracking_data["batches"]:
                    print(f"Error: Batch {args.batch_id} not found")
                    return 1
                
                batch_info = tracking_data["batches"][args.batch_id]
                batch_file = relevant_batches_dir / batch_info["filename"]
                
                if not batch_file.exists():
                    print(f"Error: Batch file {batch_file} does not exist")
                    return 1
                
                # Update status to in_progress
                update_batch_status(tracking_file, args.batch_id, "in_progress")
                
                # Process the batch
                result = process_batch(batch_file, args.batch_id, tracking_file, scraper, temp_dir, config)
                
                # Update final status
                update_batch_status(tracking_file, args.batch_id, result["status"], result)
                
                print(f"\n✅ Batch {args.batch_id} processing complete: {result['status']}")
                
            elif args.resume:
                # Process all pending batches
                pending_batches = get_pending_batches(tracking_data)
                
                if not pending_batches:
                    print("No pending batches found.")
                    return 0
                
                print(f"Found {len(pending_batches)} pending batches")
                
                for batch_id in pending_batches:
                    batch_info = tracking_data["batches"][batch_id]
                    batch_file = relevant_batches_dir / batch_info["filename"]
                    
                    if not batch_file.exists():
                        print(f"⚠️  Skipping {batch_id}: file {batch_file} not found")
                        continue
                    
                    # Update status to in_progress
                    update_batch_status(tracking_file, batch_id, "in_progress")
                    
                    # Process the batch
                    result = process_batch(batch_file, batch_id, tracking_file, scraper, temp_dir, config)
                    
                    # Update final status
                    update_batch_status(tracking_file, batch_id, result["status"], result)
                    
                    # Reload tracking data for next iteration
                    tracking_data = load_batch_tracking(tracking_file)
                
                print(f"\n✅ All pending batches processed")
                
            else:
                print("Error: When using --filtered-dir, specify either --batch-id or --resume")
                return 1
        
        else:
            print("Error: Specify --filtered-dir or use --list-pending")
            print("Use --help for usage information")
            return 1
    
    finally:
        # Clean up temp directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
    
    return 0


if __name__ == "__main__":
    exit(main())