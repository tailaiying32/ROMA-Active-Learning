#!/usr/bin/env python3
"""
Analyze scraped .jsonl files and generate a summary report.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def analyze_scraped_files(metadata_dir: str, output_file: str = None):
    """Analyze all scraped .jsonl files and generate a summary."""
    
    metadata_path = Path(metadata_dir)
    if not metadata_path.exists():
        print(f"Error: Directory {metadata_dir} does not exist")
        return
    
    # Find all .jsonl files
    jsonl_files = list(metadata_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No .jsonl files found in {metadata_dir}")
        return
    
    print(f"Found {len(jsonl_files)} .jsonl files")
    
    # Track results by keyword and database
    keyword_counts = defaultdict(int)
    database_counts = defaultdict(int)
    total_articles = 0
    
    # Process each file
    for file_path in jsonl_files:
        print(f"Processing: {file_path.name}")
        
        # Extract database and keyword from filename
        filename = file_path.stem
        parts = filename.split('_')
        
        if len(parts) >= 3:
            database = parts[0]
            
            # Find timestamp to extract keyword properly
            timestamp_found = False
            for i in range(len(parts) - 1, -1, -1):
                if len(parts[i]) == 8 and parts[i].isdigit():
                    keyword_parts = parts[1:i]
                    keyword = ' '.join(keyword_parts).replace('_', ' ')
                    timestamp_found = True
                    break
            
            if not timestamp_found:
                keyword_parts = parts[1:-1]
                keyword = ' '.join(keyword_parts).replace('_', ' ')
            
            # Count articles in this file
            article_count = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            article_count += 1
                
                print(f"  Database: {database}, Keyword: '{keyword}', Articles: {article_count}")
                
                # Update counts
                keyword_counts[keyword] += article_count
                database_counts[database] += article_count
                total_articles += article_count
                
            except Exception as e:
                print(f"  Error reading file: {e}")
    
    # Generate summary
    summary = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_articles": total_articles,
        "total_keywords": len(keyword_counts),
        "total_databases": len(database_counts),
        "keywords": dict(sorted(keyword_counts.items())),
        "databases": dict(sorted(database_counts.items()))
    }
    
    # Print summary
    print("\n" + "="*60)
    print("SCRAPED ARTICLES SUMMARY")
    print("="*60)
    print(f"Total articles: {total_articles}")
    print(f"Total keywords: {len(keyword_counts)}")
    print(f"Total databases: {len(database_counts)}")
    
    print(f"\nKeywords and article counts:")
    for keyword, count in sorted(keyword_counts.items()):
        print(f"  {keyword}: {count}")
    
    print(f"\nDatabase breakdown:")
    for database, count in sorted(database_counts.items()):
        print(f"  {database}: {count}")
    
    # Save to file
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"scraped_articles_summary_{timestamp}.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to: {output_file}")
    except Exception as e:
        print(f"Error saving summary: {e}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Analyze scraped .jsonl files and generate summary")
    parser.add_argument("metadata_dir", nargs="?", default="scraped_articles/metadata",
                       help="Directory containing .jsonl files (default: scraped_articles/metadata)")
    parser.add_argument("--output", "-o", help="Output JSON file (default: auto-generated filename)")
    
    args = parser.parse_args()
    
    analyze_scraped_files(args.metadata_dir, args.output)

if __name__ == "__main__":
    main()