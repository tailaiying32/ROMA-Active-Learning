#!/usr/bin/env python3
"""
Example usage of the ArticleScraper class.
This script demonstrates how to use the scraper programmatically.
"""

from article_scraper import ArticleScraper
import json

def example_basic_usage():
    """Example of basic usage with default settings."""
    print("=== Example: Basic Usage ===")
    
    # Initialize scraper
    scraper = ArticleScraper("example_output")
    
    # Define keywords
    keywords = ["caregiving", "caregiver burden", "dementia care"]
    
    # Step 1: Search for articles
    print("Step 1: Searching for articles...")
    results = scraper.search_articles(keywords)
    
    # Step 2: Save metadata
    print("Step 2: Saving metadata...")
    metadata_file = scraper.save_metadata(results, keywords)
    
    # Step 3: Download PDFs (limited to 5 for this example)
    print("Step 3: Downloading PDFs...")
    scraper.download_pdfs(metadata_file, max_pdfs=5)
    
    print(f"Example complete! Check the 'example_output' directory for results.")

def example_search_only():
    """Example of search-only usage."""
    print("\n=== Example: Search Only ===")
    
    scraper = ArticleScraper("example_search_only")
    keywords = ["caregiver stress", "family caregiver"]
    
    # Search and save metadata
    results = scraper.search_articles(keywords)
    metadata_file = scraper.save_metadata(results, keywords)
    
    print(f"Search complete! Metadata saved to: {metadata_file}")
    print("You can review the results and run download separately.")

def example_custom_config():
    """Example with custom configuration."""
    print("\n=== Example: Custom Configuration ===")
    
    # Custom output directory
    scraper = ArticleScraper("custom_output")
    
    # Custom keywords
    keywords = ["respite care", "caregiver support"]
    
    # Search
    results = scraper.search_articles(keywords)
    
    # Save metadata
    metadata_file = scraper.save_metadata(results, keywords)
    
    # Download with custom limit
    scraper.download_pdfs(metadata_file, max_pdfs=2)
    
    print("Custom configuration example complete!")

def example_review_results():
    """Example of how to review search results before downloading."""
    print("\n=== Example: Review Results ===")
    
    scraper = ArticleScraper("review_example")
    keywords = ["caregiver intervention"]
    
    # Search
    results = scraper.search_articles(keywords)
    metadata_file = scraper.save_metadata(results, keywords)
    
    # Load and review results
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    
    print("Search Results Summary:")
    for db_name, db_results in data['results'].items():
        print(f"  {db_name}: {len(db_results)} articles")
        
        # Show first result from each database
        if db_results:
            first_article = db_results[0]
            title = first_article.get('title', 'No title')[:80]
            print(f"    Sample: {title}...")
    
    print(f"\nMetadata saved to: {metadata_file}")
    print("Review the results and decide how many PDFs to download.")

def main():
    """Run all examples."""
    print("Article Scraper Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_search_only()
        example_custom_config()
        example_review_results()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nTo run the scraper with your own keywords:")
        print("1. Edit config/keywords.txt with your search terms")
        print("2. Run: python run_scraper.py --search-only")
        print("3. Review results in the metadata files")
        print("4. Run: python run_scraper.py --download-only --metadata-file <path>")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 