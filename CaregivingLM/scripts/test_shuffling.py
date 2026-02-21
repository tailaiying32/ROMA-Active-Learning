#!/usr/bin/env python3
"""
Test script to demonstrate the shuffling functionality.
This script creates a mock dataset and shows how articles are distributed across keywords.
"""

import json
import random
from pathlib import Path
from article_scraper import ArticleScraper

def create_mock_data():
    """Create mock article data for testing."""
    keywords = ["caregiving", "dementia", "alzheimer"]
    mock_articles = {
        'pubmed': []
    }
    
    # Create mock articles for each keyword
    for i, keyword in enumerate(keywords):
        # Create 10 articles for each keyword
        for j in range(10):
            article = {
                'title': f"Article {j+1} about {keyword}",
                'doi': f"10.1000/test.{keyword}.{j+1}",
                'keyword': keyword,
                'database': 'pubmed',
                'abstract': f"This is a test article about {keyword}",
                'authors': [f"Author {j+1}"],
                'journal': f"Test Journal {j+1}",
                'date': "2024-01-01"
            }
            mock_articles['pubmed'].append(article)
    
    return mock_articles, keywords

def test_shuffling():
    """Test the shuffling functionality."""
    print("=== TESTING ARTICLE SHUFFLING ===")
    
    # Create mock data
    mock_results, keywords = create_mock_data()
    
    # Create a temporary scraper
    scraper = ArticleScraper("test_output")
    
    # Save mock metadata
    metadata_file = scraper.save_metadata(mock_results, keywords)
    
    print(f"Created mock data with {len(keywords)} keywords:")
    for keyword in keywords:
        count = len([a for a in mock_results['pubmed'] if a['keyword'] == keyword])
        print(f"  - {keyword}: {count} articles")
    
    print(f"\nTotal articles: {len(mock_results['pubmed'])}")
    
    # Test with different max_pdfs values
    test_cases = [5, 10, 15, 30]
    
    for max_pdfs in test_cases:
        print(f"\n--- Testing with max_pdfs = {max_pdfs} ---")
        
        # Test without seed (random)
        print("Without seed (random):")
        articles_random = scraper._create_shuffled_list(
            {'pubmed': mock_results['pubmed']}, max_pdfs
        )
        
        # Count articles by keyword
        keyword_counts = {}
        for article in articles_random:
            keyword = article['keyword']
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        for keyword in keywords:
            print(f"  {keyword}: {keyword_counts.get(keyword, 0)} articles")
        
        # Test with fixed seed (reproducible)
        print("With seed (reproducible):")
        random.seed(42)  # Fixed seed
        articles_seeded = scraper._create_shuffled_list(
            {'pubmed': mock_results['pubmed']}, max_pdfs
        )
        
        # Count articles by keyword
        keyword_counts = {}
        for article in articles_seeded:
            keyword = article['keyword']
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        for keyword in keywords:
            print(f"  {keyword}: {keyword_counts.get(keyword, 0)} articles")
    
    # Clean up
    if Path("test_output").exists():
        import shutil
        shutil.rmtree("test_output")
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    test_shuffling() 