#!/usr/bin/env python3
"""
Test script to verify paperscraper installation and basic functionality.
"""

import sys
import traceback

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        from paperscraper.pubmed import get_and_dump_pubmed_papers
        print("✓ paperscraper.pubmed.get_and_dump_pubmed_papers imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import paperscraper.pubmed.get_and_dump_pubmed_papers: {e}")
        return False
    
    try:
        from paperscraper.arxiv import get_and_dump_arxiv_papers
        print("✓ paperscraper.arxiv.get_and_dump_arxiv_papers imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import paperscraper.arxiv.get_and_dump_arxiv_papers: {e}")
        return False
    
    try:
        from paperscraper.xrxiv.xrxiv_query import XRXivQuery
        print("✓ paperscraper.xrxiv.xrxiv_query.XRXivQuery imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import paperscraper.xrxiv.xrxiv_query.XRXivQuery: {e}")
        return False
    
    try:
        from paperscraper.pdf import save_pdf_from_dump
        print("✓ paperscraper.pdf.save_pdf_from_dump imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import paperscraper.pdf.save_pdf_from_dump: {e}")
        return False
    
    return True

def test_basic_search():
    """Test a basic search to verify functionality."""
    print("\nTesting basic search...")
    
    try:
        from paperscraper.pubmed import get_and_dump_pubmed_papers
        import tempfile
        import os
        
        # Test a simple search
        query = [["caregiving"]]
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        temp_file.close()
        
        try:
            get_and_dump_pubmed_papers(query, output_filepath=temp_file.name)
            
            # Check if file was created and has content
            if os.path.exists(temp_file.name) and os.path.getsize(temp_file.name) > 0:
                print(f"✓ Basic search successful. Results saved to: {temp_file.name}")
                
                # Count results
                with open(temp_file.name, 'r') as f:
                    line_count = sum(1 for line in f if line.strip())
                print(f"  Found {line_count} results")
                return True
            else:
                print("✗ Search completed but no results found")
                return False
                
        finally:
            # Clean up
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            
    except Exception as e:
        print(f"✗ Search failed: {e}")
        traceback.print_exc()
        return False

def test_scraper_class():
    """Test if our ArticleScraper class can be imported."""
    print("\nTesting ArticleScraper class...")
    
    try:
        from article_scraper import ArticleScraper
        print("✓ ArticleScraper class imported successfully")
        
        # Test instantiation
        scraper = ArticleScraper("test_output")
        print("✓ ArticleScraper instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ ArticleScraper test failed: {e}")
        traceback.print_exc()
        return False

def test_dump_setup():
    """Test if dump setup functions can be imported."""
    print("\nTesting dump setup functions...")
    
    try:
        from paperscraper.get_dumps import biorxiv, medrxiv
        print("✓ paperscraper.get_dumps functions imported successfully")
        print("  Note: Running dumps will take significant time (30min-1h)")
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import dump functions: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Paperscraper Installation Test ===\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Search Test", test_basic_search),
        ("Scraper Class Test", test_scraper_class),
        ("Dump Setup Test", test_dump_setup)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! Paperscraper is ready to use.")
        print("\nNext steps:")
        print("1. For bioRxiv/medRxiv support, run: python article_scraper.py --setup-dumps")
        print("2. Then run the scraper: python run_scraper.py --search-only")
    else:
        print("❌ Some tests failed. Please check the installation.")
        print("\nTry installing paperscraper with:")
        print("  pip install paperscraper")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 