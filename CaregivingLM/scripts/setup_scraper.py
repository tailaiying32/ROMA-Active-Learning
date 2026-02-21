#!/usr/bin/env python3
"""
Setup script for the article scraper.
This script helps users initialize the scraper system.
"""

import os
import sys
from pathlib import Path

def check_paperscraper_installation():
    """Check if paperscraper is installed."""
    try:
        import paperscraper
        print("✓ paperscraper is installed")
        return True
    except ImportError:
        print("✗ paperscraper is not installed")
        print("  Install with: pip install paperscraper")
        return False

def check_config_files():
    """Check if configuration files exist."""
    config_dir = Path("../config")
    
    print("\n=== Checking Configuration Files ===")
    
    # Check keywords file
    keywords_file = config_dir / "keywords.txt"
    if keywords_file.exists():
        print(f"✓ Keywords file found: {keywords_file}")
        with open(keywords_file, 'r') as f:
            keyword_count = len([line.strip() for line in f if line.strip()])
        print(f"  Contains {keyword_count} keywords")
    else:
        print(f"✗ Keywords file missing: {keywords_file}")
        print("  Create this file with your search terms (one per line)")
    
    # Check config file
    config_file = config_dir / "scraper_config.json"
    if config_file.exists():
        print(f"✓ Config file found: {config_file}")
    else:
        print(f"✗ Config file missing: {config_file}")
        print("  This should have been created automatically")

def check_dumps():
    """Check if bioRxiv and medRxiv dumps exist."""
    print("\n=== Checking Dumps ===")
    
    dump_path = Path("server_dumps")
    if not dump_path.exists():
        print("✗ server_dumps directory not found")
        print("  This will be created when you run --setup-dumps")
        return False
    
    biorxiv_dumps = list(dump_path.glob("biorxiv_*.jsonl"))
    medrxiv_dumps = list(dump_path.glob("medrxiv_*.jsonl"))
    
    if biorxiv_dumps:
        latest_biorxiv = max(biorxiv_dumps, key=lambda x: x.stat().st_mtime)
        size_mb = latest_biorxiv.stat().st_size / (1024 * 1024)
        print(f"✓ bioRxiv dump found: {latest_biorxiv.name} ({size_mb:.1f} MB)")
    else:
        print("❌ bioRxiv dump not found")
        print("  Run: python run_scraper.py --setup-dumps")
    
    if medrxiv_dumps:
        latest_medrxiv = max(medrxiv_dumps, key=lambda x: x.stat().st_mtime)
        size_mb = latest_medrxiv.stat().st_size / (1024 * 1024)
        print(f"✓ medRxiv dump found: {latest_medrxiv.name} ({size_mb:.1f} MB)")
    else:
        print("❌ medRxiv dump not found")
        print("  Run: python run_scraper.py --setup-dumps")
    
    return len(biorxiv_dumps) > 0 and len(medrxiv_dumps) > 0

def run_test():
    """Run a quick test to verify everything works."""
    print("\n=== Running Quick Test ===")
    
    try:
        from test_scraper import main as test_main
        success = test_main()
        if success:
            print("✓ All tests passed!")
        else:
            print("✗ Some tests failed")
        return success
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def show_next_steps():
    """Show the next steps for the user."""
    print("\n=== Next Steps ===")
    print("1. Edit config/keywords.txt with your search terms")
    print("2. For bioRxiv/medRxiv support, run:")
    print("   python run_scraper.py --setup-dumps")
    print("3. Test the scraper:")
    print("   python run_scraper.py --search-only")
    print("4. Review results and download PDFs:")
    print("   python run_scraper.py --download-only --metadata-file <path>")

def main():
    """Main setup function."""
    print("=== Article Scraper Setup ===\n")
    
    # Check installation
    if not check_paperscraper_installation():
        print("\nPlease install paperscraper first:")
        print("pip install paperscraper")
        return False
    
    # Check config files
    check_config_files()
    
    # Check dumps
    dumps_ready = check_dumps()
    
    # Run test
    test_passed = run_test()
    
    # Summary
    print("\n=== Setup Summary ===")
    if test_passed:
        print("✓ Basic setup is complete!")
        if dumps_ready:
            print("✓ Dumps are ready for bioRxiv/medRxiv")
        else:
            print("⚠ Dumps needed for bioRxiv/medRxiv (optional)")
    else:
        print("✗ Setup incomplete - check the issues above")
    
    show_next_steps()
    
    return test_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 