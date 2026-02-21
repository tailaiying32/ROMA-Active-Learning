#!/usr/bin/env python3
"""
Batched Article Filter Script

Processes articles in batches using GPU batching for efficient relevance assessment,
and separates relevant from irrelevant entries into two output files.

Optimized with native HuggingFace batching for speed.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import argparse
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


def setup_logging(log_file: str) -> logging.Logger:
    """Set up logging to both file and console"""
    logger = logging.getLogger('batched_article_filter')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler for all logs
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler for warnings and errors only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_articles_from_search_results(file_path: str) -> List[Dict]:
    """Load articles from search_results JSON file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    articles = []
    # Extract all articles from all databases in the results
    for db_name, db_results in data['results'].items():
        for article in db_results:
            # Only include articles with DOI or arXiv ID for filtering
            if article.get('doi') or article.get('arxiv_id'):
                articles.append(article)
    
    return articles


def load_prompt_context(file_path: str) -> str:
    """Load prompt context from a text file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


def save_articles_to_jsonl(articles: List[Dict], file_path: str) -> None:
    """Save articles to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as file:
        for article in articles:
            file.write(json.dumps(article, ensure_ascii=False) + '\n')

def append_articles_to_jsonl(articles: List[Dict], file_path: str) -> None:
    """Append articles to JSONL file"""
    with open(file_path, 'a', encoding='utf-8') as file:
        for article in articles:
            file.write(json.dumps(article, ensure_ascii=False) + '\n')


def update_tracking_file(tracking_file: Path, tracking_data: dict):
    """Update the tracking file with current status"""
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(tracking_data, f, indent=2, ensure_ascii=False)


def create_individual_prompt(article: Dict, prompt_context: str) -> str:
    """Create a prompt for a single article"""
    article_data = f"\n\nHere is the research article to evaluate:\n\n"
    article_data += f"1. Title: {article['title']}\n"
    if article.get('journal'):
        article_data += f"   Journal: {article['journal']}\n"
    article_data += "\n"
    
    return prompt_context + article_data


def create_grouped_prompt(articles: List[Dict], prompt_context: str) -> str:
    """Create a prompt for multiple articles grouped together"""
    article_data = f"\n\nHere are the research articles to evaluate:\n\n"
    
    for i, article in enumerate(articles, 1):
        article_data += f"{i}. Title: {article['title']}\n"
        if article.get('journal'):
            article_data += f"   Journal: {article['journal']}\n"
        article_data += "\n"
    
    return prompt_context + article_data


def prepare_individual_prompts(articles: List[Dict], tokenizer, prompt_context: str, enable_thinking: bool = True) -> List[str]:
    """Prepare individual prompts for each article in the batch"""
    prompts = []
    for article in articles:
        prompt = create_individual_prompt(article, prompt_context)
        
        # Create chat message
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        prompts.append(text)
    
    return prompts


def prepare_grouped_prompts(articles: List[Dict], tokenizer, prompt_context: str, article_group_size: int, enable_thinking: bool = True) -> Tuple[List[str], List[List[Dict]]]:
    """Prepare prompts with articles grouped together to reduce prompt overhead"""
    prompts = []
    article_groups = []
    
    # Group articles into chunks
    for i in range(0, len(articles), article_group_size):
        group = articles[i:i + article_group_size]
        article_groups.append(group)
        
        # Create grouped prompt
        prompt = create_grouped_prompt(group, prompt_context)
        
        # Create chat message
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        prompts.append(text)
    
    return prompts, article_groups


def parse_grouped_response(content: str, expected_articles: int, verbose: bool = False) -> List[bool]:
    """Parse response for grouped articles and return list of relevance decisions"""
    import re
    
    relevance_decisions = []
    
    # Find all ARTICLE patterns with their numbers and relevance
    article_pattern = r'ARTICLE\s+(\d+).*?RELEVANT:\s*(YES|NO)'
    matches = re.findall(article_pattern, content.upper(), re.DOTALL)
    
    if verbose:
        print(f"    Parsing response for {expected_articles} expected articles")
        print(f"    Found {len(matches)} article patterns in response")
        for match in matches:
            print(f"      Article {match[0]}: {match[1]}")
    
    # SANITY CHECK: Warn if we found more patterns than expected
    if len(matches) > expected_articles:
        print(f"⚠️  WARNING: Found {len(matches)} article patterns but expected only {expected_articles}")
        if verbose:
            print(f"    All matches: {matches}")
    
    # Create a mapping from article number to relevance
    article_relevance = {}
    duplicate_articles = []
    
    for article_num_str, relevance in matches:
        try:
            article_num = int(article_num_str)
            
            # SANITY CHECK: Detect duplicate article numbers
            if article_num in article_relevance:
                duplicate_articles.append(article_num)
                if verbose:
                    print(f"      Warning: Duplicate decision for Article {article_num}")
                continue
            
            # SANITY CHECK: Detect out-of-range article numbers
            if article_num < 1 or article_num > expected_articles:
                if verbose:
                    print(f"      Warning: Article number {article_num} out of range (expected 1-{expected_articles})")
                continue
                
            article_relevance[article_num] = (relevance == 'YES')
        except ValueError:
            if verbose:
                print(f"      Warning: Could not parse article number '{article_num_str}'")
    
    # Report duplicates if found
    if duplicate_articles:
        print(f"⚠️  WARNING: Found duplicate article decisions: {duplicate_articles}")
    
    # Build the final list based on expected number of articles
    missing_articles = []
    for i in range(1, expected_articles + 1):
        if i in article_relevance:
            relevance_decisions.append(article_relevance[i])
        else:
            # If article not found in response, default to irrelevant
            relevance_decisions.append(False)
            missing_articles.append(i)
    
    # Report missing articles
    if missing_articles:
        print(f"⚠️  WARNING: Articles {missing_articles} not found in response, defaulting to IRRELEVANT")
        if verbose:
            print(f"    Response content preview: {content[:500]}...")
    
    # FINAL SANITY CHECK
    if len(relevance_decisions) != expected_articles:
        error_msg = f"CRITICAL PARSING ERROR: Generated {len(relevance_decisions)} decisions for {expected_articles} articles"
        print(f"⚠️  {error_msg}")
        # Ensure we always return the right number of decisions
        while len(relevance_decisions) < expected_articles:
            relevance_decisions.append(False)
        relevance_decisions = relevance_decisions[:expected_articles]
    
    return relevance_decisions


def process_articles_batch(model, tokenizer, prompt_context: str, articles: List[Dict], article_group_size: int = 1, enable_thinking: bool = True, verbose: bool = False, logger: logging.Logger = None) -> Tuple[List[Dict], List[Dict]]:
    """Process a batch of articles using GPU batching with article grouping to reduce prompt overhead"""
    if len(articles) == 0:
        return [], []
    
    # Set left padding for decoder-only models
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if verbose:
        print(f"  Processing batch of {len(articles)} articles...")
    
    # Prepare prompts - either individual or grouped based on article_group_size
    if article_group_size == 1:
        # Original behavior: one prompt per article
        prompts = prepare_individual_prompts(articles, tokenizer, prompt_context, enable_thinking)
        article_groups = [[article] for article in articles]  # Each article is its own group
    else:
        # New behavior: group articles to reduce prompt overhead
        prompts, article_groups = prepare_grouped_prompts(articles, tokenizer, prompt_context, article_group_size, enable_thinking)
    
    if verbose:
        print(f"\nDEBUG: Batch processing info:")
        print(f"  Total articles: {len(articles)}")
        print(f"  Article group size: {article_group_size}")
        print(f"  Number of prompts: {len(prompts)}")
        print(f"  Number of article groups: {len(article_groups)}")
        print(f"  First prompt length: {len(prompts[0])}")
        print(f"  First prompt preview (last 500 chars): ...{prompts[0][-500:]}")
    
    
    # Tokenize with dynamic padding (TRUE GPU BATCHING)
    model_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024  # Reduced from 4096 for speed
    )
    
    # Move to GPU only when needed (like fast version)
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
    
    # Save input lengths before generation (one per prompt, not per article)
    input_lengths = [len(model_inputs['input_ids'][i]) for i in range(len(prompts))]
    
    # Generate responses with memory optimization
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024,  # Increased to allow full thinking + response
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
    finally:
        # Immediately clear input tensors from GPU (like fast version)
        del model_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Process each response (which may contain multiple articles)
    relevant_articles = []
    irrelevant_articles = []
    
    for i, article_group in enumerate(article_groups):
        try:
            # Extract only the generated part using saved input lengths
            input_length = input_lengths[i]
            output_ids = generated_ids[i][input_length:].tolist()
            
            # Parse thinking content - works for both thinking enabled/disabled
            if enable_thinking:
                try:
                    # Find </think> token (151668)
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0
                
                thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            else:
                # No thinking mode - all content is the response
                thinking_content = ""
                content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            
            if verbose:
                print(f"\nDEBUG Group {i+1} ({len(article_group)} articles):")
                print(f"  Input length: {input_length}")
                print(f"  Output length: {len(output_ids)}")
                print(f"  Thinking content: {thinking_content[:200]}...")
                print(f"  Response content: {content}")
            
            
            # Parse grouped response - extract relevance for each article
            parsed_relevance = parse_grouped_response(content, len(article_group), verbose)
            
        except Exception as group_error:
            # If parsing fails for this group, mark all articles in group as irrelevant
            error_msg = f"Error processing article group {i+1}: {group_error}"
            if logger:
                logger.error(error_msg)
                logger.warning(f"Marking {len(article_group)} articles from failed group as irrelevant")
            else:
                print(f"⚠️  ERROR: {error_msg}")
                print(f"   Marking {len(article_group)} articles from failed group as irrelevant")
            
            # Mark all articles in this group as irrelevant
            parsed_relevance = [False] * len(article_group)
        
        # SANITY CHECK: Ensure we have decisions for all articles in the group
        if len(parsed_relevance) != len(article_group):
            error_msg = f"Mismatch in group {i+1}: expected {len(article_group)} decisions, got {len(parsed_relevance)}"
            if logger:
                logger.warning(error_msg)
                logger.info(f"Articles in group: {[a['title'][:50] + '...' for a in article_group]}")
                logger.info(f"Parsed decisions: {parsed_relevance}")
                logger.info(f"Response content: {content}")
            elif verbose:
                print(f"⚠️  {error_msg}")
                print(f"  Articles in group: {[a['title'][:50] + '...' for a in article_group]}")
                print(f"  Parsed decisions: {parsed_relevance}")
                print(f"  Response content: {content}")
            # Default missing decisions to irrelevant to maintain data integrity
            while len(parsed_relevance) < len(article_group):
                parsed_relevance.append(False)
                if logger:
                    logger.info("Defaulting missing decision to IRRELEVANT")
                elif verbose:
                    print(f"  -> Defaulting missing decision to IRRELEVANT")
        
        # Assign articles based on parsed relevance
        group_relevant_count = 0
        group_irrelevant_count = 0
        
        for j, article in enumerate(article_group):
            if j < len(parsed_relevance) and parsed_relevance[j]:
                relevant_articles.append(article)
                group_relevant_count += 1
                if verbose:
                    print(f"  -> Article {j+1}: RELEVANT")
            else:
                irrelevant_articles.append(article)
                group_irrelevant_count += 1
                if verbose:
                    print(f"  -> Article {j+1}: IRRELEVANT")
        
        # SANITY CHECK: Verify group processing integrity
        total_decisions_made = group_relevant_count + group_irrelevant_count
        if total_decisions_made != len(article_group):
            error_msg = f"Group {i+1} decision count mismatch: {total_decisions_made} decisions for {len(article_group)} articles"
            if logger:
                logger.error(error_msg)
                logger.warning("Continuing processing despite integrity issue")
            else:
                print(f"⚠️  ERROR: {error_msg}")
                print("   Continuing processing despite integrity issue")
            # Continue processing instead of failing
        
        if verbose:
            print(f"  ✅ Group {i+1} integrity check passed: {len(article_group)} articles → {group_relevant_count} relevant, {group_irrelevant_count} irrelevant")
    
    # Clear generation results from GPU
    del generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # FINAL BATCH SANITY CHECKS
    total_output_articles = len(relevant_articles) + len(irrelevant_articles)
    total_input_articles = len(articles)
    
    if total_output_articles != total_input_articles:
        error_msg = f"Input/output article count mismatch: {total_input_articles} in → {total_output_articles} out"
        details = f"Relevant: {len(relevant_articles)}, Irrelevant: {len(irrelevant_articles)}, Expected total: {total_input_articles}"
        if logger:
            logger.error(f"BATCH ERROR: {error_msg}")
            logger.error(details)
        else:
            print(f"⚠️  BATCH ERROR: {error_msg}")
            print(f"   {details}")
        # Continue processing instead of raising error
    
    # Create unique identifiers for articles (DOI/arXiv preferred, fallback to title+journal)
    def get_article_id(article):
        if article.get('doi'):
            return f"doi:{article['doi']}"
        elif article.get('arxiv_id'):
            return f"arxiv:{article['arxiv_id']}"
        else:
            # Fallback to title+journal combination
            journal = article.get('journal', 'unknown')
            return f"title:{article['title']}|journal:{journal}"
    
    # Verify no duplicate articles in output using unique identifiers
    relevant_ids = [get_article_id(a) for a in relevant_articles]
    irrelevant_ids = [get_article_id(a) for a in irrelevant_articles]
    all_output_ids = relevant_ids + irrelevant_ids
    
    # Check for duplicates within relevant articles
    if len(relevant_ids) != len(set(relevant_ids)):
        seen = set()
        duplicates = []
        for aid in relevant_ids:
            if aid in seen:
                duplicates.append(aid)
            seen.add(aid)
        if logger:
            logger.warning(f"Duplicate articles in relevant list: {duplicates[:3]}...")
        else:
            print(f"⚠️  WARNING: Duplicate articles in relevant list: {duplicates[:3]}...")
        # Deduplicate relevant articles by keeping first occurrence
        seen = set()
        dedupe_relevant = []
        for article in relevant_articles:
            aid = get_article_id(article)
            if aid not in seen:
                dedupe_relevant.append(article)
                seen.add(aid)
        relevant_articles = dedupe_relevant
        relevant_ids = [get_article_id(a) for a in relevant_articles]
        if logger:
            logger.info(f"Deduplicated relevant articles: {len(relevant_articles)}")
        else:
            print(f"   Deduplicated relevant articles: {len(relevant_articles)}")
    
    # Check for duplicates within irrelevant articles
    if len(irrelevant_ids) != len(set(irrelevant_ids)):
        seen = set()
        duplicates = []
        for aid in irrelevant_ids:
            if aid in seen:
                duplicates.append(aid)
            seen.add(aid)
        if logger:
            logger.warning(f"Duplicate articles in irrelevant list: {duplicates[:3]}...")
        else:
            print(f"⚠️  WARNING: Duplicate articles in irrelevant list: {duplicates[:3]}...")
        # Deduplicate irrelevant articles by keeping first occurrence
        seen = set()
        dedupe_irrelevant = []
        for article in irrelevant_articles:
            aid = get_article_id(article)
            if aid not in seen:
                dedupe_irrelevant.append(article)
                seen.add(aid)
        irrelevant_articles = dedupe_irrelevant
        irrelevant_ids = [get_article_id(a) for a in irrelevant_articles]
        if logger:
            logger.info(f"Deduplicated irrelevant articles: {len(irrelevant_articles)}")
        else:
            print(f"   Deduplicated irrelevant articles: {len(irrelevant_articles)}")
    
    # Check for articles that appear in both lists (this should never happen)
    overlap = set(relevant_ids) & set(irrelevant_ids)
    if overlap:
        if logger:
            logger.error(f"Articles appear in both relevant and irrelevant lists: {list(overlap)[:3]}...")
        else:
            print(f"⚠️  CRITICAL: Articles appear in both relevant and irrelevant lists: {list(overlap)[:3]}...")
        # Remove from irrelevant list (prefer relevant classification)
        irrelevant_articles = [a for a in irrelevant_articles if get_article_id(a) not in overlap]
        irrelevant_ids = [get_article_id(a) for a in irrelevant_articles]
        if logger:
            logger.info(f"Removed {len(overlap)} overlapping articles from irrelevant list")
        else:
            print(f"   Removed {len(overlap)} overlapping articles from irrelevant list")
    
    # Recalculate total after deduplication
    all_output_ids = relevant_ids + irrelevant_ids
    if len(all_output_ids) != len(set(all_output_ids)):
        # Find the actual duplicates for debugging
        seen = set()
        duplicates = []
        for aid in all_output_ids:
            if aid in seen:
                duplicates.append(aid)
            seen.add(aid)
        error_msg = f"Duplicate articles still detected after deduplication"
        details = f"Total articles: {len(all_output_ids)}, Unique articles: {len(set(all_output_ids))}, Duplicate IDs: {duplicates[:3]}..., Input batch size: {len(articles)}, Article group size: {article_group_size}"
        distribution = f"Relevant articles: {len(relevant_articles)}, Irrelevant articles: {len(irrelevant_articles)}"
        
        if logger:
            logger.error(f"BATCH ERROR: {error_msg}")
            logger.error(details)
            logger.error(distribution)
            if relevant_articles:
                logger.error(f"First relevant ID: {get_article_id(relevant_articles[0])}")
            if irrelevant_articles:
                logger.error(f"First irrelevant ID: {get_article_id(irrelevant_articles[0])}")
            logger.warning("Continuing processing despite duplicate detection failure")
        else:
            print(f"⚠️  BATCH ERROR: {error_msg}")
            print(f"   {details}")
            print(f"   {distribution}")
            if relevant_articles:
                print(f"   First relevant ID: {get_article_id(relevant_articles[0])}")
            if irrelevant_articles:
                print(f"   First irrelevant ID: {get_article_id(irrelevant_articles[0])}")
            print("   Continuing processing despite duplicate detection failure")
        
        # Continue processing instead of raising error
    
    # Verify all input articles are accounted for (accounting for potential deduplication)
    input_ids = set(get_article_id(a) for a in articles)
    output_ids = set(all_output_ids)
    if not output_ids.issubset(input_ids):
        extra_ids = output_ids - input_ids
        error_msg = f"Extra articles in output that weren't in input: {list(extra_ids)[:3]}..."
        if logger:
            logger.error(f"BATCH ERROR: {error_msg}")
            logger.warning("Continuing processing despite extra articles")
        else:
            print(f"⚠️  BATCH ERROR: {error_msg}")
            print("   Continuing processing despite extra articles")
    
    # Check for missing articles (some input articles might be legitimately missing due to deduplication)
    missing_ids = input_ids - output_ids
    if missing_ids:
        info_msg = f"{len(missing_ids)} input articles not found in output (likely due to deduplication or processing errors)"
        if logger:
            logger.info(info_msg)
            if verbose:
                logger.info(f"Missing IDs: {list(missing_ids)[:5]}...")
        else:
            print(f"⚠️  INFO: {info_msg}")
            if verbose:
                print(f"   Missing IDs: {list(missing_ids)[:5]}...")
        
        # Add missing articles as irrelevant to maintain data integrity
        missing_articles = [a for a in articles if get_article_id(a) in missing_ids]
        if missing_articles:
            if logger:
                logger.info(f"Adding {len(missing_articles)} missing articles as irrelevant")
            else:
                print(f"   Adding {len(missing_articles)} missing articles as irrelevant")
            irrelevant_articles.extend(missing_articles)
            irrelevant_ids.extend([get_article_id(a) for a in missing_articles])
            all_output_ids = relevant_ids + irrelevant_ids
    
    if verbose:
        print(f"✅ BATCH INTEGRITY VERIFIED: {total_input_articles} articles → {len(relevant_articles)} relevant, {len(irrelevant_articles)} irrelevant")
    
    return relevant_articles, irrelevant_articles


def main():
    parser = argparse.ArgumentParser(description="Filter articles using GPU batching")
    parser.add_argument("input_file", help="Input search_results JSON file path")
    parser.add_argument("--gpu-batch-size", type=int, default=10,
                       help="Number of articles per GPU batch")
    parser.add_argument("--article-group-size", type=int, default=5,
                       help="Number of articles to group together per prompt (reduces prompt overhead)")
    parser.add_argument("--prompt-file", default="src/prompts/article_filter.txt",
                       help="Prompt context file")
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                       help="Model name to use")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose debug output")
    parser.add_argument("--enable-thinking", action="store_true",
                       help="Enable thinking mode for model responses")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file {args.input_file} does not exist")
        return 1
    
    # Create organized output directory structure
    input_path = Path(args.input_file)
    base_name = input_path.stem
    
    # Create main output directory for this filtering session
    output_base_dir = input_path.parent / f"{base_name}_filtered"
    output_base_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    relevant_batches_dir = output_base_dir / "relevant_batches"
    relevant_batches_dir.mkdir(exist_ok=True)
    
    irrelevant_dir = output_base_dir / "irrelevant"
    irrelevant_dir.mkdir(exist_ok=True)
    
    tracking_dir = output_base_dir / "tracking"
    tracking_dir.mkdir(exist_ok=True)
    
    # Set up logging
    log_file = tracking_dir / "processing_log.txt"
    logger = setup_logging(str(log_file))
    
    # Output files
    output_irrelevant = str(irrelevant_dir / f"{base_name}_irrelevant.jsonl")
    
    # Create tracking file for batch status
    tracking_file = tracking_dir / "batch_status.json"
    
    if not Path(args.prompt_file).exists():
        print(f"Error: Prompt file {args.prompt_file} does not exist")
        return 1
    
    logger.info(f"Starting batch article filtering session")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Model: {args.model}")
    logger.info(f"GPU batch size: {args.gpu_batch_size}, Article group size: {args.article_group_size}")
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",  # Use auto precision instead of fp16
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    print(f"Loading articles from: {args.input_file}")
    articles = load_articles_from_search_results(args.input_file)
    print(f"Total articles loaded: {len(articles)}")
    logger.info(f"Total articles loaded: {len(articles)}")
    
    print(f"Loading prompt context from: {args.prompt_file}")
    prompt_context = load_prompt_context(args.prompt_file)
    
    print(f"Using GPU batch size {args.gpu_batch_size}, article group size {args.article_group_size}")
    
    # Initialize output files 
    save_articles_to_jsonl([], output_irrelevant)
    
    # Initialize batch tracking
    relevant_batch_buffer = []
    batch_number = 1
    output_batch_size = 2000  # Size for output batches (different from GPU batch size)
    
    # Initialize tracking data
    tracking_data = {
        "session_info": {
            "input_file": str(args.input_file),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_name": base_name
        },
        "batches": {},
        "summary": {
            "total_batches": 0,
            "total_relevant": 0,
            "total_irrelevant": 0,
            "completed": False
        }
    }
    
    # Process articles directly using GPU batching
    print(f"\nProcessing {len(articles)} articles...")
    start_time = time.time()
    
    # Process articles in batches with progress tracking
    total_relevant = 0
    total_irrelevant = 0
    
    try:
        # Create progress bar
        with tqdm(total=len(articles), desc="Processing articles", unit="article") as pbar:
            # Process articles in GPU batches - SINGLE LEVEL ONLY
            for batch_start in range(0, len(articles), args.gpu_batch_size):
                batch_articles = articles[batch_start:batch_start + args.gpu_batch_size]
                
                try:
                    # Process this batch with article grouping
                    batch_relevant, batch_irrelevant = process_articles_batch(
                        model, tokenizer, prompt_context, batch_articles, 
                        article_group_size=args.article_group_size,
                        enable_thinking=args.enable_thinking, verbose=args.verbose, logger=logger
                    )
                except Exception as batch_error:
                    # If batch processing fails, mark only this batch as irrelevant and continue
                    error_msg = f"Error processing batch {batch_start//args.gpu_batch_size + 1}: {batch_error}"
                    logger.error(error_msg)
                    logger.warning(f"Marking {len(batch_articles)} articles from failed batch as irrelevant")
                    
                    batch_relevant = []
                    batch_irrelevant = batch_articles  # Mark entire batch as irrelevant
                    
                    # GPU cleanup after batch error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Periodic GPU cleanup after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Write irrelevant articles immediately
                if batch_irrelevant:
                    append_articles_to_jsonl(batch_irrelevant, output_irrelevant)
                
                # Handle relevant articles
                for article in batch_relevant:
                    # Add to current batch buffer
                    relevant_batch_buffer.append(article)
                    
                    # Write to current batch file
                    current_batch_file = relevant_batches_dir / f"{base_name}_relevant_batch_{batch_number:03d}.jsonl"
                    append_articles_to_jsonl([article], str(current_batch_file))
                    
                    # Check if current batch has reached the size limit
                    if len(relevant_batch_buffer) >= output_batch_size:
                        # Finalize current batch in tracking
                        batch_filename = f"{base_name}_relevant_batch_{batch_number:03d}.jsonl"
                        
                        tracking_data["batches"][f"batch_{batch_number:03d}"] = {
                            "filename": batch_filename,
                            "article_count": len(relevant_batch_buffer),
                            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "download_status": "pending"
                        }
                        
                        print(f"📄 Completed batch {batch_number}: {batch_filename} ({len(relevant_batch_buffer)} articles)")
                        
                        # Start new batch
                        batch_number += 1
                        relevant_batch_buffer = []
                        
                        # Update tracking file
                        tracking_data["summary"]["total_batches"] = batch_number - 1
                        update_tracking_file(tracking_file, tracking_data)
                
                # Update counters and progress
                total_relevant += len(batch_relevant)
                total_irrelevant += len(batch_irrelevant)
                
                pbar.update(len(batch_articles))
                pbar.set_postfix({
                    'Relevant': total_relevant,
                    'Irrelevant': total_irrelevant
                })
                
        
        # Finalize any remaining articles as the final batch
        if relevant_batch_buffer:
            batch_filename = f"{base_name}_relevant_batch_{batch_number:03d}.jsonl"
            
            tracking_data["batches"][f"batch_{batch_number:03d}"] = {
                "filename": batch_filename,
                "article_count": len(relevant_batch_buffer),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "download_status": "pending"
            }
            
            print(f"📄 Finalized final batch {batch_number}: {batch_filename} ({len(relevant_batch_buffer)} articles)")
            tracking_data["summary"]["total_batches"] = batch_number
        else:
            tracking_data["summary"]["total_batches"] = max(0, batch_number - 1)
        
        # Finalize tracking data
        tracking_data["summary"]["total_relevant"] = total_relevant
        tracking_data["summary"]["total_irrelevant"] = total_irrelevant
        tracking_data["summary"]["completed"] = True
        tracking_data["session_info"]["completed"] = time.strftime("%Y-%m-%d %H:%M:%S")
        update_tracking_file(tracking_file, tracking_data)
        
    except Exception as e:
        error_msg = f"Critical error during processing: {e}"
        print(f"\n{error_msg}")
        logger.error(error_msg)
        logger.error(f"Exception type: {type(e).__name__}")
        
        # GPU cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Update tracking with partial results and error info
        tracking_data["summary"]["total_relevant"] = total_relevant
        tracking_data["summary"]["total_irrelevant"] = total_irrelevant
        tracking_data["summary"]["completed"] = False  # Mark as incomplete due to critical error
        tracking_data["session_info"]["error"] = str(e)
        tracking_data["session_info"]["note"] = "Processing stopped due to critical error. Partial results available."
        update_tracking_file(tracking_file, tracking_data)
        
        logger.error("Processing stopped due to critical error. Partial results have been saved.")
        return 1
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time/60:.1f} minutes")
    
    print(f"\nProcessing complete!")
    print(f"Total relevant: {total_relevant}")
    print(f"Total irrelevant: {total_irrelevant}")
    print(f"Total processed: {total_relevant + total_irrelevant}")
    print(f"\nOutput structure:")
    print(f"  📁 Main directory: {output_base_dir}")
    print(f"  📁 Relevant batches: {relevant_batches_dir} ({tracking_data['summary']['total_batches']} batches)")
    print(f"  📄 Irrelevant file: {output_irrelevant}")
    print(f"  📊 Tracking file: {tracking_file}")
    
    # Final sanity check - critical for data integrity
    final_total_processed = total_relevant + total_irrelevant
    if final_total_processed != len(articles):
        print(f"\n⚠️  CRITICAL ERROR: Article count mismatch!")
        print(f"   Input articles: {len(articles)}")
        print(f"   Output articles: {final_total_processed}")
        print(f"   Missing articles: {len(articles) - final_total_processed}")
        print(f"   This indicates articles were lost during processing!")
        return 1
    else:
        print(f"✅ Sanity check passed: All {len(articles)} articles accounted for")
    
    return 0


if __name__ == "__main__":
    exit(main())