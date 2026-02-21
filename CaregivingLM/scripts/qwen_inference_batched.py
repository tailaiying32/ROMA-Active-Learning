#!/usr/bin/env python3
"""
Batched Qwen3 Inference for Article Relevance Filtering
Optimized for GPU efficiency using native HuggingFace batching.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.cuda
import gc
import json
import random
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class BatchConfig:
    """Configuration for batching parameters"""
    batch_size: int = 6  # Start conservative, can be increased
    max_length: int = 10000  # Should be sufficient for title + journal + prompt
    dynamic_padding: bool = True
    verbose: bool = True

class BatchedQwenInference:
    """Optimized batched inference for article relevance filtering"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B", config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.model_name = model_name
        
        print(f"Loading model: {model_name}")
        print(f"Batch configuration: size={self.config.batch_size}, max_length={self.config.max_length}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token and configure for decoder-only models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set left padding for decoder-only models (critical for batching)
        self.tokenizer.padding_side = "left"
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 to save memory
            device_map="auto",
            low_cpu_mem_usage=True  # Minimize CPU memory during loading
        )
        
        # Load prompt context
        self.prompt_context = self._load_prompt_context()
        
        # Initial GPU memory cleanup
        self._clear_gpu_memory()
        print("Model loaded successfully!")
        print(f"GPU memory after loading: {self._get_gpu_memory_info()}")
    
    def _clear_gpu_memory(self):
        """Aggressively clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def _get_gpu_memory_info(self) -> str:
        """Get current GPU memory usage info"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
            return f"Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB"
        return "No GPU available"
    
    def _load_prompt_context(self, file_path: str = "src/prompts/article_filter.txt") -> str:
        """Load prompt context from file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    
    def create_individual_prompt(self, article: Dict) -> str:
        """Create a prompt for a single article"""
        article_data = f"\n\nHere is the research article to evaluate:\n\n"
        article_data += f"1. Title: {article['title']}\n"
        if article.get('journal'):
            article_data += f"   Journal: {article['journal']}\n"
        article_data += "\n"
        
        return self.prompt_context + article_data
    
    def prepare_batch_inputs(self, articles: List[Dict]) -> List[str]:
        """Prepare individual prompts for each article in the batch"""
        prompts = []
        for article in articles:
            prompt = self.create_individual_prompt(article)
            
            # Create chat message
            messages = [{"role": "user", "content": prompt}]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompts.append(text)
        
        return prompts
    
    def process_batch(self, articles: List[Dict]) -> List[Dict]:
        """Process a batch of articles and return results"""
        if len(articles) == 0:
            return []
            
        # Prepare inputs
        prompts = self.prepare_batch_inputs(articles)
        
        # Tokenize with dynamic padding
        model_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=self.config.dynamic_padding,
            truncation=True,
            max_length=self.config.max_length
        )
        
        # Move to GPU only when needed
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
        
        if self.config.verbose:
            print(f"  Processing batch of {len(articles)} articles...")
            print(f"  Input shape: {model_inputs['input_ids'].shape}")
            print(f"  GPU memory before generation: {self._get_gpu_memory_info()}")
        
        # Save input lengths before generation (needed for output processing)
        input_lengths = [len(model_inputs['input_ids'][i]) for i in range(len(articles))]
        
        # Generate responses with memory optimization
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=256,  # Reduced further for memory
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache for efficiency
                )
        finally:
            # Immediately clear input tensors from GPU
            del model_inputs
            self._clear_gpu_memory()
        
        # Process outputs
        results = []
        for i, (article, prompt) in enumerate(zip(articles, prompts)):
            # Extract only the generated part using saved input lengths
            input_length = input_lengths[i]
            output_ids = generated_ids[i][input_length:].tolist()
            
            # Parse thinking content (similar to original)
            try:
                # Find </think> token (151668)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            
            results.append({
                'article': article,
                'thinking': thinking_content,
                'response': content,
                'prompt_length': len(prompt)
            })
        
        # Clear generation results from GPU
        del generated_ids
        self._clear_gpu_memory()
        
        if self.config.verbose:
            print(f"  GPU memory after processing: {self._get_gpu_memory_info()}")
        
        return results
    
    def process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Process all articles using batching"""
        all_results = []
        total_batches = (len(articles) + self.config.batch_size - 1) // self.config.batch_size
        
        print(f"\nProcessing {len(articles)} articles in {total_batches} batches...")
        
        start_time = time.time()
        
        for batch_idx in range(0, len(articles), self.config.batch_size):
            batch_articles = articles[batch_idx:batch_idx + self.config.batch_size]
            batch_num = (batch_idx // self.config.batch_size) + 1
            
            if self.config.verbose:
                print(f"\nBatch {batch_num}/{total_batches}:")
            
            batch_start = time.time()
            batch_results = self.process_batch(batch_articles)
            batch_time = time.time() - batch_start
            
            if self.config.verbose:
                print(f"  Completed in {batch_time:.2f}s ({len(batch_articles)/batch_time:.1f} articles/sec)")
            
            all_results.extend(batch_results)
            
            # Clear memory between batches
            self._clear_gpu_memory()
        
        total_time = time.time() - start_time
        print(f"\nCompleted all batches in {total_time:.2f}s")
        print(f"Average speed: {len(articles)/total_time:.1f} articles/sec")
        print(f"Final GPU memory: {self._get_gpu_memory_info()}")
        
        return all_results

def load_articles_from_search_results(file_path: str) -> List[Dict]:
    """Load articles from search_results JSON file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    articles = []
    for db_name, db_results in data['results'].items():
        for article in db_results:
            if article.get('doi') or article.get('arxiv_id'):
                articles.append(article)
    
    return articles

def select_random_articles(articles: List[Dict], n: int) -> List[Dict]:
    """Randomly select n articles from the list"""
    return random.sample(articles, min(n, len(articles)))

def print_results(results: List[Dict], total_time: float):
    """Print results in a readable format"""
    print("\n" + "="*80)
    print("BATCHED ANALYSIS RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        article = result['article']
        print(f"\n{i}. {article['title']}")
        if article.get('journal'):
            print(f"   Journal: {article['journal']}")
        if article.get('date'):
            print(f"   Date: {article['date']}")
        
        print(f"\n   Model response:")
        print(f"   {'-'*40}")
        print(f"   {result['response']}")
    
    # Print timing summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Total articles processed: {len(results)}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per article: {total_time/len(results):.2f} seconds")
    print(f"Processing rate: {len(results)/total_time:.1f} articles/second")

def main():
    """Main execution function"""
    # Configuration
    SEARCH_RESULTS_FILE = "/home/ziang/Workspace/CaregivingLM/scraped_articles/metadata/search_results_20250718_082402.json"
    NUM_ARTICLES = 100  # Increased since batching is more efficient
    
    # Batch configuration - adjust based on your GPU memory
    config = BatchConfig(
        batch_size=100,  # Start conservative
        max_length=1024,
        dynamic_padding=True,
        verbose=True
    )
    
    print("="*80)
    print("BATCHED QWEN3 ARTICLE RELEVANCE FILTERING")
    print("="*80)
    
    # Load articles
    print("\nLoading articles from search results file...")
    all_articles = load_articles_from_search_results(SEARCH_RESULTS_FILE)
    print(f"Total articles loaded: {len(all_articles)}")
    
    # Select articles for testing
    selected_articles = select_random_articles(all_articles, NUM_ARTICLES)
    print(f"Randomly selected {len(selected_articles)} articles for testing")
    
    # Initialize batched inference
    inferencer = BatchedQwenInference(config=config)
    
    # Process articles and track total time
    start_time = time.time()
    results = inferencer.process_articles(selected_articles)
    total_time = time.time() - start_time
    
    # Print results with timing
    print_results(results, total_time)
    
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()