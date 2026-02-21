#!/usr/bin/env python3
"""
Comprehensive Inference Benchmark

Compares local Qwen model vs OpenAI API models for article filtering.
Measures pure inference time excluding setup.
"""

import os
import json
import random
import time
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

def load_articles_from_search_results(file_path):
    """Load articles from search_results JSON file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    articles = []
    for db_name, db_results in data['results'].items():
        for article in db_results:
            if article.get('doi') or article.get('arxiv_id'):
                articles.append(article)
    
    return articles

def select_random_articles(articles, n):
    """Randomly select n articles from the list"""
    return random.sample(articles, min(n, len(articles)))

def create_prompt_data(selected_articles):
    """Create prompt data from selected articles"""
    prompt_data = "\n\nHere are the research article titles to evaluate:\n\n"
    for i, article in enumerate(selected_articles, 1):
        prompt_data += f"{i}. Title: {article['title']}\n"
        if article.get('journal'):
            prompt_data += f"   Journal: {article['journal']}\n"
        prompt_data += "\n"
    return prompt_data

def load_prompt_context(file_path):
    """Load prompt context from a text file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def benchmark_openai_model(prompt: str, model: str) -> Tuple[str, float, bool]:
    """Benchmark OpenAI model inference time"""
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.1
        )
        
        inference_time = time.time() - start_time
        content = response.choices[0].message.content
        
        return content, inference_time, True
        
    except Exception as e:
        inference_time = time.time() - start_time
        return f"Error: {str(e)}", inference_time, False

def benchmark_qwen_model(prompt: str, model, tokenizer) -> Tuple[str, float, bool]:
    """Benchmark local Qwen model inference time"""
    try:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Start timing after tokenization (pure inference)
        start_time = time.time()
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4000
        )
        
        inference_time = time.time() - start_time
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)  # </think>
        except ValueError:
            index = 0
        
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        return content, inference_time, True
        
    except Exception as e:
        return f"Error: {str(e)}", 0.0, False

def run_benchmark_comparison(articles: List[Dict], prompt_context: str, num_runs: int = 2, articles_per_run: int = 100):
    """Run comprehensive benchmark comparing all models"""
    
    print(f"🔬 COMPREHENSIVE INFERENCE BENCHMARK")
    print(f"Testing {num_runs} runs with {articles_per_run} articles each")
    print("="*80)
    
    # Test configurations
    test_configs = [
        ("gpt-4o-mini", "openai"),
        ("gpt-4o", "openai"), 
        ("gpt-4", "openai"),
        ("qwen-local", "qwen")
    ]
    
    results = {}
    
    for model_name, model_type in test_configs:
        print(f"\n📊 Testing {model_name.upper()}")
        print("-" * 50)
        
        if model_type == "qwen":
            # Load Qwen model (one-time setup, not counted in benchmark)
            print("Loading Qwen model (setup time not counted)...")
            setup_start = time.time()
            
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
                model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen3-1.7B",
                    torch_dtype="auto",
                    device_map="auto"
                )
                setup_time = time.time() - setup_start
                print(f"Model loaded in {setup_time:.1f}s (not counted in benchmark)")
                
            except Exception as e:
                print(f"❌ Failed to load Qwen model: {e}")
                continue
        
        times = []
        successful_runs = 0
        
        for run in range(1, num_runs + 1):
            print(f"  Run {run}/{num_runs}: ", end="")
            
            # Select random articles
            selected_articles = select_random_articles(articles, articles_per_run)
            prompt_data = create_prompt_data(selected_articles)
            full_prompt = prompt_context + prompt_data
            
            # Run inference based on model type
            if model_type == "openai":
                response, inference_time, success = benchmark_openai_model(full_prompt, model_name)
            else:  # qwen
                response, inference_time, success = benchmark_qwen_model(full_prompt, model, tokenizer)
            
            if success:
                times.append(inference_time)
                successful_runs += 1
                print(f"✅ {inference_time:.2f}s")
            else:
                print(f"❌ Failed ({inference_time:.2f}s)")
                print(f"     Error: {response}")
        
        # Calculate statistics
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            time_per_article = avg_time / articles_per_run
            
            results[model_name] = {
                'type': model_type,
                'successful_runs': successful_runs,
                'total_runs': num_runs,
                'articles_per_run': articles_per_run,
                'average_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'time_per_article': time_per_article,
                'all_times': times
            }
            
            print(f"  Results: avg={avg_time:.2f}s, per_article={time_per_article:.3f}s")
        else:
            print(f"  ❌ No successful runs")
    
    # Display comparison
    if results:
        print(f"\n" + "="*80)
        print("📈 PERFORMANCE COMPARISON")
        print("="*80)
        
        print(f"{'Model':<15} {'Type':<8} {'Avg Time':<10} {'Per Article':<12} {'Success':<10}")
        print("-" * 70)
        
        # Sort by time per article
        sorted_results = sorted(results.items(), key=lambda x: x[1]['time_per_article'])
        
        for model_name, data in sorted_results:
            success_rate = f"{data['successful_runs']}/{data['total_runs']}"
            print(f"{model_name:<15} {data['type']:<8} {data['average_time']:<10.2f}s {data['time_per_article']:<12.3f}s {success_rate:<10}")
        
        # Speed comparison
        print(f"\n🏆 SPEED RANKING:")
        for i, (model_name, data) in enumerate(sorted_results, 1):
            speedup = sorted_results[-1][1]['time_per_article'] / data['time_per_article']
            print(f"{i}. {model_name}: {data['time_per_article']:.3f}s per article ({speedup:.1f}x faster than slowest)")
        
        # Time estimates for large datasets
        print(f"\n⏱️  TIME ESTIMATES FOR 10,000 ARTICLES")
        print("="*50)
        
        for model_name, data in sorted_results:
            total_time_seconds = 10000 * data['time_per_article']
            hours = total_time_seconds / 3600
            days = hours / 24
            
            if days >= 1:
                time_str = f"{days:.1f} days"
            elif hours >= 1:
                time_str = f"{hours:.1f} hours"
            else:
                time_str = f"{total_time_seconds/60:.1f} minutes"
            
            print(f"{model_name:<15}: {time_str}")
        
        # Cost analysis for OpenAI models (rough estimates)
        print(f"\n💰 ROUGH COST ESTIMATES FOR 10,000 ARTICLES")
        print("="*50)
        
        # Approximate pricing (as of early 2024, may change)
        pricing = {
            "gpt-4o-mini": {"input": 0.15/1000, "output": 0.6/1000},    # per 1000 tokens
            "gpt-4o": {"input": 5/1000, "output": 15/1000},
            "gpt-4": {"input": 30/1000, "output": 60/1000}
        }
        
        avg_input_tokens = 1000  # Rough estimate for prompt + articles
        avg_output_tokens = 500  # Rough estimate for response
        
        for model_name, data in results.items():
            if data['type'] == 'openai' and model_name in pricing:
                input_cost = (avg_input_tokens * 10000 / 1000) * pricing[model_name]["input"]
                output_cost = (avg_output_tokens * 10000 / 1000) * pricing[model_name]["output"]
                total_cost = input_cost + output_cost
                print(f"{model_name:<15}: ~${total_cost:.2f}")
            elif data['type'] == 'qwen':
                print(f"{model_name:<15}: $0 (local)")
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"outputs/inference_benchmark_{timestamp}.json"
        os.makedirs("outputs", exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        return results
    
    else:
        print("\n❌ No successful benchmark results")
        return None

def main():
    # Configuration
    SEARCH_RESULTS_FILE = "/home/ziang/Workspace/CaregivingLM/scraped_articles/metadata/search_results_20250718_082402.json"
    PROMPT_CONTEXT_FILE = "src/prompts/article_filter.txt"
    
    print("Loading test data...")
    all_articles = load_articles_from_search_results(SEARCH_RESULTS_FILE)
    prompt_context = load_prompt_context(PROMPT_CONTEXT_FILE)
    
    print(f"Loaded {len(all_articles)} articles")
    print(f"Prompt context: {len(prompt_context)} characters")
    
    # Run comprehensive benchmark
    results = run_benchmark_comparison(
        all_articles, 
        prompt_context, 
        num_runs=2,      # Number of test runs per model
        articles_per_run=100   # Articles per test run
    )
    
    if results:
        print(f"\n✅ Benchmark complete! Check outputs/ for detailed results.")
    else:
        print(f"\n❌ Benchmark failed!")

if __name__ == "__main__":
    main()