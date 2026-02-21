import os
import json
import random
import time
from typing import List, Dict
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

def load_articles_from_search_results(file_path):
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

def run_openai_inference(prompt: str, model: str = "gpt-4") -> tuple:
    """Run inference using OpenAI API and return response with timing"""
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Measure only the API call time
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.1  # Low temperature for consistent results
        )
        
        inference_time = time.time() - start_time
        
        content = response.choices[0].message.content
        
        return content, inference_time, True
        
    except Exception as e:
        inference_time = time.time() - start_time
        return f"Error: {str(e)}", inference_time, False

def benchmark_multiple_runs(articles: List[Dict], prompt_context: str, num_runs: int = 3, articles_per_run: int = 100, model: str = "gpt-4"):
    """Run multiple inference tests and calculate benchmark statistics"""
    
    print(f"\n🔬 BENCHMARKING OpenAI {model.upper()}")
    print(f"Running {num_runs} inference tests with {articles_per_run} articles each")
    print("="*60)
    
    times = []
    successful_runs = 0
    
    for run in range(1, num_runs + 1):
        print(f"\nRun {run}/{num_runs}:")
        
        # Select random articles for this run
        selected_articles = select_random_articles(articles, articles_per_run)
        prompt_data = create_prompt_data(selected_articles)
        full_prompt = prompt_context + prompt_data
        
        print(f"  Selected {len(selected_articles)} articles")
        print(f"  Prompt length: {len(full_prompt)} characters")
        print(f"  Running inference...", end=" ")
        
        # Run inference
        response, inference_time, success = run_openai_inference(full_prompt, model)
        
        if success:
            times.append(inference_time)
            successful_runs += 1
            print(f"✅ {inference_time:.2f}s")
            
            # Show sample of response
            if len(response) > 200:
                sample_response = response[:200] + "..."
            else:
                sample_response = response
            print(f"  Sample response: {sample_response}")
            
        else:
            print(f"❌ Failed ({inference_time:.2f}s)")
            print(f"  Error: {response}")
    
    # Calculate statistics
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 BENCHMARK RESULTS")
        print("="*40)
        print(f"Model: {model}")
        print(f"Successful runs: {successful_runs}/{num_runs}")
        print(f"Articles per run: {articles_per_run}")
        print(f"Average time: {avg_time:.2f}s")
        print(f"Fastest time: {min_time:.2f}s")
        print(f"Slowest time: {max_time:.2f}s")
        print(f"Time per article: {avg_time/articles_per_run:.3f}s")
        
        # Estimate time for large datasets
        print(f"\n⏱️  TIME ESTIMATES FOR LARGE DATASETS")
        print("="*45)
        for dataset_size in [1000, 5000, 10000, 50000]:
            estimated_time_seconds = (dataset_size / articles_per_run) * avg_time
            estimated_hours = estimated_time_seconds / 3600
            estimated_days = estimated_hours / 24
            
            if estimated_hours < 1:
                time_str = f"{estimated_time_seconds/60:.1f} minutes"
            elif estimated_days < 1:
                time_str = f"{estimated_hours:.1f} hours"
            else:
                time_str = f"{estimated_days:.1f} days"
            
            print(f"{dataset_size:,} articles: {time_str}")
        
        return {
            'model': model,
            'successful_runs': successful_runs,
            'total_runs': num_runs,
            'articles_per_run': articles_per_run,
            'average_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'time_per_article': avg_time/articles_per_run,
            'all_times': times
        }
    else:
        print(f"\n❌ No successful runs completed")
        return None

def main():
    # Configuration
    SEARCH_RESULTS_FILE = "/home/ziang/Workspace/CaregivingLM/scraped_articles/metadata/search_results_20250718_082402.json"
    PROMPT_CONTEXT_FILE = "src/prompts/article_filter.txt"
    
    # Load data
    print("Loading articles from search results file...")
    all_articles = load_articles_from_search_results(SEARCH_RESULTS_FILE)
    print(f"Total articles loaded: {len(all_articles)}")
    
    print("Loading prompt context...")
    prompt_context = load_prompt_context(PROMPT_CONTEXT_FILE)
    print(f"Prompt context loaded: {len(prompt_context)} characters")
    
    # Test different models
    models_to_test = [
        ("gpt-4o-mini", 3, 100),    # Fast, cheap model
        ("gpt-4o", 2, 100),        # Latest GPT-4 model  
        ("gpt-4", 2, 100),         # Standard GPT-4
    ]
    
    all_results = {}
    
    for model, num_runs, articles_per_run in models_to_test:
        print(f"\n" + "="*80)
        print(f"TESTING {model.upper()}")
        print("="*80)
        
        try:
            results = benchmark_multiple_runs(
                all_articles, 
                prompt_context, 
                num_runs=num_runs,
                articles_per_run=articles_per_run,
                model=model
            )
            
            if results:
                all_results[model] = results
                
        except Exception as e:
            print(f"❌ Error testing {model}: {e}")
    
    # Final comparison
    if all_results:
        print(f"\n" + "="*80)
        print("📈 FINAL COMPARISON")
        print("="*80)
        
        print(f"{'Model':<15} {'Avg Time':<10} {'Per Article':<12} {'Success Rate':<12}")
        print("-" * 60)
        
        for model, results in all_results.items():
            success_rate = f"{results['successful_runs']}/{results['total_runs']}"
            print(f"{model:<15} {results['average_time']:<10.2f}s {results['time_per_article']:<12.3f}s {success_rate:<12}")
        
        # Find fastest model
        fastest_model = min(all_results.items(), key=lambda x: x[1]['time_per_article'])
        print(f"\n🏆 Fastest model: {fastest_model[0]} ({fastest_model[1]['time_per_article']:.3f}s per article)")
        
        # Save detailed results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"outputs/openai_benchmark_{timestamp}.json"
        os.makedirs("outputs", exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
    
    else:
        print("\n❌ No successful benchmark results")

if __name__ == "__main__":
    main()