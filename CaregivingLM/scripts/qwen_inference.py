from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random

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
        if article.get('abstract'):
            prompt_data += f"   Abstract: {article['abstract'][:200]}...\n"  # Truncate abstract for brevity
        prompt_data += "\n"
    return prompt_data

model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Configuration
SEARCH_RESULTS_FILE = "/home/ziang/Workspace/CaregivingLM/scraped_articles/metadata/search_results_20250718_082402.json"
NUM_ARTICLES = 10  # Number of articles to randomly select for testing

# Load and select articles
print("Loading articles from search results file...")
all_articles = load_articles_from_search_results(SEARCH_RESULTS_FILE)
print(f"Total articles loaded: {len(all_articles)}")

selected_articles = select_random_articles(all_articles, NUM_ARTICLES)
print(f"Randomly selected {len(selected_articles)} articles for testing")

# Load prompt context from file
def load_prompt_context(file_path):
    """Load prompt context from a text file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

# prepare the model input
PROMPT_CONTEXT_FILE = "src/prompts/article_filter.txt"
prompt_context = load_prompt_context(PROMPT_CONTEXT_FILE)

prompt_data = create_prompt_data(selected_articles)

prompt = prompt_context + prompt_data

messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# Print results for examination
print("\n" + "="*80)
print("ANALYSIS RESULTS")
print("="*80)

print(f"\nSelected Articles ({len(selected_articles)} total):")
for i, article in enumerate(selected_articles, 1):
    print(f"\n{i}. {article['title']}")
    if article.get('journal'):
        print(f"   Journal: {article['journal']}")
    if article.get('date'):
        print(f"   Date: {article['date']}")

print(f"\nPrompt sent to model:")
print("-" * 40)
print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

print(f"\nModel thinking process:")
print("-" * 40)
print(thinking_content)

print(f"\nModel response:")
print("-" * 40)
print(content)

print("\n" + "="*80)
