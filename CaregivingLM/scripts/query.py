#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
from typing import Optional, Dict, Any
import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.vector_store.chroma_store import ChromaVectorStore
from src.rag.pipeline import RAGPipeline, parse_functionality_response, load_task_description

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def query_pipeline(
    query: str,
    config: Dict[str, Any],
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    mode: str = "default",
    functionality_cache: str = None,
    task_name: str = None
) -> Dict[str, Any]:
    """
    Query the RAG pipeline.
    
    Args:
        query: User query
        config: Configuration dictionary
        n_results: Number of results to retrieve
        where: Optional filter conditions
        debug: Whether to run in debug mode
        mode: Query mode ("default", "functionality", or "robot")
        functionality_cache: Path to functionality cache file for robot mode
        task_name: Name of the task for robot mode
        
    Returns:
        Query results
    """
    # Initialize vector store
    vector_store = ChromaVectorStore(
        persist_directory=config["vector_store_dir"],
        collection_name=config["collection_name"]
    )
    
    if debug:
        print("\n=== Debug Mode ===")
        print(f"Query Mode: {mode}")
        if mode == "robot":
            print(f"Functionality Cache: {functionality_cache}")
            print(f"Task Name: {task_name}")
        print("\nVector Store Information:")
        vector_store.list_collections()
        
        print("\nDebug Search Results:")
        vector_store.debug_search(query, k=n_results)
        print("\n" + "="*50 + "\n")
    
    # Select prompt template based on mode
    if mode == "functionality":
        prompt_template_path = "src/prompts/functionality_prompt.txt"
    elif mode == "robot":
        prompt_template_path = "src/prompts/robot_adaptation.txt"
    else:
        prompt_template_path = config["prompt_template_path"]
    
    # Initialize RAG pipeline
    pipeline = RAGPipeline(
        vector_store=vector_store,
        prompt_template_path=prompt_template_path,
        cache_dir=config["cache_dir"],
        openai_model=config["openai_model"]
    )
    
    # Prepare additional parameters for robot mode
    functionality_input = None
    task_description = None
    
    if mode == "robot":
        if not functionality_cache:
            raise ValueError("Functionality cache file path is required for robot mode")
        if not task_name:
            raise ValueError("Task name is required for robot mode")
        
        # Parse functionality response from cache
        functionality_input = parse_functionality_response(functionality_cache)
        
        # Load task description
        task_description = load_task_description(task_name)
    
    # Process query
    return pipeline.query(
        query=query,
        n_results=n_results,
        where=where,
        prompt_type=mode,
        functionality_input=functionality_input,
        task_description=task_description
    )

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument("query", help="Query string")
    parser.add_argument("--config", default="config/config.json", help="Path to config file")
    parser.add_argument("--n_results", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--mode", choices=["default", "functionality", "robot"], default="default", help="Query mode")
    parser.add_argument("--functionality_cache", help="Path to functionality cache file for robot mode")
    parser.add_argument("--task_name", help="Name of the task for robot mode")
    
    args = parser.parse_args()
    
    # Validate robot mode arguments
    if args.mode == "robot":
        if not args.functionality_cache:
            print("Error: --functionality_cache is required for robot mode")
            print("Example: --functionality_cache data/cache/interaction_20231201_143022.json")
            sys.exit(1)
        if not args.task_name:
            print("Error: --task_name is required for robot mode")
            print("Available tasks: dressing, feeding, hygiene, toileting, transferring")
            sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Process query
    results = query_pipeline(
        query=args.query,
        config=config,
        n_results=args.n_results,
        debug=args.debug,
        mode=args.mode,
        functionality_cache=args.functionality_cache,
        task_name=args.task_name
    )
    
    # Print results based on mode
    print(f"\nQuery Mode: {args.mode}")
    print("=" * 80)
    
    if args.mode == "functionality":
        print("\nFunctional Assessment Response:")
        print("-" * 80)
        print(results["response"])
        print("\nNote: This response has been saved to a structured format for simulation model initialization.")
    elif args.mode == "robot":
        print("\nRobot Adaptation Response:")
        print("-" * 80)
        print(results["response"])
        print("\nNote: This response has been saved to a structured format for robotic implementation.")
    else:
        print("\nResponse:")
        print("-" * 80)
        print(results["response"])
    
    print("\nSources:")
    print("-" * 80)
    for doc in results["retrieved_docs"]:
        metadata = doc["metadata"]
        print(f"- {metadata.get('title', 'Unknown')} ({metadata.get('source_path', 'Unknown')})")
    
    print(f"\nResponse saved to: {results['output_path']}")
    print(f"Interaction cached at: {results['cache_path']}")

if __name__ == "__main__":
    main() 