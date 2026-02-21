from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import os
from openai import OpenAI
from dotenv import load_dotenv

from ..vector_store.chroma_store import ChromaVectorStore
from ..document_processor.base import DocumentMetadata

def parse_functionality_response(cache_file_path: str) -> str:
    """
    Parse functionality response from a cache file.
    
    Args:
        cache_file_path: Path to the cache file containing functionality response
        
    Returns:
        Functionality response as string
    """
    try:
        with open(cache_file_path, 'r') as f:
            cache_data = json.load(f)
        
        # Extract the response from the cache
        response = cache_data.get('response', '')
        
        # If the response is a JSON string, try to parse it
        try:
            response_json = json.loads(response)
            # Return the full JSON as a formatted string
            return json.dumps(response_json, indent=2)
        except json.JSONDecodeError:
            # If it's not valid JSON, return as is
            return response
            
    except Exception as e:
        raise ValueError(f"Error parsing cache file {cache_file_path}: {str(e)}")

def load_task_description(task_name: str) -> str:
    """
    Load task description from the tasks directory.
    
    Args:
        task_name: Name of the task (e.g., "dressing", "feeding", "hygiene")
        
    Returns:
        Task description as string
    """
    task_file = Path(__file__).parent.parent / "prompts" / "tasks" / f"{task_name}.txt"
    
    if not task_file.exists():
        available_tasks = [f.stem for f in (Path(__file__).parent.parent / "prompts" / "tasks").glob("*.txt")]
        raise ValueError(f"Task '{task_name}' not found. Available tasks: {', '.join(available_tasks)}")
    
    return task_file.read_text()

class RAGPipeline:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        prompt_template_path: str,
        cache_dir: str,
        openai_model: str = "gpt-4-turbo-preview"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: Vector store instance
            prompt_template_path: Path to prompt template file
            cache_dir: Directory to store interaction cache
            openai_model: OpenAI model to use
        """
        load_dotenv()
        self.vector_store = vector_store
        self.prompt_template_path = Path(prompt_template_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.openai_model = openai_model
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load prompt template from file."""
        return self.prompt_template_path.read_text()
    
    def _format_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        prompt_type: str = "default",
        functionality_input: str = None,
        task_description: str = None
    ) -> str:
        """
        Format prompt with query and retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved documents with metadata
            prompt_type: Type of prompt to use ("default", "functionality", or "robot")
            functionality_input: Functional assessment input for robot mode
            task_description: Task description for robot mode
            
        Returns:
            Formatted prompt
        """
        # Format retrieved documents
        context = []
        for doc in retrieved_docs:
            metadata = doc["metadata"]
            context.append(
                f"Source: {metadata['title']} ({metadata['source_path']})\n"
                f"Content: {doc['document']}\n"
            )
        
        context_text = "\n".join(context)
        
        # Format prompt based on type
        if prompt_type == "functionality":
            return self.prompt_template.format(
                user_description=query,
                context=context_text
            )
        elif prompt_type == "robot":
            return self.prompt_template.format(
                functionality_input=functionality_input,
                task_description=task_description,
                context=context_text
            )
        else:
            # Default prompt format
            return self.prompt_template.format(
                query=query,
                context=context_text
            )
    
    def _cache_interaction(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        prompt: str,
        response: str
    ) -> str:
        """
        Cache interaction details.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            prompt: Formatted prompt
            response: Model response
            
        Returns:
            Path to cache file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_data = {
            "timestamp": timestamp,
            "query": query,
            "retrieved_docs": retrieved_docs,
            "prompt": prompt,
            "response": response
        }
        
        cache_path = self.cache_dir / f"interaction_{timestamp}.json"
        cache_path.write_text(json.dumps(cache_data, indent=2))
        
        return str(cache_path)
    
    def _save_response_to_file(
        self,
        query: str,
        response: str,
        prompt_type: str = "default"
    ) -> str:
        """
        Save response to a local text file.
        
        Args:
            query: User query
            response: Model response
            prompt_type: Type of prompt used
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        filename = f"{prompt_type}_response_{timestamp}.txt"
        output_path = output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Query: {query}\n")
            f.write(f"Prompt Type: {prompt_type}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("-" * 80 + "\n")
            f.write("Response:\n")
            f.write(response)
        
        return str(output_path)
    
    def query(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        prompt_type: str = "default",
        functionality_input: str = None,
        task_description: str = None
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User query
            n_results: Number of documents to retrieve
            where: Optional filter conditions
            prompt_type: Type of prompt to use ("default", "functionality", or "robot")
            functionality_input: Functional assessment input for robot mode
            task_description: Task description for robot mode
            
        Returns:
            Dictionary containing response and metadata
        """
        # Retrieve relevant documents
        search_results = self.vector_store.search(
            query=query,
            n_results=n_results,
            where=where
        )
        
        # Format retrieved documents
        retrieved_docs = [
            {
                "document": doc,
                "metadata": meta
            }
            for doc, meta in zip(
                search_results["documents"],
                search_results["metadatas"]
            )
        ]
        
        # Format prompt
        prompt = self._format_prompt(
            query, 
            retrieved_docs, 
            prompt_type, 
            functionality_input, 
            task_description
        )
        
        # Get response from OpenAI using new API
        response = self.client.responses.create(
            model=self.openai_model,
            instructions="You are a knowledgeable assistant specializing in caregiving and occupational therapy. You have access to relevant documents that may help answer the user's question. Please use the provided context to give a comprehensive and accurate response.",
            input=prompt
        ).output_text
        
        # Cache interaction
        cache_path = self._cache_interaction(
            query=query,
            retrieved_docs=retrieved_docs,
            prompt=prompt,
            response=response
        )
        
        # Save response to file
        output_path = self._save_response_to_file(
            query=query,
            response=response,
            prompt_type=prompt_type
        )
        
        return {
            "response": response,
            "retrieved_docs": retrieved_docs,
            "cache_path": cache_path,
            "output_path": output_path
        } 