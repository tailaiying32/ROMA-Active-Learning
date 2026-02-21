from typing import List, Dict, Any
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from ..config import DEFAULT_EMBEDDING_MODEL
import shutil
import numpy as np
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "caregiving_documents_openai"
    ):
        """
        Initialize Chroma vector store.
        
        Args:
            persist_directory: Directory to persist the vector store
            collection_name: Name of the collection to use
        """
        self.persist_directory = Path(persist_directory)
        
        # Only create backup if directory exists and we're using a different collection name
        if self.persist_directory.exists():
            # Check if we need to create a backup (only if collection name changed)
            backup_dir = self.persist_directory.parent / f"{self.persist_directory.name}_backup"
            if not backup_dir.exists() and collection_name != "caregiving_documents_openai":
                logger.info(f"Creating backup of existing vector store before switching to collection: {collection_name}")
                shutil.move(str(self.persist_directory), str(backup_dir))
        
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings with OpenAI model
        self.embeddings = OpenAIEmbeddings(
            model=DEFAULT_EMBEDDING_MODEL
        )
        
        # Initialize vector store with collection
        self.vector_store = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        logger.info(f"Initialized Chroma vector store with collection: {collection_name}")
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        batch_size: int = 100
    ) -> None:
        """
        Add documents to the vector store in batches to avoid token limits.
        
        Args:
            texts: List of text chunks
            metadatas: List of metadata dictionaries
            ids: List of document IDs
            batch_size: Number of documents to process in each batch
        """
        total_documents = len(texts)
        logger.info(f"Adding {total_documents} documents to vector store in batches of {batch_size}")
        
        # Process documents in batches
        for i in range(0, total_documents, batch_size):
            batch_end = min(i + batch_size, total_documents)
            batch_texts = texts[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_ids = ids[i:batch_end]
            
            # Create Document objects for this batch
            documents = [
                Document(page_content=text, metadata=metadata)
                for text, metadata in zip(batch_texts, batch_metadatas)
            ]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_documents + batch_size - 1)//batch_size} ({len(documents)} documents)")
            
            try:
                # Add batch to vector store
                self.vector_store.add_documents(documents, ids=batch_ids)
                logger.info(f"Successfully added batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {str(e)}")
                # If batch fails, try with smaller batch size
                if "max_tokens_per_request" in str(e) and batch_size > 10:
                    logger.info(f"Token limit exceeded, retrying with smaller batch size: {batch_size//2}")
                    # Recursively call with smaller batch size
                    self.add_documents(batch_texts, batch_metadatas, batch_ids, batch_size=batch_size//2)
                else:
                    raise e
        
        logger.info("All documents added successfully")
    
    def search(
        self,
        query: str,
        n_results: int = 4,
        where: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents (compatibility method).
        
        Args:
            query: Search query
            n_results: Number of results to return
            where: Optional filter conditions
            
        Returns:
            Dictionary containing search results
        """
        # Convert where filter to Chroma filter format if needed
        filter_dict = where if where else None
        
        logger.info(f"Searching for query: {query}")
        
        # Get search results
        docs = self.similarity_search(
            query=query,
            k=n_results,
            filter=filter_dict
        )
        
        logger.info(f"Found {len(docs)} results")
        
        # Format results to match expected structure
        return {
            "documents": [doc.page_content for doc in docs],
            "metadatas": [doc.metadata for doc in docs],
            "distances": [0.0] * len(docs)  # Chroma doesn't return distances by default
        }
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        logger.info(f"Performing similarity search for query: {query}")
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter=filter
        )
    
    def get_retriever(self, search_kwargs: Dict[str, Any] = None) -> Any:
        """
        Get a retriever for the vector store.
        
        Args:
            search_kwargs: Optional search parameters
            
        Returns:
            Retriever object
        """
        return self.vector_store.as_retriever(
            search_kwargs=search_kwargs or {"k": 4}
        )
    
    def debug_search(self, query: str, k: int = 4) -> None:
        """
        Debug search by showing similarity scores and document contents.
        
        Args:
            query: Search query
            k: Number of results to return
        """
        # Get query embedding
        logger.info(f"Getting embedding for query: {query}")
        query_embedding = self.embeddings.embed_query(query)
        print(f"\nQuery: {query}", flush=True)
        print(f"Query embedding dimension: {len(query_embedding)}", flush=True)
        
        # Get all documents and their embeddings
        collection = self.vector_store._collection
        results = collection.get(include=['embeddings', 'documents', 'metadatas'])
        
        if not results['ids']:
            print("No documents found in the vector store!", flush=True)
            return
        
        print(f"Found {len(results['ids'])} documents", flush=True)
        print(f"First document embedding dimension: {len(results['embeddings'][0])}", flush=True)
        
        # Calculate similarities
        similarities = []
        for doc_id, doc, metadata, embedding in zip(
            results['ids'],
            results['documents'],
            results['metadatas'],
            results['embeddings']
        ):
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((similarity, doc, metadata))
        
        # Sort by similarity score (first element of each tuple)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Print results
        print("\nTop matches:", flush=True)
        for i, (similarity, doc, metadata) in enumerate(similarities[:k], 1):
            print(f"\n{i}. Similarity: {similarity:.4f}", flush=True)
            print(f"Document ID: {metadata.get('chunk_id', 'N/A')}", flush=True)
            print(f"Source: {metadata.get('source_path', 'N/A')}", flush=True)
            print(f"Content: {doc[:200]}...", flush=True)
        
        # Force flush
        sys.stdout.flush()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def list_collections(self) -> None:
        """List all collections in the vector store."""
        collection = self.vector_store._collection
        results = collection.get()
        
        print(f"\nCollection: {collection.name}")
        print(f"Number of documents: {len(results['ids'])}")
        
        if results['ids']:
            print("\nSample documents:")
            for i, (doc_id, doc, metadata) in enumerate(zip(results['ids'][:3], results['documents'][:3], results['metadatas'][:3]), 1):
                print(f"\n{i}. ID: {doc_id}")
                print(f"Source: {metadata.get('source_path', 'N/A')}")
                print(f"Content: {doc[:200]}...") 