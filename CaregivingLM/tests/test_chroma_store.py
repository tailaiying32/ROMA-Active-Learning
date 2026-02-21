import pytest
from pathlib import Path
import shutil
from src.vector_store.chroma_store import ChromaVectorStore

# Test data
TEST_DOCUMENTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast orange fox leaps across a sleepy canine.",
    "The weather is beautiful today.",
    "It's raining cats and dogs outside."
]

TEST_METADATA = [
    {"source_path": "test1.txt", "chunk_id": "chunk1"},
    {"source_path": "test1.txt", "chunk_id": "chunk2"},
    {"source_path": "test2.txt", "chunk_id": "chunk1"},
    {"source_path": "test2.txt", "chunk_id": "chunk2"}
]

TEST_IDS = ["doc1", "doc2", "doc3", "doc4"]

@pytest.fixture
def vector_store(tmp_path):
    """Create a temporary vector store for testing."""
    store = ChromaVectorStore(
        persist_directory=str(tmp_path),
        collection_name="test_collection"
    )
    yield store
    # Cleanup
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

def test_initialization(vector_store):
    """Test vector store initialization."""
    assert vector_store.persist_directory.exists()
    assert vector_store.vector_store is not None

def test_add_documents(vector_store):
    """Test adding documents to the vector store."""
    vector_store.add_documents(TEST_DOCUMENTS, TEST_METADATA, TEST_IDS)
    
    # Verify documents were added
    vector_store.list_collections()
    
    # Check if we can retrieve documents
    results = vector_store.similarity_search("fox", k=2)
    assert len(results) > 0
    assert any("fox" in doc.page_content.lower() for doc in results)

def test_search_functionality(vector_store):
    """Test search functionality with different queries."""
    vector_store.add_documents(TEST_DOCUMENTS, TEST_METADATA, TEST_IDS)
    
    # Test semantic search
    results = vector_store.search("fox jumping", n_results=2)
    assert len(results["documents"]) > 0
    assert any("fox" in doc.lower() for doc in results["documents"])
    
    # Test with filter
    results = vector_store.search(
        "fox",
        n_results=2,
        where={"source_path": "test1.txt"}
    )
    assert len(results["documents"]) > 0
    assert all("test1.txt" in meta["source_path"] for meta in results["metadatas"])

def test_debug_search(vector_store, capsys):
    """Test debug search functionality."""
    # Add test documents
    vector_store.add_documents(TEST_DOCUMENTS, TEST_METADATA, TEST_IDS)
    
    # First verify documents were added
    vector_store.list_collections()
    initial_output = capsys.readouterr()
    print("\nInitial collection state:")
    print(initial_output.out)
    
    # Run debug search
    vector_store.debug_search("fox", k=2)
    
    # Capture and verify output
    captured = capsys.readouterr()
    print("\nDebug search output:")
    print(captured.out)
    
    # Verify the output contains expected information
    assert "Query: fox" in captured.out, "Query not found in output"
    assert "Found" in captured.out, "Document count not found in output"
    assert "Similarity:" in captured.out, "Similarity scores not found in output"
    assert "fox" in captured.out.lower(), "Expected content not found in output"
    
    # Verify we got the expected number of results
    lines = captured.out.split('\n')
    similarity_lines = [line for line in lines if "Similarity:" in line]
    assert len(similarity_lines) == 2, f"Expected 2 results, got {len(similarity_lines)}"

def test_empty_store(vector_store):
    """Test behavior with empty store."""
    # Test search on empty store
    results = vector_store.similarity_search("test", k=2)
    assert len(results) == 0
    
    # Test debug search on empty store
    vector_store.debug_search("test")
    # Should not raise any errors

def test_cosine_similarity(vector_store):
    """Test cosine similarity calculation."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    vec3 = [1.0, 1.0, 0.0]
    
    # Test orthogonal vectors
    assert vector_store._cosine_similarity(vec1, vec2) == 0.0
    
    # Test same vector
    assert vector_store._cosine_similarity(vec1, vec1) == 1.0
    
    # Test similar vectors
    similarity = vector_store._cosine_similarity(vec1, vec3)
    assert 0.0 < similarity < 1.0

def test_retriever(vector_store):
    """Test retriever functionality."""
    vector_store.add_documents(TEST_DOCUMENTS, TEST_METADATA, TEST_IDS)
    
    retriever = vector_store.get_retriever(search_kwargs={"k": 2})
    results = retriever.get_relevant_documents("fox")
    
    assert len(results) > 0
    assert any("fox" in doc.page_content.lower() for doc in results)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 