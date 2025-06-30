#!/usr/bin/env python3
"""
Simple test script to debug vector search functionality
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.vector_service import vector_service
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_vector_search():
    """Test vector search functionality"""
    try:
        print("=== Testing Vector Search ===")

        # Get collection stats
        stats = vector_service.get_collection_stats()
        print(f"Collection stats: {stats}")

        # Test simple search
        query = "name person"
        print(f"\nTesting search with query: '{query}'")

        results = vector_service.search_similar(query, 5)
        print(f"Search returned {len(results)} results")

        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"  Source: {result.get('source', 'Unknown')}")
            print(f"  Similarity: {result.get('similarity_score', 0):.3f}")
            print(f"  Text preview: {result.get('text', '')[:100]}...")
            print(f"  Metadata: {result.get('metadata', {})}")

        # Test hybrid search
        print(f"\nTesting hybrid search with query: '{query}'")
        hybrid_results = vector_service.hybrid_search(query, 5)
        print(f"Hybrid search returned {len(hybrid_results)} results")

        for i, result in enumerate(hybrid_results):
            print(f"\nHybrid Result {i+1}:")
            print(f"  Source: {result.get('source', 'Unknown')}")
            print(f"  Similarity: {result.get('similarity_score', 0):.3f}")
            print(f"  Hybrid score: {result.get('hybrid_score', 0):.3f}")
            print(f"  Text preview: {result.get('text', '')[:100]}...")

    except Exception as e:
        print(f"Error testing vector search: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_vector_search()
