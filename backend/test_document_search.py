#!/usr/bin/env python3
"""
Test script to check document search via the FastAPI endpoint
"""
import requests
import sys

BACKEND_URL = "http://localhost:8000"


def search_documents(query: str, limit: int = 5):
    url = f"{BACKEND_URL}/api/documents/search"
    params = {"query": query, "limit": limit}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"Search query: '{query}'")
        print(f"Total results: {data.get('total', 0)}\n")
        for i, result in enumerate(data.get("results", []), 1):
            meta = result.get("metadata", {})
            print(f"Result {i}:")
            print(f"  Document ID: {meta.get('document_id', 'N/A')}")
            print(f"  Filename: {meta.get('source', 'N/A')}")
            print(f"  Similarity Score: {result.get('similarity_score', 'N/A')}")
            print(f"  Text Preview: {result.get('text', '')[:120]}...\n")
    except Exception as e:
        print(f"Error searching documents: {e}")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Naren Allam"
    search_documents(query)
