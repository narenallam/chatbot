#!/usr/bin/env python3
"""
Test script to verify document preview functionality
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path


async def test_document_endpoints():
    """Test document preview endpoints"""
    base_url = "http://localhost:8000"

    async with aiohttp.ClientSession() as session:
        print("🔍 Testing document endpoints...")

        # Test 1: List documents
        print("\n1. Testing /api/documents/ endpoint...")
        try:
            async with session.get(f"{base_url}/api/documents/") as response:
                if response.status == 200:
                    data = await response.json()
                    documents = data.get("documents", [])
                    print(f"✅ Found {len(documents)} documents")

                    if documents:
                        # Test 2: Get document info for first document
                        first_doc = documents[0]
                        doc_id = first_doc["id"]
                        filename = first_doc["name"]
                        print(
                            f"\n2. Testing document info for: {filename} (ID: {doc_id})"
                        )

                        # Test document info endpoint
                        async with session.get(
                            f"{base_url}/api/documents/{doc_id}/info"
                        ) as info_response:
                            if info_response.status == 200:
                                doc_info = await info_response.json()
                                print(f"✅ Document info retrieved successfully")
                                print(
                                    f"   - File path: {doc_info.get('metadata', {}).get('file_path', 'Not found')}"
                                )
                                print(
                                    f"   - Content type: {doc_info.get('metadata', {}).get('content_type', 'Unknown')}"
                                )

                                # Test 3: Test preview endpoint
                                print(f"\n3. Testing preview endpoint for: {filename}")
                                async with session.get(
                                    f"{base_url}/api/documents/preview/{doc_id}"
                                ) as preview_response:
                                    if preview_response.status == 200:
                                        content_type = preview_response.headers.get(
                                            "content-type", ""
                                        )
                                        print(
                                            f"✅ Preview endpoint working - Content-Type: {content_type}"
                                        )
                                    else:
                                        print(
                                            f"❌ Preview endpoint failed - Status: {preview_response.status}"
                                        )
                                        error_text = await preview_response.text()
                                        print(f"   Error: {error_text}")

                                # Test 4: Test original endpoint
                                print(f"\n4. Testing original endpoint for: {filename}")
                                async with session.get(
                                    f"{base_url}/api/documents/original/{doc_id}"
                                ) as original_response:
                                    if original_response.status == 200:
                                        content_type = original_response.headers.get(
                                            "content-type", ""
                                        )
                                        print(
                                            f"✅ Original endpoint working - Content-Type: {content_type}"
                                        )
                                    else:
                                        print(
                                            f"❌ Original endpoint failed - Status: {original_response.status}"
                                        )
                                        error_text = await original_response.text()
                                        print(f"   Error: {error_text}")
                            else:
                                print(
                                    f"❌ Document info endpoint failed - Status: {info_response.status}"
                                )
                                error_text = await info_response.text()
                                print(f"   Error: {error_text}")
                    else:
                        print("⚠️  No documents found to test")

                else:
                    print(
                        f"❌ List documents endpoint failed - Status: {response.status}"
                    )
                    error_text = await response.text()
                    print(f"   Error: {error_text}")

        except Exception as e:
            print(f"❌ Error testing endpoints: {e}")

        print("\n" + "=" * 50)
        print("Test completed!")


if __name__ == "__main__":
    asyncio.run(test_document_endpoints())
