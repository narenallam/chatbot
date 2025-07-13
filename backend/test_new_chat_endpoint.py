#!/usr/bin/env python3
"""
Test script for the new send_message endpoint
"""

import asyncio
import json
from app.api.routes.chat import send_message_endpoint
from app.models.schemas import ChatRequest


async def test_new_endpoint():
    """Test the new send_message endpoint"""

    # Test case 1: LLM only
    print("Testing LLM only mode...")
    request1 = ChatRequest(
        message="What is 2 + 2?",
        search_modes={"llm": True, "documents": False, "web": False},
    )

    try:
        response1 = await send_message_endpoint(request1)
        print(f"LLM Response: {response1.llm[:100]}...")
        print(f"Documents Response: {response1.documents}")
        print(f"Web Response: {response1.web}")
        print("✅ LLM only test passed")
    except Exception as e:
        print(f"❌ LLM only test failed: {e}")

    print("\n" + "=" * 50 + "\n")

    # Test case 2: Documents only (if you have documents uploaded)
    print("Testing Documents only mode...")
    request2 = ChatRequest(
        message="What documents do you have?",
        search_modes={"llm": False, "documents": True, "web": False},
    )

    try:
        response2 = await send_message_endpoint(request2)
        print(f"LLM Response: {response2.llm}")
        print(
            f"Documents Response: {response2.documents[:100] if response2.documents else 'None'}..."
        )
        print(f"Web Response: {response2.web}")
        print("✅ Documents only test passed")
    except Exception as e:
        print(f"❌ Documents only test failed: {e}")

    print("\n" + "=" * 50 + "\n")

    # Test case 3: Multiple modes
    print("Testing multiple modes...")
    request3 = ChatRequest(
        message="What is the weather like?",
        search_modes={"llm": True, "documents": False, "web": True},
    )

    try:
        response3 = await send_message_endpoint(request3)
        print(f"LLM Response: {response3.llm[:100] if response3.llm else 'None'}...")
        print(f"Documents Response: {response3.documents}")
        print(f"Web Response: {response3.web[:100] if response3.web else 'None'}...")
        print("✅ Multiple modes test passed")
    except Exception as e:
        print(f"❌ Multiple modes test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_new_endpoint())
