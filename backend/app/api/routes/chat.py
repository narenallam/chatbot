"""
Chat API routes for the Personal Assistant AI Chatbot
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import List, AsyncGenerator
import logging
import json

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    ConversationHistory,
    ErrorResponse,
)
from app.services.chat_service import chat_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat message with optional RAG context

    Args:
        request: Chat request with message and optional parameters

    Returns:
        ChatResponse with AI reply and sources
    """
    try:
        response = await chat_service.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            use_context=request.use_context,
            temperature=request.temperature,
        )
        return response

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process chat message: {str(e)}"
        )


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Process a chat message with streaming response

    Args:
        request: Chat request with message and optional parameters

    Returns:
        Streaming response with AI reply chunks
    """
    try:

        async def generate_stream() -> AsyncGenerator[str, None]:
            async for chunk in chat_service.chat_stream(
                message=request.message,
                conversation_id=request.conversation_id,
                use_context=request.use_context,
                temperature=request.temperature,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"

            # Send end signal
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    except Exception as e:
        logger.error(f"Chat stream endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process streaming chat message: {str(e)}",
        )


@router.get("/chat/history/{conversation_id}", response_model=List[ChatMessage])
async def get_conversation_history(conversation_id: str):
    """
    Get conversation history for a specific conversation

    Args:
        conversation_id: Conversation identifier

    Returns:
        List of chat messages
    """
    try:
        history = chat_service.get_conversation_history(conversation_id)
        return history

    except Exception as e:
        logger.error(f"Get history error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get conversation history: {str(e)}"
        )


@router.delete("/chat/history/{conversation_id}")
async def clear_conversation_history(conversation_id: str):
    """
    Clear conversation history for a specific conversation

    Args:
        conversation_id: Conversation identifier

    Returns:
        Success message
    """
    try:
        success = chat_service.clear_conversation(conversation_id)
        if success:
            return {"message": f"Conversation {conversation_id} cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")

    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to clear conversation history: {str(e)}"
        )


@router.post("/chat/restore/{conversation_id}")
async def restore_conversation_context(conversation_id: str):
    """
    Restore in-memory context for a conversation from the database.
    Args:
        conversation_id: Conversation/session ID
    Returns:
        Success message
    """
    try:
        chat_service.restore_conversation_context(conversation_id)
        return {
            "message": f"Context for conversation {conversation_id} restored successfully"
        }
    except Exception as e:
        logger.error(f"Restore context error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to restore context: {str(e)}"
        )
