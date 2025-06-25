"""
Chat Service with RAG (Retrieval Augmented Generation) capabilities
"""

from typing import List, Dict, Any, Optional
import logging
import uuid
from datetime import datetime
import json

from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import settings
from app.services.vector_service import vector_service
from app.models.schemas import ChatMessage, ChatRole, ChatResponse
from app.core.prompts import get_system_prompt, get_rag_prompt

logger = logging.getLogger(__name__)


class ChatService:
    """Chat service with RAG capabilities"""

    def __init__(self):
        self.llm = None
        self.conversations: Dict[str, ConversationBufferWindowMemory] = {}
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            if settings.llm_provider == "ollama":
                self.llm = ChatOllama(
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_model,
                    temperature=settings.default_temperature,
                )
                logger.info(f"Initialized Ollama model: {settings.ollama_model}")
                
            elif settings.llm_provider == "openai" and settings.openai_api_key:
                self.llm = ChatOpenAI(
                    openai_api_key=settings.openai_api_key,
                    model_name=settings.openai_model,
                    temperature=settings.default_temperature,
                )
                logger.info(f"Initialized OpenAI model: {settings.openai_model}")
                
            else:
                # Fallback to Ollama if no provider is configured
                logger.warning(f"LLM provider '{settings.llm_provider}' not configured properly, falling back to Ollama")
                self.llm = ChatOllama(
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_model,
                    temperature=settings.default_temperature,
                )
                logger.info(f"Fallback: Initialized Ollama model: {settings.ollama_model}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        use_context: bool = True,
        temperature: Optional[float] = None,
    ) -> ChatResponse:
        """
        Process a chat message with optional RAG context

        Args:
            message: User message
            conversation_id: Optional conversation ID for history
            use_context: Whether to use RAG context
            temperature: Optional temperature override

        Returns:
            ChatResponse with AI reply and sources
        """
        try:
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            # Get or create conversation memory
            memory = self._get_conversation_memory(conversation_id)

            # Retrieve relevant context if requested
            context_docs = []
            sources = []
            if use_context:
                context_docs = await self._retrieve_context(message)
                sources = [
                    {
                        "document_id": doc["document_id"],
                        "filename": doc["metadata"].get("filename", "Unknown"),
                        "chunk_text": (
                            doc["text"][:200] + "..."
                            if len(doc["text"]) > 200
                            else doc["text"]
                        ),
                        "similarity_score": doc["similarity_score"],
                    }
                    for doc in context_docs
                ]

            # Build prompt with context
            prompt = self._build_chat_prompt(message, context_docs, memory)

            # Set temperature if provided
            if temperature is not None:
                self.llm.temperature = temperature

            # Generate response
            response = await self._generate_response(prompt)

            # Add messages to memory
            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(response)

            # Count tokens (approximate)
            tokens_used = self._estimate_tokens(message + response)

            return ChatResponse(
                message=response,
                conversation_id=conversation_id,
                sources=sources,
                tokens_used=tokens_used,
            )

        except Exception as e:
            logger.error(f"Failed to process chat message: {e}")
            raise

    async def _retrieve_context(
        self, query: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context documents for the query

        Args:
            query: User query
            n_results: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        try:
            return vector_service.search_similar(query=query, n_results=n_results)
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []

    def _build_chat_prompt(
        self,
        message: str,
        context_docs: List[Dict[str, Any]],
        memory: ConversationBufferWindowMemory,
    ) -> str:
        """
        Build the chat prompt with context and conversation history

        Args:
            message: User message
            context_docs: Retrieved context documents
            memory: Conversation memory

        Returns:
            Formatted prompt string
        """
        try:
            # Get conversation history
            history = memory.chat_memory.messages

            # Build context string
            context_text = ""
            if context_docs:
                context_parts = []
                for i, doc in enumerate(context_docs, 1):
                    filename = doc["metadata"].get("filename", "Unknown")
                    context_parts.append(f"[{i}] From {filename}:\n{doc['text']}")
                context_text = "\n\n".join(context_parts)

            # Use appropriate prompt template
            if context_docs:
                prompt_template = get_rag_prompt()
                return prompt_template.format(
                    context=context_text,
                    chat_history=self._format_chat_history(history),
                    question=message,
                )
            else:
                prompt_template = get_system_prompt()
                return prompt_template.format(
                    chat_history=self._format_chat_history(history), question=message
                )

        except Exception as e:
            logger.error(f"Failed to build chat prompt: {e}")
            return message

    def _format_chat_history(self, messages) -> str:
        """Format chat history for prompt"""
        if not messages:
            return "No previous conversation."

        formatted = []
        for msg in messages[-10:]:  # Last 10 messages
            if hasattr(msg, "content"):
                role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
                formatted.append(f"{role}: {msg.content}")

        return "\n".join(formatted)

    async def _generate_response(self, prompt: str) -> str:
        """
        Generate response from LLM

        Args:
            prompt: Formatted prompt

        Returns:
            AI response
        """
        try:
            # Use invoke for both Ollama and OpenAI
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"

    def _get_conversation_memory(
        self, conversation_id: str
    ) -> ConversationBufferWindowMemory:
        """
        Get or create conversation memory

        Args:
            conversation_id: Conversation identifier

        Returns:
            Conversation memory
        """
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationBufferWindowMemory(
                k=settings.max_chat_history, return_messages=True
            )
        return self.conversations[conversation_id]

    def get_conversation_history(self, conversation_id: str) -> List[ChatMessage]:
        """
        Get conversation history

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of chat messages
        """
        try:
            if conversation_id not in self.conversations:
                return []

            memory = self.conversations[conversation_id]
            messages = []

            for msg in memory.chat_memory.messages:
                if isinstance(msg, HumanMessage):
                    role = ChatRole.USER
                elif isinstance(msg, AIMessage):
                    role = ChatRole.ASSISTANT
                else:
                    role = ChatRole.SYSTEM

                messages.append(
                    ChatMessage(
                        role=role,
                        content=msg.content,
                        timestamp=datetime.now(),  # TODO: Store actual timestamp
                    )
                )

            return messages

        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear conversation history

        Args:
            conversation_id: Conversation identifier

        Returns:
            Success status
        """
        try:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to clear conversation: {e}")
            return False

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation)

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4

    async def generate_content(
        self,
        content_type: str,
        prompt: str,
        context_query: Optional[str] = None,
        tone: str = "professional",
        length: str = "medium",
        additional_instructions: Optional[str] = None,
    ) -> str:
        """
        Generate content based on type and context

        Args:
            content_type: Type of content to generate
            prompt: Content prompt/topic
            context_query: Optional query to retrieve relevant context
            tone: Content tone
            length: Content length
            additional_instructions: Additional instructions

        Returns:
            Generated content
        """
        try:
            # Retrieve context if query provided
            context_docs = []
            if context_query:
                context_docs = await self._retrieve_context(context_query)

            # Build content generation prompt
            content_prompt = self._build_content_prompt(
                content_type, prompt, context_docs, tone, length, additional_instructions
            )

            # Generate content
            return await self._generate_response(content_prompt)

        except Exception as e:
            logger.error(f"Failed to generate content: {e}")
            return f"Failed to generate content: {str(e)}"

    def _build_content_prompt(
        self,
        content_type: str,
        prompt: str,
        context_docs: List[Dict[str, Any]],
        tone: str,
        length: str,
        additional_instructions: Optional[str],
    ) -> str:
        """
        Build content generation prompt

        Args:
            content_type: Type of content
            prompt: Content prompt
            context_docs: Context documents
            tone: Content tone
            length: Content length
            additional_instructions: Additional instructions

        Returns:
            Formatted prompt
        """
        context_text = ""
        if context_docs:
            context_parts = []
            for i, doc in enumerate(context_docs, 1):
                filename = doc["metadata"].get("filename", "Unknown")
                context_parts.append(f"[{i}] From {filename}:\n{doc['text']}")
            context_text = f"\n\nRelevant context:\n{chr(10).join(context_parts)}"

        additional_text = ""
        if additional_instructions:
            additional_text = f"\n\nAdditional instructions: {additional_instructions}"

        return f"""Generate a {content_type} with the following specifications:
- Topic: {prompt}
- Tone: {tone}
- Length: {length}{context_text}{additional_text}

Please create high-quality content that meets these requirements."""


# Global chat service instance
chat_service = ChatService()
