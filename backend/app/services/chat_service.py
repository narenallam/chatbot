"""
Chat Service with RAG (Retrieval Augmented Generation) capabilities
"""

from typing import List, Dict, Any, Optional
import logging
import uuid
from datetime import datetime
import json
import asyncio

from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import settings
from app.services.vector_service import vector_service
from app.services.database_service import database_service
from app.services.ai_service_manager import ai_service_manager
from app.models.schemas import ChatMessage, ChatRole, ChatResponse
from app.core.prompts import get_system_prompt, get_rag_prompt
from app.core.interfaces import SearchStrategy

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
                    timeout=120,  # 2 minutes timeout
                    request_timeout=120,  # 2 minutes request timeout
                    num_predict=2048,  # Max tokens to generate
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
                logger.warning(
                    f"LLM provider '{settings.llm_provider}' not configured properly, falling back to Ollama"
                )
                self.llm = ChatOllama(
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_model,
                    temperature=settings.default_temperature,
                    timeout=120,  # 2 minutes timeout
                    request_timeout=120,  # 2 minutes request timeout
                    num_predict=2048,  # Max tokens to generate
                )
                logger.info(
                    f"Fallback: Initialized Ollama model: {settings.ollama_model}"
                )

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        use_context: bool = True,
        include_web_search: bool = True,
        selected_search_engine: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> ChatResponse:
        """
        Process a chat message with optional RAG context

        Args:
            message: User message
            conversation_id: Optional conversation ID for history
            use_context: Whether to use RAG context
            include_web_search: Whether to include web search in results
            selected_search_engine: Optional search engine selection
            temperature: Optional temperature override

        Returns:
            ChatResponse with AI reply and sources
        """
        try:
            logger.info(
                f"Chat request received: message='{message}', use_context={use_context}"
            )

            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            # Get or create conversation memory
            memory = self._get_conversation_memory(conversation_id)

            # Retrieve relevant context if requested
            context_docs = []
            sources = []
            if use_context:
                logger.info("Retrieving context documents...")
                context_docs = await self._retrieve_context(
                    message,
                    include_web_search=include_web_search,
                    selected_search_engine=selected_search_engine,
                )
                logger.info(f"Retrieved {len(context_docs)} context documents")

                # Create sources with proper document IDs
                sources = []
                for doc in context_docs:
                    # Handle web search results differently
                    if (
                        doc.get("search_type", "").startswith("web")
                        or doc.get("search_type") == "hybrid_web"
                    ):
                        # Web search result
                        sources.append(
                            {
                                "document_id": None,  # No document ID for web results
                                "filename": doc["metadata"].get(
                                    "title", "Web Search Result"
                                ),
                                "url": doc["metadata"].get("url"),
                                "source_type": "web_search",
                                "provider": doc["metadata"].get("source", "Web"),
                                "chunk_text": (
                                    doc["text"][:200] + "..."
                                    if len(doc["text"]) > 200
                                    else doc["text"]
                                ),
                                "similarity_score": doc.get("similarity_score", 0.0),
                                "is_recent": doc["metadata"].get("is_recent", False),
                                "authority_score": doc["metadata"].get(
                                    "authority_score", 0.5
                                ),
                            }
                        )
                    else:
                        # Document search result
                        file_hash = doc["metadata"].get("file_hash", "unknown")
                        document_id = await self._get_document_id_by_hash(file_hash)

                        sources.append(
                            {
                                "document_id": document_id,
                                "filename": doc["metadata"].get("source", "Unknown"),
                                "source_type": "document",
                                "chunk_text": (
                                    doc["text"][:200] + "..."
                                    if len(doc["text"]) > 200
                                    else doc["text"]
                                ),
                                "similarity_score": doc.get("similarity_score", 0.0),
                            }
                        )

            # Build prompt with context
            logger.info("Building chat prompt...")
            prompt = self._build_chat_prompt(message, context_docs, memory)

            # Set temperature if provided
            if temperature is not None:
                self.llm.temperature = temperature

            # Generate response
            logger.info("Generating LLM response...")
            response = await self._generate_response(prompt)
            logger.info(f"LLM response generated: {len(response)} characters")

            # Add messages to memory
            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(response)

            # Save to database after every message
            try:
                database_service.save_conversation(
                    session_id=conversation_id,  # Use conversation_id as session_id
                    user_message=message,
                    ai_response=response,
                    sources=sources,
                    metadata={"timestamp": datetime.now().isoformat()},
                )
            except Exception as db_exc:
                logger.error(f"Failed to save conversation turn: {db_exc}")

            # Count tokens (approximate)
            tokens_used = self._estimate_tokens(message + response)

            logger.info(f"Chat completed successfully with {len(sources)} sources")
            return ChatResponse(
                message=response,
                conversation_id=conversation_id,
                sources=sources,
                tokens_used=tokens_used,
            )

        except Exception as e:
            logger.error(f"Failed to process chat message: {e}", exc_info=True)
            raise

    async def chat_stream(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        use_context: bool = True,
        include_web_search: bool = True,
        selected_search_engine: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """
        Process a chat message with streaming response

        Args:
            message: User message
            conversation_id: Optional conversation ID for history
            use_context: Whether to use RAG context
            temperature: Optional temperature override

        Yields:
            Streaming response chunks
        """
        try:
            logger.info(
                f"STREAM: Chat stream request received: message='{message}', use_context={use_context}"
            )

            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            # Get or create conversation memory
            memory = self._get_conversation_memory(conversation_id)

            # Retrieve relevant context if requested
            context_docs = []
            sources = []
            if use_context:
                logger.info("STREAM: Retrieving context documents...")
                context_docs = await self._retrieve_context(
                    message,
                    include_web_search=include_web_search,
                    selected_search_engine=selected_search_engine,
                )
                logger.info(f"STREAM: Retrieved {len(context_docs)} context documents")

                # Create sources with proper document IDs
                sources = []
                for doc in context_docs:
                    # Handle web search results differently
                    if (
                        doc.get("search_type", "").startswith("web")
                        or doc.get("search_type") == "hybrid_web"
                    ):
                        # Web search result
                        sources.append(
                            {
                                "document_id": None,  # No document ID for web results
                                "filename": doc["metadata"].get(
                                    "title", "Web Search Result"
                                ),
                                "url": doc["metadata"].get("url"),
                                "source_type": "web_search",
                                "provider": doc["metadata"].get("source", "Web"),
                                "chunk_text": (
                                    doc["text"][:200] + "..."
                                    if len(doc["text"]) > 200
                                    else doc["text"]
                                ),
                                "similarity_score": doc.get("similarity_score", 0.0),
                                "is_recent": doc["metadata"].get("is_recent", False),
                                "authority_score": doc["metadata"].get(
                                    "authority_score", 0.5
                                ),
                            }
                        )
                    else:
                        # Document search result
                        file_hash = doc["metadata"].get("file_hash", "unknown")
                        document_id = await self._get_document_id_by_hash(file_hash)

                        sources.append(
                            {
                                "document_id": document_id,
                                "filename": doc["metadata"].get("source", "Unknown"),
                                "source_type": "document",
                                "chunk_text": (
                                    doc["text"][:200] + "..."
                                    if len(doc["text"]) > 200
                                    else doc["text"]
                                ),
                                "similarity_score": doc.get("similarity_score", 0.0),
                            }
                        )

            # Send sources first
            if sources:
                logger.info(f"STREAM: Sending {len(sources)} sources to frontend")
                yield {
                    "type": "sources",
                    "sources": sources,
                    "conversation_id": conversation_id,
                }

            # Build prompt with context
            logger.info("STREAM: Building chat prompt...")
            prompt = self._build_chat_prompt(message, context_docs, memory)

            # Set temperature if provided
            if temperature is not None:
                self.llm.temperature = temperature

            # Generate streaming response
            logger.info("STREAM: Starting LLM streaming response...")
            full_response = ""
            chunk_count = 0
            async for chunk in self._generate_streaming_response(prompt):
                if chunk:
                    chunk_count += 1
                    full_response += chunk
                    yield {
                        "type": "content",
                        "content": chunk,
                        "conversation_id": conversation_id,
                    }

            logger.info(
                f"STREAM: Completed streaming with {chunk_count} chunks, total length: {len(full_response)}"
            )

            # Add messages to memory after completion
            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(full_response)

            # Save to database after every message
            try:
                database_service.save_conversation(
                    session_id=conversation_id,  # Use conversation_id as session_id
                    user_message=message,
                    ai_response=full_response,
                    sources=sources,
                    metadata={"timestamp": datetime.now().isoformat()},
                )
            except Exception as db_exc:
                logger.error(f"Failed to save conversation turn: {db_exc}")

            # Send final metadata
            yield {
                "type": "end",
                "conversation_id": conversation_id,
                "total_tokens": self._estimate_tokens(message + full_response),
            }

        except Exception as e:
            logger.error(
                f"STREAM: Failed to process streaming chat message: {e}", exc_info=True
            )
            yield {
                "type": "error",
                "error": str(e),
                "conversation_id": conversation_id or "unknown",
            }

    async def _retrieve_context(
        self,
        query: str,
        n_results: int = 5,
        include_web_search: bool = True,
        selected_search_engine: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context documents for the query

        Args:
            query: User query
            n_results: Number of documents to retrieve
            include_web_search: Whether to include web search in results

        Returns:
            List of relevant documents
        """
        try:
            # Use document service directly for document search (more reliable than AI Service Manager)
            from app.services.document_service import document_service

            # Get document results using the same service as /api/documents/search
            document_results = await document_service.search_documents(
                query, limit=n_results
            )
            logger.info(
                f"Document service returned {len(document_results)} document results"
            )

            # Convert document results to legacy format
            legacy_results = []
            for result in document_results:
                legacy_results.append(
                    {
                        "text": result.get("text", ""),
                        "metadata": {
                            "document_id": result.get("metadata", {}).get(
                                "document_id", ""
                            ),
                            "source": result.get("source", "Unknown"),
                            "filename": result.get("source", "Unknown"),
                            "chunk_index": result.get("chunk_index", 0),
                        },
                        "similarity_score": result.get("similarity_score", 0),
                        "search_type": "document",
                    }
                )
                logger.debug(
                    f"Document result: {result.get('source', 'Unknown')}, score: {result.get('similarity_score', 0)}"
                )

            # Add web search results if AI Service Manager is available and web search is enabled
            if ai_service_manager.is_initialized and include_web_search:
                try:
                    web_search_results = await ai_service_manager.search(
                        query=query,
                        strategy=SearchStrategy.AUTO,
                        k=n_results // 2,  # Split results between documents and web
                        include_web_search=True,
                        selected_search_engine=selected_search_engine,
                    )

                    # Add web results to legacy format
                    for result in web_search_results:
                        if result.search_type in ["web_search", "hybrid_web"]:
                            legacy_results.append(
                                {
                                    "text": result.content,
                                    "metadata": result.metadata,
                                    "similarity_score": result.score,
                                    "search_type": result.search_type,
                                }
                            )

                    logger.info(f"Added {len(web_search_results)} web search results")
                except Exception as e:
                    logger.warning(f"Web search failed: {e}")

            logger.info(
                f"Retrieved {len(legacy_results)} total results (documents + web)"
            )
            return legacy_results

        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            # Final fallback to vector service
            try:
                results = vector_service.hybrid_search(query=query, n_results=n_results)
                return results
            except Exception as fallback_e:
                logger.error(f"Fallback context retrieval also failed: {fallback_e}")
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

            # Build context string with enhanced markdown formatting
            context_text = ""
            if context_docs:
                context_parts = []
                web_sources = []
                doc_sources = []

                for i, doc in enumerate(context_docs, 1):
                    search_type = doc.get("search_type", "")

                    if search_type.startswith("web") or search_type == "hybrid_web":
                        # Web search result
                        title = doc["metadata"].get("title", "Web Search Result")
                        url = doc["metadata"].get("url", "")
                        provider = doc["metadata"].get("source", "Web")
                        is_recent = doc["metadata"].get("is_recent", False)
                        recency_indicator = " 🆕" if is_recent else ""

                        source_info = f"**Web Source {len(web_sources) + 1}**: [{title}]({url}) via {provider}{recency_indicator}"
                        web_sources.append(source_info)
                        context_parts.append(
                            f"### Web Source {len(web_sources)} - {title}\n{doc['text']}"
                        )
                    else:
                        # Document source
                        filename = doc["metadata"].get("source", "Unknown Document")
                        doc_sources.append(
                            f"**Document Source {len(doc_sources) + 1}**: {filename}"
                        )
                        context_parts.append(
                            f"### Document Source {len(doc_sources) + 1} - {filename}\n{doc['text']}"
                        )

                # Combine sources with headers
                all_sources = []
                if doc_sources:
                    all_sources.append("**📄 Document Sources:**")
                    all_sources.extend(doc_sources)
                if web_sources:
                    all_sources.append("\n**🌐 Web Sources:**")
                    all_sources.extend(web_sources)

                # Build final context
                if all_sources:
                    context_text = (
                        "\n".join(all_sources)
                        + "\n\n---\n\n"
                        + "\n\n".join(context_parts)
                    )
                else:
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
            logger.info(f"Calling LLM with prompt length: {len(prompt)} characters")
            logger.debug(f"Prompt content: {prompt[:500]}...")  # Log first 500 chars

            # Use invoke for both Ollama and OpenAI
            messages = [HumanMessage(content=prompt)]
            logger.info("Invoking LLM...")
            response = await self.llm.ainvoke(messages)
            logger.info(f"LLM response received: {len(response.content)} characters")
            logger.debug(
                f"Response content: {response.content[:200]}..."
            )  # Log first 200 chars

            return response.content

        except Exception as e:
            logger.error(f"Failed to generate response: {e}", exc_info=True)
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"

    async def _generate_streaming_response(self, prompt: str):
        """
        Generate streaming AI response using the configured LLM

        Args:
            prompt: Formatted prompt string

        Yields:
            Response chunks
        """
        try:
            logger.info(
                f"STREAM LLM: Starting streaming response with prompt length: {len(prompt)}"
            )
            logger.debug(f"STREAM LLM: Prompt preview: {prompt[:300]}...")

            messages = [HumanMessage(content=prompt)]
            chunk_count = 0

            logger.info("STREAM LLM: Calling LLM astream...")
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    chunk_count += 1
                    logger.debug(
                        f"STREAM LLM: Received chunk {chunk_count}: {len(chunk.content)} chars"
                    )
                    yield chunk.content

            logger.info(f"STREAM LLM: Completed streaming with {chunk_count} chunks")

        except asyncio.CancelledError:
            logger.info("STREAM LLM: Streaming cancelled by client")
            raise
        except Exception as e:
            logger.error(
                f"STREAM LLM: Failed to generate streaming response: {e}", exc_info=True
            )
            raise

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
                context_docs = await self._retrieve_context(
                    context_query, include_web_search=True
                )

            # Build content generation prompt
            content_prompt = self._build_content_prompt(
                content_type,
                prompt,
                context_docs,
                tone,
                length,
                additional_instructions,
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

    async def _get_document_id_by_hash(self, file_hash: str) -> str:
        """Get document ID by file hash"""
        try:
            files = database_service.get_files()

            for file in files:
                if file.get("file_hash") == file_hash:
                    return file.get("id", "unknown")

            logger.warning(f"No document found for hash: {file_hash}")
            return "unknown"
        except Exception as e:
            logger.error(f"Error getting document ID by hash: {str(e)}")
            return "unknown"

    def restore_conversation_context(self, conversation_id: str):
        """
        Restore in-memory context for a conversation from the database.
        Args:
            conversation_id: Conversation/session ID
        """
        try:
            # Clear current memory
            self.conversations[conversation_id] = ConversationBufferWindowMemory(
                k=settings.max_chat_history, return_messages=True
            )
            memory = self.conversations[conversation_id]
            # Load all messages from DB (in chronological order)
            history = database_service.get_conversation_history(
                conversation_id, limit=1000
            )
            for turn in reversed(
                history
            ):  # DB returns DESC, so reverse for chronological
                memory.chat_memory.add_user_message(turn["user_message"])
                memory.chat_memory.add_ai_message(turn["ai_response"])
            logger.info(
                f"Restored context for conversation {conversation_id} with {len(history)} turns."
            )
        except Exception as e:
            logger.error(
                f"Failed to restore context for conversation {conversation_id}: {e}"
            )


# Global chat service instance
chat_service = ChatService()
