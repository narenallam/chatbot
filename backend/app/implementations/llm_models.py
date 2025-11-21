"""
Technology-Specific LLM Model Implementations
"""

import logging
from typing import Dict, Any, Optional, AsyncGenerator
import asyncio

from app.core.interfaces import LLMInterface, ServiceFactory

logger = logging.getLogger(__name__)

class OllamaLLM(LLMInterface):
    """Ollama LLM implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model_name = config.get('model_name', 'llama3.1:8b')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2048)
        
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Ollama client"""
        try:
            from langchain_ollama import ChatOllama
            
            self.client = ChatOllama(
                base_url=self.base_url,
                model=self.model_name,
                temperature=self.temperature,
                timeout=120,
                request_timeout=120,
                num_predict=self.max_tokens
            )
            
            logger.info(f"Initialized Ollama client: {self.model_name}")
            
        except ImportError:
            logger.error("langchain_ollama not available")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion"""
        try:
            if not self.client:
                raise RuntimeError("Client not initialized")
            
            # Override temperature if provided
            temperature = kwargs.get('temperature', self.temperature)
            self.client.temperature = temperature
            
            # Generate response
            from langchain.schema import HumanMessage
            response = await self.client.ainvoke([HumanMessage(content=prompt)])
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            raise
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming text completion"""
        try:
            if not self.client:
                raise RuntimeError("Client not initialized")
            
            # Override temperature if provided
            temperature = kwargs.get('temperature', self.temperature)
            self.client.temperature = temperature
            
            from langchain.schema import HumanMessage
            
            async for chunk in self.client.astream([HumanMessage(content=prompt)]):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Error in Ollama streaming: {e}")
            raise
    
    async def chat(
        self, 
        messages: list, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """Chat-based text generation"""
        try:
            if not self.client:
                raise RuntimeError("Client not initialized")
            
            from langchain.schema import HumanMessage, AIMessage, SystemMessage
            
            # Convert messages to LangChain format
            lc_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'system':
                    lc_messages.append(SystemMessage(content=content))
                elif role == 'assistant':
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))
            
            # Set parameters
            self.client.temperature = temperature
            
            response = await self.client.ainvoke(lc_messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in Ollama chat: {e}")
            raise
    
    async def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: roughly 4 characters per token
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "type": "ollama",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

class OpenAILLM(LLMInterface):
    """OpenAI LLM implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key')
        self.model_name = config.get('model_name', 'gpt-4')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2048)
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            from langchain_openai import ChatOpenAI
            
            self.client = ChatOpenAI(
                openai_api_key=self.api_key,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=120
            )
            
            logger.info(f"Initialized OpenAI client: {self.model_name}")
            
        except ImportError:
            logger.error("langchain_openai not available")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion"""
        try:
            if not self.client:
                raise RuntimeError("Client not initialized")
            
            # Override temperature if provided
            temperature = kwargs.get('temperature', self.temperature)
            self.client.temperature = temperature
            
            from langchain.schema import HumanMessage
            response = await self.client.ainvoke([HumanMessage(content=prompt)])
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {e}")
            raise
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming text completion"""
        try:
            if not self.client:
                raise RuntimeError("Client not initialized")
            
            # Override temperature if provided
            temperature = kwargs.get('temperature', self.temperature)
            self.client.temperature = temperature
            
            from langchain.schema import HumanMessage
            
            async for chunk in self.client.astream([HumanMessage(content=prompt)]):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Error in OpenAI streaming: {e}")
            raise
    
    async def chat(
        self, 
        messages: list, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """Chat-based text generation"""
        try:
            if not self.client:
                raise RuntimeError("Client not initialized")
            
            from langchain.schema import HumanMessage, AIMessage, SystemMessage
            
            # Convert messages to LangChain format
            lc_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'system':
                    lc_messages.append(SystemMessage(content=content))
                elif role == 'assistant':
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))
            
            # Set parameters
            self.client.temperature = temperature
            
            response = await self.client.ainvoke(lc_messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in OpenAI chat: {e}")
            raise
    
    async def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model_name)
            return len(encoding.encode(text))
        except:
            # Fallback estimation
            return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "type": "openai",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

class HuggingFaceLLM(LLMInterface):
    """HuggingFace LLM implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get('model_name', 'microsoft/DialoGPT-medium')
        self.device = config.get('device', 'cpu')
        self.max_length = config.get('max_length', 1000)
        
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize HuggingFace model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Initialized HuggingFace model: {self.model_name}")
            
        except ImportError:
            logger.error("transformers not available")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model: {e}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion"""
        try:
            if not self.model or not self.tokenizer:
                raise RuntimeError("Model not initialized")
            
            import torch
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            
            # Generate response
            with torch.no_grad():
                outputs = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.generate(
                        inputs,
                        max_length=min(inputs.shape[1] + 200, self.max_length),
                        do_sample=True,
                        temperature=kwargs.get('temperature', 0.7),
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating with HuggingFace: {e}")
            raise
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming text completion (simplified for HuggingFace)"""
        try:
            # For simplicity, generate full response and yield in chunks
            full_response = await self.generate(prompt, **kwargs)
            
            # Yield response in chunks
            chunk_size = 10
            for i in range(0, len(full_response), chunk_size):
                chunk = full_response[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.01)  # Small delay for streaming effect
                
        except Exception as e:
            logger.error(f"Error in HuggingFace streaming: {e}")
            raise
    
    async def chat(
        self, 
        messages: list, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """Chat-based text generation"""
        try:
            # Convert messages to a single prompt
            prompt_parts = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    prompt_parts.append(f"System: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
                else:
                    prompt_parts.append(f"User: {content}")
            
            prompt = "\n".join(prompt_parts) + "\nAssistant:"
            
            return await self.generate(prompt, temperature=temperature)
            
        except Exception as e:
            logger.error(f"Error in HuggingFace chat: {e}")
            raise
    
    async def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        try:
            if self.tokenizer:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            else:
                # Fallback estimation
                return len(text) // 4
        except:
            return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "type": "huggingface",
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length
        }

class GeminiLLM(LLMInterface):
    """Google Gemini LLM implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key')
        self.model_name = config.get('model_name', 'gemini-pro')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2048)
        
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
            
            logger.info(f"Initialized Gemini client: {self.model_name}")
            
        except ImportError:
            logger.error("google-generativeai not available. Install with: pip install google-generativeai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion"""
        try:
            if not self.client:
                raise RuntimeError("Client not initialized")
            
            # Override temperature if provided
            temperature = kwargs.get('temperature', self.temperature)
            
            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.generate_content(
                    prompt,
                    generation_config={
                        'temperature': temperature,
                        'max_output_tokens': self.max_tokens,
                    }
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating with Gemini: {e}")
            raise
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming text completion"""
        try:
            if not self.client:
                raise RuntimeError("Client not initialized")
            
            # Override temperature if provided
            temperature = kwargs.get('temperature', self.temperature)
            
            # Generate streaming response
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.generate_content(
                    prompt,
                    generation_config={
                        'temperature': temperature,
                        'max_output_tokens': self.max_tokens,
                    },
                    stream=True
                )
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Error in Gemini streaming: {e}")
            raise
    
    async def chat(
        self, 
        messages: list, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """Chat-based text generation"""
        try:
            if not self.client:
                raise RuntimeError("Client not initialized")
            
            # Convert messages to Gemini format (simple concatenation for now)
            # Gemini uses a different chat structure, but for compatibility
            # we'll convert the messages to a single prompt
            prompt_parts = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'system':
                    prompt_parts.append(f"System: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Model: {content}")
                else:
                    prompt_parts.append(f"User: {content}")
            
            prompt = "\n\n".join(prompt_parts)
            
            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.generate_content(
                    prompt,
                    generation_config={
                        'temperature': temperature,
                        'max_output_tokens': max_tokens,
                    }
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error in Gemini chat: {e}")
            raise
    
    async def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: roughly 4 characters per token
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "type": "gemini",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

# Register implementations
ServiceFactory.register_llm_model("ollama", OllamaLLM)
ServiceFactory.register_llm_model("openai", OpenAILLM)
ServiceFactory.register_llm_model("huggingface", HuggingFaceLLM)
ServiceFactory.register_llm_model("gemini", GeminiLLM)