"""
Technology-Agnostic AI Configuration System
Manages configuration for swappable AI components
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict

# Load environment variables to ensure API keys are available
from dotenv import load_dotenv

load_dotenv()

from app.core.interfaces import AIConfig, ConfigManager, ServiceFactory

logger = logging.getLogger(__name__)

# Default configurations for different technology stacks

DEFAULT_CONFIGS = {
    "local_llama_stack": {
        "embedding_model": "sentence_transformers",
        "embedding_config": {
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "device": "cpu",
            "batch_size": 32,
        },
        "vector_database": "chromadb",
        "vector_config": {
            "collection_name": "documents_ai_manager",
            "persist_directory": "./embeddings",
            "distance_metric": "cosine",
        },
        "precise_retriever": "bm25",
        "precise_config": {"k1": 1.2, "b": 0.75},
        "llm_model": "ollama",
        "llm_config": {
            "base_url": "http://localhost:11434",
            "model_name": "llama3:8b-instruct-q8_0",
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        "search_config": {
            "table_confidence_threshold": 0.6,
            "hybrid_weight_general": 0.4,
            "hybrid_weight_table": 0.6,
            "web_search_enabled": True,
            "web_search_providers": ["brave_search"],
            "web_search_max_results": 5,
            "enabled": True,
            "analyzer_type": "rule_based",
            "analyzer_config": {},
            "processor_type": "advanced",
            "processor_config": {
                "timeout": 15,
                "max_content_length": 10000,
                "enable_trafilatura": True,
                "enable_newspaper": True,
                "enable_readability": True,
            },
            "fusion_type": "intelligent",
            "fusion_config": {
                "web_weight": 0.6,
                "document_weight": 0.4,
                "recency_boost": 0.2,
                "authority_boost": 0.3,
                "diversity_threshold": 0.7,
                "max_results": 10,
            },
            "agent_type": "multi_provider",
            "agent_config": {
                "provider_priority": ["serpapi"],
                "max_concurrent_searches": 1,
                "timeout_per_provider": 15,
                "fallback_enabled": False,
                "providers": {
                    "serpapi": {
                        "api_key": "e51c4394a63d71148aa5cc386e4d5586ba49e0e4ea65056b88a01f78da49016c",
                        "timeout": 15,
                        "max_results": 10,
                        "location": "United States",
                        "device": "desktop",
                    }
                },
            },
        },
    },
    "openai_stack": {
        "embedding_model": "openai",
        "embedding_config": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model_name": "text-embedding-3-large",
            "batch_size": 100,
        },
        "vector_database": "chromadb",
        "vector_config": {
            "collection_name": "documents_openai",
            "persist_directory": "./embeddings_openai",
            "distance_metric": "cosine",
        },
        "precise_retriever": "colbert",
        "precise_config": {
            "model_name": "colbert-ir/colbertv2.0",
            "index_name": "table_precise_openai",
            "index_path": "./colbert_indexes_openai",
        },
        "llm_model": "openai",
        "llm_config": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model_name": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        "search_config": {
            "table_confidence_threshold": 0.6,
            "hybrid_weight_general": 0.3,
            "hybrid_weight_table": 0.7,
            "enabled": True,
            "analyzer_type": "llm_powered",  # Use LLM for analysis with OpenAI
            "analyzer_config": {},
            "processor_type": "advanced",
            "processor_config": {
                "timeout": 15,
                "max_content_length": 15000,  # Higher for OpenAI stack
                "enable_trafilatura": True,
                "enable_newspaper": True,
                "enable_readability": True,
            },
            "fusion_type": "intelligent",
            "fusion_config": {
                "web_weight": 0.5,
                "document_weight": 0.5,
                "recency_boost": 0.3,
                "authority_boost": 0.4,
                "diversity_threshold": 0.6,
                "max_results": 15,
            },
            "agent_type": "multi_provider",
            "agent_config": {
                "provider_priority": ["serpapi"],
                "max_concurrent_searches": 1,
                "timeout_per_provider": 15,
                "fallback_enabled": False,
                "providers": {
                    "serpapi": {
                        "api_key": "e51c4394a63d71148aa5cc386e4d5586ba49e0e4ea65056b88a01f78da49016c",
                        "timeout": 15,
                        "max_results": 10,
                        "location": "United States",
                        "device": "desktop",
                    }
                },
            },
        },
    },
    "huggingface_stack": {
        "embedding_model": "huggingface",
        "embedding_config": {
            "model_name": "microsoft/DialoGPT-medium",
            "device": "cpu",
            "batch_size": 16,
        },
        "vector_database": "faiss",
        "vector_config": {
            "index_type": "IndexFlatIP",
            "dimension": 768,
            "persist_path": "./faiss_index",
        },
        "precise_retriever": "bm25",
        "precise_config": {"k1": 1.2, "b": 0.75},
        "llm_model": "huggingface",
        "llm_config": {
            "model_name": "microsoft/DialoGPT-medium",
            "device": "cpu",
            "max_length": 1000,
        },
        "search_config": {
            "table_confidence_threshold": 0.5,
            "hybrid_weight_general": 0.5,
            "hybrid_weight_table": 0.5,
            "enabled": True,
            "analyzer_type": "rule_based",
            "analyzer_config": {},
            "processor_type": "advanced",
            "processor_config": {
                "timeout": 10,
                "max_content_length": 8000,
                "enable_trafilatura": False,  # Disabled for performance
                "enable_newspaper": True,
                "enable_readability": False,
            },
            "fusion_type": "simple",  # Simple fusion for performance
            "fusion_config": {"max_results": 8},
            "agent_type": "multi_provider",
            "agent_config": {
                "provider_priority": ["serpapi"],
                "max_concurrent_searches": 1,
                "timeout_per_provider": 15,
                "fallback_enabled": False,
                "providers": {
                    "serpapi": {
                        "api_key": "e51c4394a63d71148aa5cc386e4d5586ba49e0e4ea65056b88a01f78da49016c",
                        "timeout": 15,
                        "max_results": 10,
                        "location": "United States",
                        "device": "desktop",
                    }
                },
            },
        },
    },
    "performance_stack": {
        "embedding_model": "sentence_transformers",
        "embedding_config": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",  # Faster model
            "device": "cpu",
            "batch_size": 64,
        },
        "vector_database": "faiss",
        "vector_config": {
            "index_type": "IndexIVFFlat",  # Faster for large datasets
            "dimension": 384,
            "persist_path": "./faiss_index_fast",
        },
        "precise_retriever": "bm25",  # Faster than ColBERT
        "precise_config": {"k1": 1.2, "b": 0.75},
        "llm_model": "ollama",
        "llm_config": {
            "base_url": "http://localhost:11434",
            "model_name": "llama3:8b-instruct-q4_0",  # Quantized for speed
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "search_config": {
            "table_confidence_threshold": 0.7,
            "hybrid_weight_general": 0.6,
            "hybrid_weight_table": 0.4,
            "enabled": True,
            "analyzer_type": "rule_based",
            "analyzer_config": {},
            "processor_type": "advanced",
            "processor_config": {
                "timeout": 8,
                "max_content_length": 5000,  # Reduced for performance
                "enable_trafilatura": False,
                "enable_newspaper": False,
                "enable_readability": False,
            },
            "fusion_type": "simple",
            "fusion_config": {"max_results": 5},
            "agent_type": "multi_provider",
            "agent_config": {
                "provider_priority": ["serpapi"],
                "max_concurrent_searches": 1,
                "timeout_per_provider": 15,
                "fallback_enabled": False,
                "providers": {
                    "serpapi": {
                        "api_key": "e51c4394a63d71148aa5cc386e4d5586ba49e0e4ea65056b88a01f78da49016c",
                        "timeout": 15,
                        "max_results": 10,
                        "location": "United States",
                        "device": "desktop",
                    }
                },
            },
        },
    },
}


class AIConfigManager(ConfigManager):
    """Enhanced configuration manager for AI components"""

    def __init__(self, config_file: Optional[str] = None):
        super().__init__(config_file)
        self.config_directory = Path("./configs")
        self.config_directory.mkdir(exist_ok=True)

        # Initialize with default config if none specified
        if not config_file:
            self.config_file = self.config_directory / "ai_config.json"

    async def load_default_config(
        self, stack_name: str = "local_llama_stack"
    ) -> AIConfig:
        """Load a default configuration stack"""
        try:
            if stack_name not in DEFAULT_CONFIGS:
                available_stacks = list(DEFAULT_CONFIGS.keys())
                raise ValueError(
                    f"Unknown stack '{stack_name}'. Available: {available_stacks}"
                )

            config_dict = DEFAULT_CONFIGS[stack_name]
            config = AIConfig(**config_dict)

            # Validate the configuration
            is_valid, errors = await self.validate_config(config)
            if not is_valid:
                logger.warning(f"Configuration validation errors: {errors}")
                # Try to auto-fix some common issues
                config = await self._auto_fix_config(config, errors)

            self.current_config = config
            logger.info(f"Loaded default config stack: {stack_name}")

            return config

        except Exception as e:
            logger.error(f"Error loading default config: {e}")
            raise

    async def _auto_fix_config(self, config: AIConfig, errors: List[str]) -> AIConfig:
        """Attempt to auto-fix common configuration issues"""
        try:
            # Create a copy to modify
            config_dict = asdict(config)

            # Fix missing API keys by falling back to local alternatives
            if "api_key" in str(errors):
                logger.info(
                    "Missing API keys detected, falling back to local alternatives"
                )

                # Replace OpenAI with local alternatives
                if config.embedding_model == "openai":
                    config_dict["embedding_model"] = "sentence_transformers"
                    config_dict["embedding_config"] = DEFAULT_CONFIGS[
                        "local_llama_stack"
                    ]["embedding_config"]

                if config.llm_model == "openai":
                    config_dict["llm_model"] = "ollama"
                    config_dict["llm_config"] = DEFAULT_CONFIGS["local_llama_stack"][
                        "llm_config"
                    ]

                # Replace cloud vector DBs with local ones
                if config.vector_database in ["pinecone", "weaviate"]:
                    config_dict["vector_database"] = "chromadb"
                    config_dict["vector_config"] = DEFAULT_CONFIGS["local_llama_stack"][
                        "vector_config"
                    ]

            # Fix unregistered implementations
            available = ServiceFactory.list_available_implementations()

            if config.embedding_model not in available["embedding_models"]:
                logger.info(
                    f"Embedding model '{config.embedding_model}' not available, using sentence_transformers"
                )
                config_dict["embedding_model"] = "sentence_transformers"
                config_dict["embedding_config"] = DEFAULT_CONFIGS["local_llama_stack"][
                    "embedding_config"
                ]

            if config.vector_database not in available["vector_databases"]:
                logger.info(
                    f"Vector database '{config.vector_database}' not available, using chromadb"
                )
                config_dict["vector_database"] = "chromadb"
                config_dict["vector_config"] = DEFAULT_CONFIGS["local_llama_stack"][
                    "vector_config"
                ]

            return AIConfig(**config_dict)

        except Exception as e:
            logger.error(f"Error auto-fixing config: {e}")
            return config

    async def save_config_as_preset(self, config: AIConfig, preset_name: str):
        """Save configuration as a named preset"""
        try:
            preset_path = self.config_directory / f"{preset_name}_config.json"

            with open(preset_path, "w") as f:
                json.dump(asdict(config), f, indent=2)

            logger.info(f"Saved configuration preset: {preset_name}")

        except Exception as e:
            logger.error(f"Error saving config preset: {e}")
            raise

    async def load_config_preset(self, preset_name: str) -> AIConfig:
        """Load a named configuration preset"""
        try:
            preset_path = self.config_directory / f"{preset_name}_config.json"

            if not preset_path.exists():
                raise FileNotFoundError(f"Preset '{preset_name}' not found")

            with open(preset_path, "r") as f:
                config_dict = json.load(f)

            config = AIConfig(**config_dict)

            # Validate the configuration
            is_valid, errors = await self.validate_config(config)
            if not is_valid:
                logger.warning(f"Preset validation errors: {errors}")
                config = await self._auto_fix_config(config, errors)

            self.current_config = config
            logger.info(f"Loaded configuration preset: {preset_name}")

            return config

        except Exception as e:
            logger.error(f"Error loading config preset: {e}")
            raise

    def list_available_presets(self) -> List[str]:
        """List all available configuration presets"""
        try:
            presets = []

            # Add default stacks
            presets.extend(DEFAULT_CONFIGS.keys())

            # Add saved presets
            for preset_file in self.config_directory.glob("*_config.json"):
                preset_name = preset_file.stem.replace("_config", "")
                if preset_name not in presets:
                    presets.append(preset_name)

            return sorted(presets)

        except Exception as e:
            logger.error(f"Error listing presets: {e}")
            return []

    async def get_recommended_config(self) -> AIConfig:
        """Get recommended configuration based on system capabilities"""
        try:
            # Check system capabilities
            has_gpu = self._check_gpu_availability()
            has_ollama = await self._check_ollama_availability()
            has_openai_key = bool(os.getenv("OPENAI_API_KEY"))

            # Determine best stack
            if has_openai_key:
                stack = "openai_stack"
                logger.info("Recommending OpenAI stack (API key available)")
            elif has_ollama:
                stack = "local_llama_stack"
                logger.info("Recommending Local LLaMA stack (Ollama available)")
            else:
                stack = "performance_stack"
                logger.info("Recommending Performance stack (lightweight)")

            return await self.load_default_config(stack)

        except Exception as e:
            logger.error(f"Error getting recommended config: {e}")
            # Fallback to local stack
            return await self.load_default_config("local_llama_stack")

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    async def _check_ollama_availability(self) -> bool:
        """Check if Ollama is available"""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:11434/api/tags", timeout=2
                ) as response:
                    return response.status == 200
        except:
            return False

    async def switch_config_runtime(self, new_config: AIConfig) -> bool:
        """Switch configuration at runtime"""
        try:
            # Validate new configuration
            is_valid, errors = await self.validate_config(new_config)
            if not is_valid:
                logger.error(f"Cannot switch to invalid config: {errors}")
                return False

            # Store old config as backup
            old_config = self.current_config

            try:
                # Update current config
                self.current_config = new_config

                # Here you would reinitialize services with new config
                # This would be handled by the service manager
                logger.info("Configuration switched successfully")
                return True

            except Exception as e:
                # Rollback on failure
                self.current_config = old_config
                logger.error(f"Failed to switch config, rolled back: {e}")
                return False

        except Exception as e:
            logger.error(f"Error switching config: {e}")
            return False

    def get_config_comparison(
        self, config1: AIConfig, config2: AIConfig
    ) -> Dict[str, Any]:
        """Compare two configurations"""
        try:
            comparison = {
                "embedding_model": {
                    "config1": config1.embedding_model,
                    "config2": config2.embedding_model,
                    "same": config1.embedding_model == config2.embedding_model,
                },
                "vector_database": {
                    "config1": config1.vector_database,
                    "config2": config2.vector_database,
                    "same": config1.vector_database == config2.vector_database,
                },
                "llm_model": {
                    "config1": config1.llm_model,
                    "config2": config2.llm_model,
                    "same": config1.llm_model == config2.llm_model,
                },
                "precise_retriever": {
                    "config1": config1.precise_retriever,
                    "config2": config2.precise_retriever,
                    "same": config1.precise_retriever == config2.precise_retriever,
                },
            }

            return comparison

        except Exception as e:
            logger.error(f"Error comparing configs: {e}")
            return {}


# Global configuration manager instance
ai_config_manager = AIConfigManager()
