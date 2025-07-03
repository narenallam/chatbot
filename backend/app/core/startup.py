"""
Startup initialization for AI Service Manager and Web Search Components
"""

import logging
import asyncio
from typing import Optional

# Import all implementations to register them
from app.implementations import (
    web_search_providers,
    query_analyzers, 
    web_search_agents,
    content_processors,
    result_fusion
)

from app.services.ai_service_manager import ai_service_manager
from app.core.ai_config import ai_config_manager

logger = logging.getLogger(__name__)

async def initialize_ai_services() -> bool:
    """
    Initialize AI Service Manager with web search capabilities
    
    Returns:
        True if initialization successful, False otherwise
    """
    try:
        logger.info("Starting AI services initialization...")
        
        # Get recommended configuration
        config = await ai_config_manager.get_recommended_config()
        logger.info(f"Using AI configuration: {config.embedding_model}/{config.vector_database}")
        
        # Initialize AI Service Manager
        success = await ai_service_manager.initialize(config)
        if not success:
            logger.error("Failed to initialize AI Service Manager")
            return False
        
        # Log web search status
        stats = ai_service_manager.get_service_stats()
        web_search_enabled = stats.get("web_search_enabled", False)
        logger.info(f"Web search capabilities enabled: {web_search_enabled}")
        
        if web_search_enabled:
            # Get web search provider stats
            web_stats = stats.get("component_stats", {}).get("web_search_agent", {})
            if web_stats:
                providers = web_stats.get("providers", {})
                active_providers = [name for name, info in providers.items() 
                                  if info.get("successes", 0) > 0]
                logger.info(f"Web search providers available: {list(providers.keys())}")
        
        logger.info("AI services initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize AI services: {e}")
        return False

async def test_web_search() -> bool:
    """
    Test web search functionality
    
    Returns:
        True if test successful, False otherwise
    """
    try:
        if not ai_service_manager.is_initialized:
            logger.warning("AI Service Manager not initialized, skipping web search test")
            return False
        
        logger.info("Testing web search functionality...")
        
        # Test query analysis
        test_query = "latest AI developments 2024"
        query_analysis = await ai_service_manager.analyze_query(test_query)
        logger.info(f"Query analysis test - Intent: {getattr(query_analysis, 'intent', 'unknown')}")
        
        # Test search with web search
        search_results = await ai_service_manager.search(
            query=test_query,
            k=3,
            include_web_search=True
        )
        
        web_results = [r for r in search_results if r.search_type and "web" in r.search_type]
        doc_results = [r for r in search_results if not r.search_type or "web" not in r.search_type]
        
        logger.info(f"Web search test completed - Total: {len(search_results)}, Web: {len(web_results)}, Documents: {len(doc_results)}")
        
        return len(search_results) > 0
        
    except Exception as e:
        logger.error(f"Web search test failed: {e}")
        return False

def get_ai_service_status() -> dict:
    """
    Get current status of AI services
    
    Returns:
        Status dictionary
    """
    try:
        if not ai_service_manager.is_initialized:
            return {
                "status": "not_initialized",
                "web_search_enabled": False,
                "error": "AI Service Manager not initialized"
            }
        
        stats = ai_service_manager.get_service_stats()
        
        return {
            "status": "initialized",
            "web_search_enabled": stats.get("web_search_enabled", False),
            "current_config": stats.get("current_config", {}),
            "metrics": stats.get("metrics", {}),
            "component_stats": stats.get("component_stats", {})
        }
        
    except Exception as e:
        logger.error(f"Failed to get AI service status: {e}")
        return {
            "status": "error",
            "web_search_enabled": False,
            "error": str(e)
        }