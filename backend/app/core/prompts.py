"""
Prompt templates for the Personal Assistant AI Chatbot
"""


def get_system_prompt() -> str:
    """
    Get the base system prompt for regular chat without RAG context

    Returns:
        System prompt template
    """
    return """You are a helpful Personal Assistant AI. You are knowledgeable, professional, and friendly. 
Your goal is to assist users with their questions and help them accomplish their tasks.

Previous conversation:
{chat_history}

User: {question}

Please provide a helpful and accurate response based on your knowledge."""


def get_rag_prompt() -> str:
    """
    Get the RAG prompt template for context-aware chat

    Returns:
        RAG prompt template
    """
    return """You are a helpful Personal Assistant AI with access to the user's uploaded documents and data. 
Use the provided context information to answer questions accurately and helpfully.

Context Information:
{context}

Previous conversation:
{chat_history}

User: {question}

Based on the context information and conversation history, please provide a detailed and accurate response. 
If the context doesn't contain relevant information, mention that and provide a general response."""


def get_content_generation_prompt(content_type: str) -> str:
    """
    Get content generation prompt template

    Args:
        content_type: Type of content to generate

    Returns:
        Content generation prompt template
    """
    prompts = {
        "email": """You are an expert email writer. Create a professional and effective email based on the following requirements:

Topic: {topic}
Tone: {tone}
Length: {length}
Additional Instructions: {instructions}

{context_section}

Please write a well-structured email with appropriate subject line, greeting, body, and closing.""",
        "letter": """You are an expert letter writer. Create a formal letter based on the following requirements:

Topic: {topic}
Tone: {tone}
Length: {length}
Additional Instructions: {instructions}

{context_section}

Please write a well-structured formal letter with proper formatting, including date, address, salutation, body, and closing.""",
        "social_post": """You are a social media expert. Create an engaging social media post based on the following requirements:

Topic: {topic}
Tone: {tone}
Length: {length}
Additional Instructions: {instructions}

{context_section}

Please write an engaging social media post that is appropriate for the platform and audience.""",
        "blog_post": """You are a professional blog writer. Create a compelling blog post based on the following requirements:

Topic: {topic}
Tone: {tone}
Length: {length}
Additional Instructions: {instructions}

{context_section}

Please write a well-structured blog post with an engaging title, introduction, main content, and conclusion.""",
        "summary": """You are an expert at creating summaries. Create a comprehensive summary based on the following requirements:

Topic: {topic}
Tone: {tone}
Length: {length}
Additional Instructions: {instructions}

{context_section}

Please write a clear and concise summary that captures the key points and main ideas.""",
        "report": """You are a professional report writer. Create a detailed report based on the following requirements:

Topic: {topic}
Tone: {tone}
Length: {length}
Additional Instructions: {instructions}

{context_section}

Please write a comprehensive report with proper structure, including executive summary, main content, and conclusions.""",
    }

    return prompts.get(content_type, prompts["email"])


def get_search_prompt() -> str:
    """
    Get search prompt template for query expansion

    Returns:
        Search prompt template
    """
    return """Given the user's search query, help expand and improve it for better document retrieval.

Original query: {query}

Please provide:
1. Alternative phrasings of the query
2. Related keywords and terms
3. Context that might be relevant

Expanded search terms:"""


def get_document_summary_prompt() -> str:
    """
    Get document summary prompt template

    Returns:
        Document summary prompt template
    """
    return """Please create a concise summary of the following document content:

Document Title: {title}
Document Type: {doc_type}

Content:
{content}

Please provide:
1. A brief summary (2-3 sentences)
2. Key topics covered
3. Important keywords or entities mentioned

Summary:"""
