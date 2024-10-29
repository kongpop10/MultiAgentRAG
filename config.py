# Model-specific prompts and configurations
KNOWLEDGE_RETRIEVAL_PROMPT = """You are a knowledge retrieval specialist. Your task is to:
1. Extract key information from the query and provided document (if any)
2. Organize relevant context and facts
3. Identify important relationships and patterns
4. Present the information in a clear, structured format
5. If working with document chunks, focus on maintaining context
Be comprehensive but focused on relevance."""

REASONING_PLANNER_PROMPT = """You are a reasoning planning specialist. Your task is to:
1. Analyze the given context and query
2. Break down the reasoning process into 3-5 logical steps
3. Each step should build upon previous steps
4. Format your response as a numbered list of steps
5. Keep step descriptions clear and actionable

Example format:
Step 1. Initial Analysis
Step 2. Deep Dive
Step 3. Implications
Step 4. Conclusions"""

REASONING_STEP_PROMPT = """You are a specialized reasoning agent responsible for executing a specific step in the reasoning process. Your task is to:
1. Focus solely on your assigned step
2. Consider previous steps' results when available
3. Provide detailed analysis for your specific step
4. Maintain logical flow with other steps
5. Be thorough but concise"""

SYNTHESIS_PROMPT = """You are a synthesis specialist. Your task is to:
1. Integrate knowledge and multi-step reasoning insights
2. Create a coherent narrative
3. Highlight key actionable points
4. Provide clear recommendations
Be concise but comprehensive."""

# Rate limiting configurations
RATE_LIMIT_CONFIGS = {
    'default_wait': 30,
    'max_retries': 3,
    'batch_size': 2,
    'inter_batch_delay': 2
}

# Token management configurations
TOKEN_CONFIGS = {
    'max_input_tokens': 4000,
    'max_output_tokens': 2000,
    'summary_length': 150
}

# Update Google API configuration for paid tier
GOOGLE_API_CONFIG = {
    'model': 'gemini-1.5-pro',  # Using pro model instead of flash
    'api_url': 'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent',
    'max_tokens': 8192,  # Increased token limit
    'api_version': 'v1',
    'streaming': True,  # Enable streaming for faster responses
    'safety_settings': {
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
    },
    'parallel_requests': True,  # Enable parallel processing
    'chunk_size': 4000,  # Larger chunk size for processing
    'overlap': 400      # Increased overlap for better context
}

# Enhanced document handling for paid tier
DOCUMENT_CONFIG = {
    'small_doc_limit': 4000,    # Increased limits
    'medium_doc_limit': 8192,   # Maximum pro model context
    'chunk_size': 4000,         # Larger chunks
    'overlap': 400,             # Better context preservation
    'parallel_chunks': 5,       # Number of parallel chunk processing
    'max_parallel_requests': 10  # Maximum parallel requests
}
