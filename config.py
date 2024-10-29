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

# Update Google API configuration for Gemini 1.5 Flash (Free Tier)
GOOGLE_API_CONFIG = {
    'model': 'gemini-1.5-flash',
    'api_url': 'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent',
    'max_tokens': 1000000,
    'api_version': 'v1',
    'streaming': True,
    'document_thresholds': {
        'small': 32768,
        'medium': 128000,
        'large': 1000000
    },
    'chunk_settings': {
        'size': 32000,
        'overlap': 1000,
        'max_parallel': 2
    },
    'rate_limits': {
        'requests_per_minute': 15,
        'tokens_per_minute': 1000000,
        'min_delay': 4,
        'requests_per_day': 1500
    },
    'safety_settings': [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        }
    ]
}

# Update document configuration for free tier
DOCUMENT_CONFIG = {
    'default_model': 'mixtral-8x7b-32768',
    'fallback_model': 'gemini-1.5-flash',
    'model_thresholds': {
        'small': 8000,
        'medium': 16000,
        'large': 32000
    },
    'chunk_settings': {
        'size': 8000,
        'overlap': 500,
        'max_parallel': 5
    },
    'daily_request_limit': 1500,
    'max_parallel_requests': 3
}

# Token pricing configuration (per 1M tokens)
TOKEN_PRICING = {
    'gemini_flash': {
        'input': 0.075,   # $0.075 per 1M input tokens
        'output': 0.30,   # $0.30 per 1M output tokens
        'context_cache': 0.01875  # $0.01875 per 1M tokens for context caching
    },
    'mixtral': {
        'input': 0.50,    # Mixtral pricing for comparison
        'output': 1.50
    }
}

# Groq API Configuration
GROQ_API_CONFIG = {
    'model': 'mixtral-8x7b-32768',
    'api_url': 'https://api.groq.com/openai/v1/chat/completions',
    'max_tokens': 32768,
    'temperature': 0.7,
    'top_p': 0.8,
    'chunk_settings': {
        'size': 8000,
        'overlap': 500,
        'max_parallel': 5
    }
}
