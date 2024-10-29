# Enhanced RAG System with Multi-Model Fallback

A robust Retrieval-Augmented Generation (RAG) system optimized for large document processing, using Mixtral-8x7B-32K as primary model with Gemini 1.5 Flash fallback. Designed for efficient handling of large documents while managing API costs and rate limits.

## 🌟 Features

### 🤖 Multi-Model Architecture
- **Primary Model: Mixtral-8x7B-32K via Groq**
  - 32K token context window
  - Efficient parallel processing
  - Production-ready performance
  - Better data privacy
  - Initial free credits ($10)
  - Optimized for large documents

- **Fallback Model: Gemini 1.5 Flash (Free Tier)**
  - Free backup option
  - 1M token context window
  - Rate limits:
    - 15 requests per minute
    - 1,500 requests per day
    - 1M tokens per minute

### 🔄 Smart Processing Pipeline
1. **Automatic Model Selection**
   - Documents > 32K tokens: Mixtral with chunking
   - Medium docs (8K-32K): Smart chunking
   - Small docs (<8K): Direct processing
   - Automatic fallback on failures

2. **Advanced Document Handling**
   - Dynamic chunk sizing
   - Context-preserving overlap
   - Parallel processing
   - Rate limit management
   - Daily usage tracking

3. **Error Recovery**
   - Automatic model fallback
   - Request persistence
   - Rate limit handling
   - Error logging

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### API Keys Setup
1. Get Groq API key from [Groq Console](https://console.groq.com)
2. Get Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
3. Create `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

### Basic Usage
```python
from main import RAGSystem

# Initialize system
rag = RAGSystem()

# Process a query
result = rag.process_query("Analyze this document", document="Your document text here")

# Access results
print(result['knowledge_retrieval'])  # Initial information
print(result['reasoning_steps'])      # Analysis steps
print(result['final_synthesis'])      # Final summary
```

## ⚙️ Configuration

### Model Settings (`config.py`)
```python
DOCUMENT_CONFIG = {
    'model_thresholds': {
        'small': 8000,    # Direct processing
        'medium': 16000,  # Smart chunking
        'large': 32000    # Full parallel
    },
    'chunk_settings': {
        'size': 8000,     # Tokens per chunk
        'overlap': 500,   # Context overlap
        'max_parallel': 5 # Parallel requests
    }
}
```

### Rate Limits
- **Mixtral (Groq)**
  - Based on available credits
  - No strict RPM limits
  - 32K tokens per request

- **Gemini Flash (Free)**
  - 15 RPM
  - 1,500 daily requests
  - 1M TPM (tokens per minute)

## 📊 Performance & Costs

### Processing Speed
- Small docs (<8K): 1-2 seconds
- Medium docs: 2-5 seconds
- Large docs: 5-15 seconds

### Cost Estimates
- **Mixtral**: $10 free credits
  - Input: $0.50/1M tokens
  - Output: $1.50/1M tokens

- **Gemini Flash**: Free tier
  - No cost for basic usage
  - Limited by rate restrictions

## 🛠️ Development

### Project Structure
```
MultiAgentRAG/
├── main.py           # Core logic
├── config.py         # Settings
├── requirements.txt  # Dependencies
├── .env             # API keys
└── README.md        # Documentation
```

### Key Components
- `HybridKnowledgeRetriever`: Model management
- `MixtralKnowledgeRetriever`: Primary processing
- `GeminiKnowledgeRetriever`: Fallback handling
- `RAGSystem`: Orchestration

## ⚠️ Limitations

### Mixtral (Primary)
- 32K token limit per request
- Credit-based usage
- Requires API key

### Gemini Flash (Fallback)
- Rate limits (15 RPM)
- Daily request cap
- Data used by Google

## 🔮 Roadmap
- [ ] Streaming responses
- [ ] Caching system
- [ ] Web interface
- [ ] Cost monitoring
- [ ] Performance analytics
- [ ] Batch processing

## 📫 Support
- [Report Issues](https://github.com/kongpop10/MultiAgentRAG/issues)
- [Request Features](https://github.com/kongpop10/MultiAgentRAG/issues)

## 📝 License
MIT License

---
Made with 💻 by [Kongpop]