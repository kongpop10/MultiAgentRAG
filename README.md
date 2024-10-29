# O1 Multi-Agent RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system that combines Google's Gemini Pro and Groq's AI models for comprehensive analysis and problem-solving, with advanced document handling capabilities.

## 🌟 Features

### 🤖 Multi-Model Architecture
- **Knowledge Retrieval** (Google Gemini Pro)
  - Parallel document processing
  - Advanced chunking with overlap
  - Asynchronous operations
  - Adaptive model selection
  - Auto-scaling based on document size

- **Dynamic Reasoning** (Llama-3.1-70B)
  - Multi-step reasoning pipeline
  - Context-aware analysis
  - Sequential logic processing
  - Dynamic step generation

- **Final Synthesis** (Gemma-7B)
  - Comprehensive insight integration
  - Structured recommendations
  - Context preservation
  - Clear action items

### 🔄 Hybrid Processing Pipeline
1. **Document Analysis**
   - Automatic size detection
   - Smart chunking strategy
   - Parallel processing
   - Context preservation

2. **Knowledge Extraction**
   - Model selection based on complexity
   - Asynchronous chunk processing
   - Context merging
   - Intelligent synthesis

3. **Reasoning & Synthesis**
   - Multi-step analysis
   - Cross-reference validation
   - Coherent output generation

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Groq API access
Google API access (Gemini Pro)
```

### Installation
```bash
# Clone and setup
git clone https://github.com/kongpop10/MultiAgentRAG.git
cd MultiAgentRAG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your API keys to .env
```

## 💻 Usage

### Basic Query
```python
python main.py
Enter your query: Analyze the impact of AI on healthcare
```

### Document Analysis
```python
# The system automatically handles:
- Small documents (<4000 tokens): Direct processing
- Medium documents: Parallel chunk processing
- Large documents (>8192 tokens): Mixtral model
```

## 🔧 Advanced Features

### Parallel Processing
- Automatic thread pool management
- Configurable chunk sizes
- Overlap for context preservation
- Asynchronous operations

### Error Handling
- Rate limit management
- Automatic retries
- Error recovery
- Comprehensive logging

### Document Processing
- Smart chunking
- Context preservation
- Parallel execution
- Adaptive model selection

## ⚙️ Configuration

### Environment Setup
```env
GROQ_API_KEY=your_groq_api_key    # From console.groq.com
GOOGLE_API_KEY=your_google_api_key # From Google Cloud Console
```

### Customizable Parameters
```python
# config.py
- Chunk sizes
- Token limits
- Parallel requests
- Safety settings
- Processing thresholds
```

## 📊 Performance

### Processing Capabilities
- Small docs: < 2 seconds
- Medium docs: 2-5 seconds
- Large docs: 5-10 seconds

### Optimization Features
- Parallel processing
- Async operations
- Smart caching
- Rate limit optimization

## 🛠️ Development

### Project Structure
```
MultiAgentRAG/
├── main.py           # Core logic & API integration
├── config.py         # Settings & configurations
├── requirements.txt  # Dependencies
├── .env             # API keys (from .env.example)
└── README.md        # Documentation
```

### Key Components
- Hybrid Knowledge Retriever
- Reasoning Orchestrator
- Enhanced Gemini Integration
- Parallel Processing Engine

## 🔮 Future Enhancements
- [ ] Streaming responses
- [ ] Web interface
- [ ] API endpoints
- [ ] Document caching
- [ ] Enhanced parallelization
- [ ] Memory optimization

## ⚠️ Limitations & Considerations
1. **API Constraints**
   - Rate limits vary by tier
   - Token limitations
   - Response time variation

2. **Resource Usage**
   - Memory for large documents
   - CPU for parallel processing
   - Network bandwidth

## 📫 Support
- [Report Issues](https://github.com/kongpop10/MultiAgentRAG/issues)
- [Feature Requests](https://github.com/kongpop10/MultiAgentRAG/issues)
- [Documentation](https://github.com/kongpop10/MultiAgentRAG/wiki)

## 📝 License
MIT License

---
Made with 💻 by [Kongpop]

[Repository](https://github.com/kongpop10/MultiAgentRAG) | [Issues](https://github.com/kongpop10/MultiAgentRAG/issues)