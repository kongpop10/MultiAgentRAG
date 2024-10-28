# O1 Multi-Agent RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system that combines multiple Groq AI models with dynamic reasoning capabilities for in-depth analysis and problem-solving.

## 🌟 Features

### 🤖 Multi-Model Architecture
- **Knowledge Retrieval** (Mixtral-8x7B-32K)
  - Comprehensive information extraction
  - Context-aware retrieval
  - 32K token context window

- **Dynamic Reasoning** (Llama-3.1-70B)
  - Automated step planning
  - Multiple specialized reasoning agents
  - Sequential logic processing
  - Context inheritance between steps

- **Final Synthesis** (Gemma-7B)
  - Coherent integration of insights
  - Actionable recommendations
  - Concise summaries

### 🔄 Intelligent Processing Pipeline
1. **Knowledge Phase**
   - Query analysis
   - Information extraction
   - Context building

2. **Reasoning Phase**
   - Dynamic step planning
   - Specialized agent assignment
   - Step-by-step execution
   - Inter-step context sharing

3. **Synthesis Phase**
   - Information integration
   - Key insight extraction
   - Recommendation generation

### 🛡️ Robust Error Handling
- Rate limit management
- Automatic retries with exponential backoff
- Graceful error recovery
- Comprehensive error logging

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Groq API access
- Virtual environment tool

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/kongpop10/MultiAgentRAG.git
cd MultiAgentRAG
```

2. **Set Up Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment**
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

## 💻 Usage

```bash
# Run the system
python main.py

# Example interaction
Enter your query: What are the implications of quantum computing?

=== Knowledge Retrieval ===
[System retrieves comprehensive information]

=== Reasoning Steps ===
Step 1: Current State Analysis
Step 2: Impact Assessment
Step 3: Future Implications

=== Final Synthesis ===
[System provides actionable insights]
```

## 🏗️ Project Structure
```
MultiAgentRAG/
├── main.py           # Core application logic
├── config.py         # Configuration and prompts
├── requirements.txt  # Dependencies
├── .env             # API key (create from .env.example)
└── README.md        # Documentation
```

## ⚙️ Configuration

### Environment Variables
```env
GROQ_API_KEY=your_api_key_here  # Get from console.groq.com
```

### Model Configurations
Available in `config.py`:
- Token limits
- Rate limit parameters
- System prompts
- Error handling settings

## ⚠️ Known Limitations
1. **Rate Limits**
   - 6000 tokens per minute
   - Automatic handling implemented

2. **Model Constraints**
   - Context windows vary by model
   - Processing time varies with complexity

3. **API Dependence**
   - Requires stable internet connection
   - Subject to API availability

## 🔮 Ways to Improve
- [ ] Additional model support
- [ ] Enhanced error handling
- [ ] Performance optimizations
- [ ] Web interface
- [ ] API endpoint support

## 📫 Support & Contact
- [Report Issues](https://github.com/kongpop10/MultiAgentRAG/issues)
- [Request Features](https://github.com/kongpop10/MultiAgentRAG/issues)

## 📝 License
MIT License

---
[GitHub Repository](https://github.com/kongpop10/MultiAgentRAG) | [Report Bug](https://github.com/kongpop10/MultiAgentRAG/issues)