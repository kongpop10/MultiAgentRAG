# O1 Multi-Agent RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system that combines Google's Gemini and Groq's AI models for comprehensive analysis and problem-solving.

## 🌟 Features

### 🤖 Multi-Model Architecture
- **Knowledge Retrieval** (Google Gemini Pro Experimental)
  - Advanced context understanding
  - Experimental model capabilities
  - Enhanced information extraction
  - Multi-modal potential

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
1. **Knowledge Phase** (Gemini)
   - Advanced query analysis
   - Experimental model features
   - Deep context building
   - Structured information retrieval

2. **Reasoning Phase** (Llama)
   - Dynamic step planning
   - Specialized agent assignment
   - Step-by-step execution
   - Inter-step context sharing

3. **Synthesis Phase** (Gemma)
   - Information integration
   - Key insight extraction
   - Recommendation generation

### 🛡️ Robust Error Handling
- Multi-API rate limit management
- Automatic retries with exponential backoff
- Cross-platform error recovery
- Comprehensive error logging

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Groq API access
- Google API access (Gemini)
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
# Add your API keys to .env:
# GROQ_API_KEY=your_groq_api_key
# GOOGLE_API_KEY=your_google_api_key
```

## 💻 Usage

```bash
# Run the system
python main.py

# Example interaction
Enter your query: What are the implications of quantum computing?

=== Knowledge Retrieval (Gemini) ===
[Advanced information retrieval using experimental model]

=== Reasoning Steps (Llama) ===
Step 1: Current State Analysis
Step 2: Impact Assessment
Step 3: Future Implications

=== Final Synthesis (Gemma) ===
[Comprehensive insights and recommendations]
```

## 🏗️ Project Structure
```
MultiAgentRAG/
├── main.py           # Core application logic
├── config.py         # Configuration and prompts
├── requirements.txt  # Dependencies
├── .env             # API keys (create from .env.example)
└── README.md        # Documentation
```

## ⚙️ Configuration

### Environment Variables
```env
GROQ_API_KEY=your_groq_api_key_here    # From console.groq.com
GOOGLE_API_KEY=your_google_api_key_here # From Google Cloud Console
```

### Model Configurations
Available in `config.py`:
- Gemini experimental settings
- Token limits
- Rate limit parameters
- System prompts
- Error handling settings

## ⚠️ Known Limitations
1. **API Rate Limits**
   - Groq: 6000 tokens per minute
   - Gemini: Based on quota
   - Automatic handling implemented

2. **Model Constraints**
   - Gemini: Experimental features
   - Context windows vary by model
   - Processing time varies with complexity

3. **API Dependencies**
   - Requires stable internet connection
   - Multiple API availability required
   - Experimental model stability

## 🔮 Ways to Improve
- [ ] Expand Gemini capabilities
- [ ] Enhanced error handling
- [ ] Performance optimizations
- [ ] Web interface
- [ ] API endpoint support
- [ ] Multi-modal input support

## 📫 Support & Contact
- [Report Issues](https://github.com/kongpop10/MultiAgentRAG/issues)
- [Request Features](https://github.com/kongpop10/MultiAgentRAG/issues)

## 📝 License
MIT License

---
[GitHub Repository](https://github.com/kongpop10/MultiAgentRAG) | [Report Bug](https://github.com/kongpop10/MultiAgentRAG/issues)