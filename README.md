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
   git clone https://github.com/yourusername/o1-multi-agent-rag.git
   cd o1-multi-agent-rag
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
   - Create `.env` file in project root
   - Add your Groq API key:
     ```env
     GROQ_API_KEY=your_api_key_here
     ```

## 💻 Usage

### Basic Operation

## 📦 Dependencies
- `requests`: HTTP requests handling
- `python-dotenv`: Environment variable management
- `typing-extensions`: Type hinting support


## 🏗️ Project Structure
o1-multi-agent-rag/
├── main.py # Core application logic
├── config.py # Configuration and prompts
├── requirements.txt # Project dependencies
├── .env # Environment variables (not tracked)
├── .gitignore # Git ignore rules
└── README.md # Documentation

## 📦 Requirements

txt
requests==2.31.0
python-dotenv==1.0.0
typing-extensions==4.9.0

## ⚙️ Configuration

### Environment Variables
Required variables in `.env`:

env
GROQ_API_KEY=your_api_key_here

### Model Configurations
Available in `config.py`:
- Token limits
- Rate limit parameters
- System prompts
- Error handling settings

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes
   ```bash
   git commit -m 'Add AmazingFeature'
   ```
4. Push to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Groq AI for their powerful models and API
- Open source community
- All contributors

## 📫 Support & Contact
- GitHub Issues for bug reports
- Pull Requests for contributions
- Discussions for questions

## 🔮 Roadmap
- [ ] Additional model support
- [ ] Enhanced error handling
- [ ] Performance optimizations
- [ ] Extended documentation
- [ ] Testing suite
- [ ] CI/CD integration
- [ ] Web interface
- [ ] API endpoint support

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

## 🔍 Troubleshooting
Common issues and solutions:
1. Rate limit errors: System automatically handles with backoff
2. Token limits: Implemented automatic truncation
3. API errors: Comprehensive error handling included

## 📊 Performance
- Average response time: 2-5 seconds
- Token usage optimization
- Batch processing capabilities

---
Made with ❤️ by [Your Name/Organization]

[GitHub Repository](https://github.com/kongpop10/MultiAgentRAG) | [Report Bug](https://github.com/kongpop10/MultiAgentRAG/issues) | [Request Feature](https://github.com/kongpop10/MultiAgentRAG/issues)