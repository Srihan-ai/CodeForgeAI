# 🔥 CodeForge AI

**Intelligent Multi-Agent Code Generation Platform**

CodeForge AI is a powerful code generation system that uses LangGraph's multi-agent architecture to create, test, and self-correct code across multiple programming languages.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- **🤖 Multi-Agent Architecture**: Specialized agents for code generation, testing, and validation
- **🔄 Self-Correcting**: Automatically detects and fixes errors through iterative refinement
- **🌐 Multi-Language Support**: Generate Python, Java, and C++ code
- **⚡ Real-Time Streaming**: Watch your code being generated live
- **🛡️ Built-in Guardrails**: Safety checks for input and output
- **💎 Modern UI**: Sleek, futuristic interface with glassmorphism design
- **📦 Production-Ready**: FastAPI backend with comprehensive error handling

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Groq API key (free at [console.groq.com](https://console.groq.com/keys))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Srihan-ai/CodeForgeAI.git
cd CodeForgeAI
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

5. **Run the application**
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

6. **Open your browser**
```
http://localhost:8000
```

---

## 🎯 Usage

### Web Interface

1. Enter a code description (e.g., "Create a binary search tree")
2. Select target language (Python, Java, or C++)
3. Set max iterations (1-5)
4. Click "Forge Code"
5. Copy or download the generated code

### API Endpoints

#### Generate Code
```bash
POST /generate
Content-Type: application/json

{
  "task": "Write a function to reverse a string",
  "language": "python",
  "max_iterations": 3
}
```

#### Stream Generation
```bash
GET /stream?task=Create a linked list&language=python&max_iterations=3
```

#### Health Check
```bash
GET /health
```

---

## 🏗️ Architecture

CodeForge AI uses a multi-agent workflow powered by LangGraph:

```
User Input → Guardrails → Developer Agent → Tester Agent → Router
                              ↑                              ↓
                              └──────── (if tests fail) ─────┘
```

### Agents

- **Guardrails Node**: Validates input safety and relevance
- **Developer Agent**: Generates clean, compilable code
- **Tester Agent**: Creates and executes test cases
- **Router**: Decides whether to retry or complete

---

## 🛠️ Configuration

### Environment Variables

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
REDIS_URL=redis://localhost:6379  # For session persistence
```

### Supported Languages

| Language | Version | File Extension |
|----------|---------|----------------|
| Python   | 3.11    | .py            |
| Java     | 17      | .java          |
| C++      | 20      | .cpp           |

---

## 📊 Example

**Input:**
```
Task: Create a function to calculate fibonacci numbers
Language: Python
Max Iterations: 3
```

**Output:**
```python
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

if __name__ == "__main__":
    assert fibonacci(7) == 13
    print(f"fibonacci(7) = {fibonacci(7)}")
```

---

## 🔒 Security Features

- **Input Validation**: Prevents prompt injection and malicious code
- **Output Scanning**: Detects dangerous patterns and PII leaks
- **Isolated Execution**: Code runs in sandboxed environment
- **Rate Limiting**: Prevents API abuse

---

## 🚢 Deployment

### Render

1. Push code to GitHub
2. Create new Web Service on Render
3. Connect repository
4. Add environment variable: `GROQ_API_KEY`
5. Deploy

### Docker

```bash
docker build -t codeforge-ai .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key codeforge-ai
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [Groq](https://groq.com/)
- UI inspired by modern design systems

---

## 📧 Contact

Created by **Srihan AI**

- GitHub: [@Srihan-ai](https://github.com/Srihan-ai)
- Project Link: [https://github.com/Srihan-ai/CodeForgeAI](https://github.com/Srihan-ai/CodeForgeAI)

---

**Made with 🔥 by Srihan AI**
