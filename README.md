# Query Decomposer Agent

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/Version-2.0.0-brightgreen.svg)](https://github.com/Yoda4547t/Query-Decomposer-Agent)

A powerful query decomposition system that offers two approaches for breaking down complex queries:

1. **LLM-based Decomposer** (New! Recommended) - Uses large language models (like LLaMA 3) for intelligent query decomposition
2. **Rule-based Decomposer** (Original) - Uses pattern matching and rule-based approaches

Both versions are designed to work seamlessly with RAG (Retrieval-Augmented Generation) AI systems and can be used based on your specific requirements.

## Features

### LLM-based Decomposer (New!)
- **Intelligent Decomposition** - Uses advanced language models for context-aware query breakdown
- **Natural Language Understanding** - Better handles complex, nuanced queries
- **Adaptive Sub-query Generation** - Automatically adjusts the number and type of sub-queries based on query complexity
- **RAG-Optimized** - Specifically designed to work with Retrieval-Augmented Generation systems
- **Ollama Integration** - Seamlessly works with Ollama for local model inference
- **Configurable** - Customize model, temperature, and other parameters

### Rule-based Decomposer (Original)
- **Domain-agnostic design** - Works with any domain (telecom, healthcare, finance, etc.)
- **Multiple decomposition strategies** - Rule-based, entity-based, and pattern matching
- **No external dependencies** - Uses only built-in Python libraries
- **Customizable entity patterns** - Easy to adapt for different domains
- **Deterministic** - Always produces the same output for the same input

## RAG Integration

Both decomposers are specifically designed to work with RAG AI systems where:

1. **Complex user queries** are decomposed into smaller, focused sub-queries
2. **Each sub-query** is sent to a retrieval system to find relevant documents
3. **Retrieved documents** are then processed by an LLM for final responses
4. **The agent handles** the query decomposition step in the RAG pipeline

### When to Use Which Version?

| Feature | LLM-based Decomposer | Rule-based Decomposer |
|---------|----------------------|----------------------|
| **Complexity Handling** | Excellent for complex, nuanced queries | Good for structured, predictable queries |
| **Customization** | High (model parameters, prompts) | Medium (rules and patterns) |
| **Performance** | Slower (requires model inference) | Faster (direct pattern matching) |
| **Deterministic** | No (can vary with temperature) | Yes |
| **Setup** | Requires Ollama/LMM setup | No external dependencies |
| **Best For** | Production systems needing high accuracy | Simple use cases, edge devices |

## Installation

### Prerequisites
- Python 3.7 or higher
- For LLM-based decomposer: [Ollama](https://ollama.ai/) installed and running

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Yoda4547t/Query-Decomposer-Agent.git
   cd Query-Decomposer-Agent
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   For LLM-based decomposer, also install:
   ```bash
   pip install requests langchain-community
   ```

3. (For LLM-based) Pull the desired model (e.g., LLaMA 3):
   ```bash
   ollama pull llama3:8b
   ```

## Using a virtual environment (.venv) on Windows

To keep dependencies isolated, use a per-project virtual environment.

- **Create and activate (.venv) in PowerShell**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  If activation is blocked, allow scripts for the current session:
  ```powershell
  Set-ExecutionPolicy -Scope Process RemoteSigned
  .\.venv\Scripts\Activate.ps1
  ```

- **Install dependencies into this venv**
  ```powershell
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  # For the LLM agent specifically (if not in requirements.txt):
  pip install requests langchain-community
  ```

- **Run the agents**
  ```powershell
  # Rule-based interactive demo
  python -c "from query_decomposer_agent import interactive_demo; interactive_demo()"

  # LLM-based agent (Ollama must be running: `ollama serve`)
  python query_decomposer_llm_agent.py --model llama3:8b
  ```

- **Deactivate and re-activate later**
  ```powershell
  deactivate
  .\.venv\Scripts\Activate.ps1
  ```

- **Troubleshooting**
  - Ensure you’re using the same Python for pip and execution:
    ```powershell
    where python
    python -c "import sys; print(sys.executable)"
    python -m pip -V
    ```
  - If `requests` is not found in the LLM agent:
    ```powershell
    python -m pip install requests
    ```

## Quick Start

### LLM-based Decomposer

```python
from query_decomposer_llm_agent import OllamaDecomposer

# Initialize with default model (llama3:8b)
decomposer = OllamaDecomposer()

# Decompose a complex query
result = decomposer("Compare 4G and 5G network performance in urban and rural areas")

# Print results
print(f"Original Query: {result.original_query}")
print("\nSub-queries:")
for i, subq in enumerate(result.sub_queries, 1):
    print(f"  {i}. {subq}")
```

### Rule-based Decomposer

```python
from query_decomposer_agent import QueryDecomposerAgent

# Create agent instance
agent = QueryDecomposerAgent()

# Process a complex query
result = agent.process_query("Compare AI and ML performance for ProductA and ProductB in last 6 months")

# View results
print(f"Original Query: {result.decomposition_result.original_query}")
print(f"Sub-queries ({len(result.decomposition_result.sub_queries)}):")
for i, sub_query in enumerate(result.decomposition_result.sub_queries, 1):
    print(f"  {i}. {sub_query}")
```

### RAG Integration Example

```python
from query_decomposer_agent import QueryDecomposerAgent

# Create agent
agent = QueryDecomposerAgent()

# Set up your retriever callback
def my_retriever_callback(retriever_request):
    # Your retriever logic here
    return {"documents": ["doc1", "doc2"], "scores": [0.9, 0.8]}

# Set up your LLM callback
def my_llm_callback(sub_queries, context=None):
    # Your LLM logic here
    return {"response": "Generated response based on sub-queries"}

# Connect callbacks
agent.set_retriever_callback(my_retriever_callback)
agent.set_llm_callback(my_llm_callback)

# Process query through RAG pipeline
result = agent.process_query("Your complex query here")
retriever_response = agent.send_to_retriever(result.retriever_request)
llm_response = agent.send_to_llm(result.decomposition_result.sub_queries)
```

## 📋 Usage Examples

### Example 1: Basic Query Decomposition

```python
from query_decomposer_agent import QueryDecomposerAgent

agent = QueryDecomposerAgent()
result = agent.process_query("Compare network latency between 4G and 5G in Bangalore for last 3 months")

print(f"Query Type: {result.decomposition_result.query_type.value}")
print(f"Confidence: {result.decomposition_result.confidence_score:.2f}")
print(f"Sub-queries: {result.decomposition_result.sub_queries}")
```

### Example 2: Custom Entity Configuration

```python
from query_decomposer_agent import QueryDecomposerAgent

# Create agent with custom entities
custom_entities = {
    'technologies': ['AI', 'ML', 'DL', 'NLP', 'CV'],
    'locations': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai'],
    'metrics': ['performance', 'latency', 'throughput', 'accuracy']
}

agent = QueryDecomposerAgent()
agent.configure_entities(custom_entities)

result = agent.process_query("Compare AI and ML performance in Mumbai and Delhi")
```

### Example 3: Full RAG Integration

```python
from query_decomposer_agent import QueryDecomposerAgent, demo_rag_integration

# Run the complete RAG integration demo
demo_rag_integration()
```

## 🎮 Interactive Demo

Run the interactive demo to test the agent:

```python
from query_decomposer_agent import interactive_demo

interactive_demo()
```

## 🔗 Integration Points

The agent provides clear integration points for your RAG system:

- **`agent.process_query()`** - Where raw user queries first arrive
- **`agent.send_to_retriever()`** - Where decomposed queries should be sent back
- **`agent.set_retriever_callback()`** - Set your retriever integration
- **`agent.set_llm_callback()`** - Set your LLM integration

## 🛠️ Implementation Guide

1. **Look for `>>> USER IMPLEMENTATION` comments** in the code
2. **Replace placeholder functions** with your actual implementations
3. **Set up your input/output file paths** as indicated
4. **Configure your retriever and LLM integrations**
5. **Test with your domain-specific entities and queries**

## 📁 File Structure

```
Query-Decomposer-Agent/
├── query_decomposer_agent.py    # Main agent implementation
├── README.md                    # This file
├── requirements.txt             # Dependencies (none required)
├── LICENSE                      # MIT License
└── .gitignore                  # Git ignore file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Yoda4547t** - [GitHub Profile](https://github.com/Yoda4547t)

## 🙏 Acknowledgments

- Built for RAG (Retrieval-Augmented Generation) AI systems
- Designed to be domain-agnostic and easily customizable
- No external dependencies for maximum portability

## 🧠 LLM-based Query Decomposer Agent (LangChain + LLM)

This agent uses LangChain and an LLM backend (Ollama by default, swappable) to decompose complex queries into sub-queries using an LLM prompt.

- **File:** `query_decomposer_llm_agent.py`
- **Default LLM:** Ollama (llama3), but you can swap in OpenAI, Anthropic, etc.
- **LangChain Tool:** `QuerySplitterLLM` (modular, extensible)

### Usage Example

```python
from query_decomposer_llm_agent import QueryDecomposerLLMAgent

agent = QueryDecomposerLLMAgent()  # Uses Ollama/llama3 by default
query = "Compare the revenue growth of Apple and Microsoft over the last 5 years and explain the key differences."
result = agent.decompose(query)
print(result)
```

**Sample Output:**
```json
{
  "original_query": "Compare the revenue growth of Apple and Microsoft over the last 5 years and explain the key differences.",
  "sub_queries": [
    "Revenue growth of Apple over the last 5 years",
    "Revenue growth of Microsoft over the last 5 years",
    "Key differences in revenue growth between Apple and Microsoft over the last 5 years"
  ]
}
```

### Extensibility
- Swap in any LangChain-compatible LLM (OpenAI, Anthropic, etc.)
- Add more tools/agents (Retriever, Summarizer, etc.) as needed

### Testing
Run the file directly to see test queries and JSON output:
```bash
python query_decomposer_llm_agent.py
```