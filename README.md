# Query Decomposer Agent

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![No Dependencies](https://img.shields.io/badge/Dependencies-None-green.svg)](https://github.com/Yoda4547t/Query-Decomposer-Agent)

A general-purpose agent that decomposes complex queries into smaller sub-queries using rule-based and pattern matching approaches. Designed to work seamlessly with RAG (Retrieval-Augmented Generation) AI systems.

## 🚀 Features

- **Domain-agnostic design** - Works with any domain (telecom, healthcare, finance, etc.)
- **Multiple decomposition strategies** - Rule-based, entity-based, and intelligent decomposition
- **RAG Integration Ready** - Perfect for RAG AI systems
- **Dynamic sub-query generation** - Generates 5+ sub-queries based on query complexity
- **No external dependencies** - Uses only built-in Python libraries
- **Customizable entity patterns** - Easy to adapt for different domains
- **Clear integration points** - Well-defined callbacks for retriever and LLM integration

## 🔧 RAG Integration

This agent is specifically designed to work with RAG AI systems where:

1. **Complex user queries** are decomposed into smaller, focused sub-queries
2. **Each sub-query** is sent to a retrieval system to find relevant documents
3. **Retrieved documents** are then processed by an LLM for final responses
4. **The agent handles** the query decomposition step in the RAG pipeline

## 📦 Installation

No external dependencies required! This agent uses only built-in Python libraries.

```bash
git clone https://github.com/Yoda4547t/Query-Decomposer-Agent.git
cd Query-Decomposer-Agent
```

## ⚡ Quick Start

### Basic Usage

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