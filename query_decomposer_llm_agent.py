import json
import re
import requests
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable, Union, TypedDict
from pathlib import Path

@dataclass
class RAGContext:
    """Holds context information for RAG (Retrieval-Augmented Generation).
    
    Attributes:
        documents: List of relevant document chunks
        scores: Relevance scores for each document
        metadata: Additional metadata about the retrieval
    """
    documents: List[str]
    scores: List[float]
    metadata: Dict[str, Any] = None

# Optional RAG support
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

@dataclass
class QueryDecompositionResult:
    """Holds the result of query decomposition.
    
    Attributes:
        original_query (str): The original input query
        sub_queries (List[str]): List of decomposed sub-queries
        metadata (Dict[str, Any]): Additional metadata about the decomposition
    """
    original_query: str
    sub_queries: List[str]
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "original_query": self.original_query,
            "sub_queries": self.sub_queries,
            "metadata": self.metadata or {}
        }
    
    def __str__(self) -> str:
        """Return a human-readable string representation."""
        result = [
            f"Original Query: {self.original_query}",
            "\nDecomposed Sub-queries:"
        ]
        for i, subq in enumerate(self.sub_queries, 1):
            result.append(f"  {i}. {subq}")
            
        if self.metadata:
            result.append("\nMetadata:")
            for k, v in self.metadata.items():
                result.append(f"  - {k}: {v}")
                
        return "\n".join(result)

class BaseQueryDecomposer(ABC):
    """Abstract base class for query decomposers."""
    
    @abstractmethod
    def decompose(self, query: str, **kwargs) -> QueryDecompositionResult:
        """Decompose a query into sub-queries.
        
        Args:
            query: The input query to decompose
            **kwargs: Additional keyword arguments for the decomposition
            
        Returns:
            QueryDecompositionResult: The decomposition result
        """
        pass
    
    def __call__(self, query: str, **kwargs) -> QueryDecompositionResult:
        """Make the decomposer callable."""
        return self.decompose(query, **kwargs)


class OllamaDecomposer(BaseQueryDecomposer):
    """Query decomposer using Ollama models."""
    
    def __init__(
        self, 
        model_name: str = "llama3:8b", 
        temperature: float = 0.1,
        base_url: str = "http://localhost:11434"
    ):
        """Initialize the Ollama decomposer.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Temperature for generation (0-1)
            base_url: Base URL of the Ollama API
        """
        self.model_name = model_name
        self.temperature = min(max(temperature, 0), 1)  # Clamp to [0, 1]
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/generate"
    
    def _call_model(self, prompt: str) -> str:
        """Call the Ollama API with the given prompt."""
        print(f"⏳ Sending request to Ollama model '{self.model_name}'... (this may take up to 2 minutes)")
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False
                },
                timeout=120  # Increased timeout to 120 seconds for local LLMs
            )
            response.raise_for_status()
            print("✅ Received response from Ollama.")
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"❌ Error communicating with Ollama: {e}")
            raise RuntimeError(f"Failed to call Ollama API: {str(e)}")
    
    def decompose(self, query: str, **kwargs) -> QueryDecompositionResult:
        """Decompose a query using the Ollama model."""
        # Simplified prompt to generate fewer, more concise sub-queries.
        system_prompt = """You are an expert at breaking down complex queries into simpler sub-queries. 
        Your task is to decompose the given query into 2-4 smaller, independent questions that can be answered separately.
        
        Guidelines:
        1. Each sub-query should be self-contained and answerable on its own.
        2. Maintain the original meaning and intent of the query.
        3. Break down complex questions into logical components.
        4. Number each sub-query.
        
        Example:
        Input: "What are the differences between Python and Java in terms of performance and syntax?"
        Output:
        1. How does Python's performance compare to Java's?
        2. What are the key syntax differences between Python and Java?
        
        Now, decompose this query:
        """
        
        try:
            response = self._call_model(f"{system_prompt}\n\n{query}")
            
            # Simplified parsing for numbered lists.
            sub_queries = []
            for line in response.split('\n'):
                line = line.strip()
                if line and line[0].isdigit() and '.' in line:
                    sub_query = line[line.find('.')+1:].strip()
                    if sub_query:
                        sub_queries.append(sub_query)
            
            # If no sub-queries found, use the original query as a fallback.
            if not sub_queries:
                sub_queries = [query]
            
            return QueryDecompositionResult(
                original_query=query,
                sub_queries=sub_queries,
                metadata={
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "decomposer": "ollama"
                }
            )
            
        except Exception as e:
            return QueryDecompositionResult(
                original_query=query,
                sub_queries=[query],
                metadata={
                    "error": str(e),
                    "decomposer": "ollama",
                    "fallback": True
                }
            )


class RAGDecomposer(BaseQueryDecomposer):
    """Query decomposer with RAG support."""
    
    def __init__(
        self, 
        base_decomposer: BaseQueryDecomposer,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Initialize the RAG decomposer.
        
        Args:
            base_decomposer: Base decomposer to use
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        if not RAG_AVAILABLE:
            raise ImportError(
                "RAG dependencies not found. Install with: pip install langchain-community"
            )
            
        self.base_decomposer = base_decomposer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None
        self.embeddings = OllamaEmbeddings()
    
    def add_documents(self, documents: List[Any]) -> None:
        """Add documents to the vector store."""
        if not documents:
            return
            
        # Convert to Document objects if needed
        docs = []
        for doc in documents:
            if not isinstance(doc, str):
                docs.append(doc)
            else:
                docs.append(Document(page_content=doc))
        
        # Chunk documents if needed
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        split_docs = text_splitter.split_documents(docs)
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        else:
            self.vector_store.add_documents(split_docs)
    
    def decompose(self, query: str, k: int = 3, **kwargs) -> QueryDecompositionResult:
        """Decompose a query with RAG context."""
        # First get base decomposition
        result = self.base_decomposer.decompose(query, **kwargs)
        
        if self.vector_store is None:
            return result
        
        # Get relevant context for each sub-query
        contexts = {}
        for i, subq in enumerate(result.sub_queries):
            docs = self.vector_store.similarity_search(subq, k=k)
            contexts[str(i)] = [doc.page_content for doc in docs]
        
        # Update metadata with context info
        if result.metadata is None:
            result.metadata = {}
        result.metadata["rag_context"] = contexts
        result.metadata["rag_k"] = k
        
        return result
    
    def generate(self, prompt: str) -> str:
        """Generate a response from the Ollama model"""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"Error calling Ollama API: {str(e)}")
            return ""
    
    def decompose(self, query: str, **kwargs) -> Dict[str, Any]:
        """Decompose a query into sub-queries"""
        prompt = f"""
        Your task is to break down the following complex query into simpler, more manageable sub-queries.
        Return the sub-queries as a numbered list. Each sub-query should be complete and self-contained.
        
        Complex Query: {query}
        
        Sub-queries:
        1."""
        
        response = self.generate(prompt)
        
        # Extract the sub-queries from the response
        subqueries = []
        if response:
            # Try to find numbered list items
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and line[0].isdigit() and '.' in line:
                    # Extract the text after the number and period
                    subquery = line[line.find('.')+1:].strip()
                    if subquery:
                        subqueries.append(subquery)
        
        # If no subqueries were found, return the original query as a single item
        subqueries = subqueries if subqueries else [query]
        
        return {
            "original_query": query,
            "sub_queries": subqueries,
            "metadata": {"model": self.model_name}
        }

    def _chunk_documents(self, documents: List[Any]) -> List[Any]:
        if not RAG_AVAILABLE:
            return documents
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def add_documents(self, documents: List[Any]) -> None:
        if not self.use_rag or not RAG_AVAILABLE:
            raise ValueError("RAG is not enabled or dependencies are missing. Set use_rag=True and install langchain.")
            
        if not documents:
            return
            
        if any(len(getattr(doc, 'page_content', str(doc))) > self.chunk_size for doc in documents):
            documents = self._chunk_documents(documents)
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        
        self.retriever = self.vector_store.as_retriever()

    def _get_rag_context(self, query: str, k: int = 5) -> RAGContext:
        if not hasattr(self, 'retriever') or not self.retriever:
            return RAGContext(documents=[], scores=[])
            
        docs = self.retriever.get_relevant_documents(query, k=k)
        
        # Extract page content and scores if available
        documents = []
        scores = []
        
        for doc in docs:
            content = getattr(doc, 'page_content', str(doc))
            score = getattr(doc, 'score', 1.0)  # Default score if not available
            documents.append(content)
            scores.append(score)
            
        return RAGContext(documents=documents, scores=scores)

    def _get_system_prompt(self, is_rag: bool = False) -> str:
        base_prompt = """You are a query decomposition assistant. Your task is to break down complex user queries into simpler, self-contained sub-queries.

RULES:
1. Each sub-query should be independent and answerable on its own
2. Maintain the original intent and context of the main query
3. Include all necessary context in each sub-query
4. Make sub-queries as specific as possible
5. Order sub-queries logically when there are dependencies
6. Never include any explanations or additional text outside the JSON"""

        if is_rag:
            base_prompt += """

When decomposing queries for RAG:
- Focus on creating sub-queries that can be answered by retrieved documents
- Include key terms that would help retrieve relevant documents
- If comparing entities, create separate sub-queries for each aspect"""

        base_prompt += """

OUTPUT FORMAT (JSON):
{
  "original_query": "original query here",
  "sub_queries": ["sub-query 1", "sub-query 2", "..."]
}"""
        return base_prompt

    def _initialize_agent(self):
        def query_splitter_llm(query: str) -> str:
            if not query or not query.strip():
                return json.dumps({"original_query": "", "sub_queries": []})
                
            try:
                system_prompt = self._get_system_prompt(is_rag=self.use_rag)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Decompose this query: {query}"}
                ]
                
                formatted_messages = self.llm_provider.format_messages(messages)
                response = self.llm.invoke(formatted_messages)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                cleaned = re.sub(r'```json|```', '', response_text.strip())
                cleaned = re.sub(r'^[^{]*{', '{', cleaned, flags=re.DOTALL)
                cleaned = re.sub(r'}[^}]*$', '}', cleaned, flags=re.DOTALL)
                
                data = json.loads(cleaned)
                if not isinstance(data, dict) or 'sub_queries' not in data:
                    raise ValueError("Invalid response format")
                    
                if not isinstance(data['sub_queries'], list) or not all(isinstance(sq, str) for sq in data['sub_queries']):
                    raise ValueError("sub_queries must be a list of strings")
                    
                return json.dumps(data)
                
            except Exception as e:
                return json.dumps({
                    "original_query": query,
                    "sub_queries": [query],
                    "error": f"Error: {str(e)}"
                })

        tool = Tool(
            name="QuerySplitterLLM",
            description="Decompose a complex query into smaller, self-contained sub-queries.",
            func=query_splitter_llm,
            return_direct=True,
            handle_tool_error=True
        )
        
        return initialize_agent(
            tools=[tool],
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=self.max_iterations,
            early_stopping_method="generate"
        )

    def decompose(
        self, 
        query: str,
        use_rag: Optional[bool] = None,
        **kwargs
    ) -> QueryDecompositionResult:
        use_rag = use_rag if use_rag is not None else self.use_rag
        context = None
        
        if use_rag and not hasattr(self, 'retriever'):
            raise ValueError("No retriever available. Call add_documents() first or disable RAG.")
            
        if use_rag:
            context = self._get_rag_context(query, k=kwargs.get('k', 5))
        
        try:
            result = self.agent.run({
                "input": f"Decompose this query: {query}",
                "chat_history": []
            })
            
            if isinstance(result, str):
                result = re.sub(r'^.*?{', '{', result, flags=re.DOTALL)
                result = re.sub(r'}[^}]*$', '}', result, flags=re.DOTALL)
                data = json.loads(result)
                
                if not isinstance(data, dict) or 'sub_queries' not in data:
                    raise ValueError("Invalid response format")
            else:
                data = result
            
            return QueryDecompositionResult(
                original_query=data.get('original_query', query),
                sub_queries=data['sub_queries'],
                context=context,
                metadata={
                    "rag_used": use_rag,
                    "sub_query_count": len(data['sub_queries'])
                }
            )
            
        except Exception as e:
            return QueryDecompositionResult(
                original_query=query,
                sub_queries=[query],
                context=context,
                metadata={
                    "rag_used": use_rag,
                    "error": str(e)
                }
            )

def print_test_results(query: str, result):
    """Print test results in a readable format."""
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    if not result:
        print("❌ No result returned")
        return
        
    if isinstance(result, QueryDecompositionResult):
        result = result.dict()
    
    print("\n🔍 DECOMPOSED SUB-QUERIES:")
    for i, subq in enumerate(result.get('sub_queries', []), 1):
        print(f"  {i}. {subq}")
    
    if 'metadata' in result and result['metadata']:
        print("\n📊 METADATA:")
        for k, v in result['metadata'].items():
            print(f"  - {k}: {v}")
    
    print("\n" + "-"*80)

def run_tests(decomposer: BaseQueryDecomposer, queries: List[str] = None):
    """Run test queries through the decomposer.
    
    Args:
        decomposer: The query decomposer to test
        queries: List of test queries (uses default if None)
    """
    if queries is None:
        queries = [
            "What are the main differences between Python 3.8 and 3.9?",
            "Compare the performance of different machine learning algorithms on image classification",
            "Explain the key features of quantum computing and its potential applications"
        ]
    
    print("\n🔍 Running test queries...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"Test {i}: {query}")
        try:
            result = decomposer.decompose(query)
            print(f"✅ Decomposed into {len(result.sub_queries)} sub-queries:")
            for j, subq in enumerate(result.sub_queries, 1):
                print(f"   {j}. {subq}")
            
            if result.metadata and 'error' in result.metadata:
                print(f"⚠️  Warning: {result.metadata['error']}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        print()  # Add space between tests

def interactive_mode(decomposer: BaseQueryDecomposer):
    """Run the decomposer in interactive mode.
    
    Args:
        decomposer: The query decomposer to use
    """
    print("\n🔍 Interactive Query Decomposer")
    print("=" * 50)
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            user_input = input("Enter your query: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Goodbye!")
                break
                
            if not user_input:
                continue
                
            print("\n🔄 Processing...")
            
            try:
                result = decomposer.decompose(user_input)
                print("\n" + str(result))
                
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
            
            print("\n" + "-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


def check_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running.
    
    Args:
        base_url: Base URL of the Ollama API
        
    Returns:
        bool: True if the server is running, False otherwise
    """
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Query Decomposer with LLM')
    
    # Model configuration
    parser.add_argument("--model", type=str, default="llama3:8b",
                        help="Ollama model to use (default: llama3:8b)")
    parser.add_argument('--temperature', type=float, default=0.3,
                      help='Temperature for generation (0-1, default: 0.3)')
    parser.add_argument('--base-url', type=str, default='http://localhost:11434',
                      help='Base URL for the model API (default: http://localhost:11434)')
    
    # RAG configuration
    rag_group = parser.add_argument_group('RAG options')
    rag_group.add_argument('--use-rag', action='store_true',
                         help='Enable RAG functionality (requires langchain)')
    rag_group.add_argument('--chunk-size', type=int, default=1000,
                         help='Chunk size for RAG (default: 1000)')
    rag_group.add_argument('--chunk-overlap', type=int, default=200,
                         help='Chunk overlap for RAG (default: 200)')
    
    # Runtime options
    parser.add_argument('--test', action='store_true',
                      help='Run test queries instead of interactive mode')
    parser.add_argument('--query', type=str,
                      help='Process a single query and exit')
    
    return parser.parse_args()

def test_retrieval_optimization():
    """Test function to demonstrate retrieval-optimized decomposition."""
    print("🧪 Testing General-Purpose Retrieval-Optimized LLM Query Decomposer")
    print("=" * 70)
    
    # Test queries covering different domains and types
    test_queries = [
        "explain why 5G is better than 4G",
        "what is difference between 5G and 4G compared in last 3 months in goa",
        "best restaurants in Paris for vegetarian food",
        "how to learn Python programming for beginners",
        "compare machine learning and traditional algorithms for data analysis",
        "what are the health benefits of yoga and meditation",
        "climate change effects on agriculture in developing countries",
        "investment strategies for retirement planning"
    ]
    
    try:
        decomposer = OllamaDecomposer(temperature=0.2)  # Lower temperature for more focused output
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: {query}")
            print("-" * 60)
            
            try:
                result = decomposer.decompose(query)
                print(f"✅ Generated {len(result.sub_queries)} retrieval-optimized sub-queries:")
                
                for j, sub_query in enumerate(result.sub_queries, 1):
                    print(f"  {j}. {sub_query}")
                    
                print(f"\n📊 Optimization: {result.metadata.get('optimization', 'N/A')}")
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                
    except Exception as e:
        print(f"❌ Failed to initialize decomposer: {e}")
        print("\n💡 Make sure Ollama is running locally on port 11434")
        print("   Install: https://ollama.ai/")
        print("   Run: ollama serve")
        print("   Pull a model: ollama pull llama3:8b")

def main():
    """Main entry point for the query decomposer."""
    print("🚀 Starting Query Decomposer")
    print("=" * 50)
    
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Initialize the base decomposer
        if not check_ollama_running(args.base_url):
            print("\n❌ Error: Ollama server is not running.")
            print(f"   Please start it first with: ollama serve")
            print(f"   Then make sure you have the model: ollama pull {args.model}")
            return 1
        
        decomposer = OllamaDecomposer(
            model_name=args.model,
            temperature=args.temperature,
            base_url=args.base_url
        )
        
        # Add RAG if enabled
        if args.use_rag:
            if not RAG_AVAILABLE:
                print("\n⚠️  RAG dependencies not found. Install with:")
                print("   pip install langchain-community")
                return 1
                
            decomposer = RAGDecomposer(
                base_decomposer=decomposer,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
        
        # Print configuration
        print("\n✅ Decomposer initialized successfully!")
        print(f"   Model: {args.model}")
        print(f"   Temperature: {args.temperature}")
        print(f"   RAG: {'Enabled' if args.use_rag else 'Disabled'}")
        
        # Process a single query if provided
        if args.query:
            print(f"\n🔍 Processing query: {args.query}")
            result = decomposer.decompose(args.query)
            print("\n" + str(result))
            return 0
            
        # Run in test mode or interactive mode
        if args.test:
            run_tests(decomposer)
        else:
            print("\nType your query and press Enter. Type 'quit' or 'exit' to end the session.\n")
            interactive_mode(decomposer)
            
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print(f"1. Make sure Ollama is running: 'ollama serve'")
        print(f"2. Make sure you have the model: 'ollama pull {args.model}'")
        print(f"3. Check that the Ollama server is accessible at {args.base_url}")
        if args.use_rag:
            print("4. For RAG support, make sure langchain is installed: 'pip install langchain-community'")
        return 1


if __name__ == "__main__":
    import sys
    
    # Check if user wants to run retrieval optimization test
    if len(sys.argv) > 1 and sys.argv[1] == "--test-retrieval":
        test_retrieval_optimization()
    else:
        main()
