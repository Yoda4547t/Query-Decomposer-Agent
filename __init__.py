"""
Query Decomposer Agent Package

A general-purpose agent for decomposing complex queries into smaller sub-queries
for RAG (Retrieval-Augmented Generation) AI systems.
"""

from .query_decomposer_agent import (
    QueryDecomposerAgent,
    QueryDecomposer,
    DetectedEntity,
    DecompositionResult,
    AgentOutput,
    RetrieverRequest,
    QueryType,
    DecompositionRule,
    demo_rag_integration,
    test_enhanced_decomposition,
    test_dynamic_decomposition,
    interactive_demo,
    process_single_query,
)

__version__ = "1.0.0"
__author__ = "Yoda4547t"
__email__ = ""
__description__ = "A general-purpose agent for decomposing complex queries into smaller sub-queries for RAG systems"

__all__ = [
    "QueryDecomposerAgent",
    "QueryDecomposer", 
    "DetectedEntity",
    "DecompositionResult",
    "AgentOutput",
    "RetrieverRequest",
    "QueryType",
    "DecompositionRule",
    "demo_rag_integration",
    "test_enhanced_decomposition",
    "test_dynamic_decomposition",
    "interactive_demo",
    "process_single_query",
]
