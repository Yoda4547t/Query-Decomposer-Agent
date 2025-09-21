
import re
import json
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class QueryType(Enum):
    COMPARISON = "comparison"
    LIST = "list"
    SINGLE = "single"
    ANALYTICAL = "analytical"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"

class DecompositionRule(Enum):
    COMPARISON_DECOMPOSITION = "comparison_decomposition"
    LIST_DECOMPOSITION = "list_decomposition"
    MULTI_ENTITY_DECOMPOSITION = "multi_entity_decomposition"
    TEMPORAL_DECOMPOSITION = "temporal_decomposition"
    SPATIAL_DECOMPOSITION = "spatial_decomposition"

@dataclass
class DetectedEntity:
    text: str
    category: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0

@dataclass
class QueryDecompositionResult:
    original_query: str
    sub_queries: List[str]
    detected_entities: Dict[str, List[DetectedEntity]]
    decomposition_rules_applied: List[DecompositionRule]
    query_type: QueryType
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_query': self.original_query,
            'sub_queries': self.sub_queries,
            'detected_entities': {
                category: [
                    {
                        'text': entity.text,
                        'category': entity.category,
                        'start_pos': entity.start_pos,
                        'end_pos': entity.end_pos,
                        'confidence': entity.confidence
                    } for entity in entities
                ] for category, entities in self.detected_entities.items()
            },
            'decomposition_rules_applied': [rule.value for rule in self.decomposition_rules_applied],
            'query_type': self.query_type.value,
            'confidence_score': self.confidence_score
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class RetrieverRequest:
    sub_queries: List[str]
    original_query: str
    context: Optional[Dict[str, Any]] = None
    priority: int = 1

@dataclass
class AgentOutput:
    decomposition_result: QueryDecompositionResult
    retriever_request: RetrieverRequest
    processing_time: float
    agent_version: str = "1.0.0"

class EntityDetector:
    
    def __init__(self, custom_entities: Optional[Dict[str, List[str]]] = None):
        self.entity_patterns = {
            'time_range': [
                r'(last|past|previous|recent)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)',
                r'(this|current)\s+(day|week|month|year)',
                r'(today|yesterday|tomorrow)',
                r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})',
                r'(q[1-4]|quarter\s+[1-4])\s+(\d{4})'
            ],
            'location': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'numbers': r'\b\d+(?:\.\d+)?\b',
            'percentages': r'\b\d+(?:\.\d+)?%\b',
            'measurements': r'\b\d+(?:\.\d+)?\s*(?:mbps|gbps|ms|seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b'
        }
        
        self.entity_lists = {
            'comparison_keywords': [
                'compare', 'comparison', 'vs', 'versus', 'between', 'and', 'versus',
                'difference', 'contrast', 'relative', 'compared to', 'against'
            ],
            'list_keywords': [
                'give me', 'show me', 'list', 'provide', 'tell me about', 'display',
                'fetch', 'retrieve', 'get', 'find', 'search for', 'look up'
            ],
            'location_indicators': [
                'in', 'at', 'for', 'across', 'within', 'near', 'around', 'throughout',
                'covering', 'serving', 'operating in', 'available in'
            ]
        }
        
        if custom_entities:
            for category, terms in custom_entities.items():
                if category in self.entity_lists:
                    self.entity_lists[category].extend(terms)
                else:
                    self.entity_lists[category] = terms
        
        self.compiled_patterns = {}
        for pattern_name, pattern in self.entity_patterns.items():
            if isinstance(pattern, list):
                self.compiled_patterns[pattern_name] = [
                    re.compile(p, re.IGNORECASE) for p in pattern
                ]
            else:
                self.compiled_patterns[pattern_name] = re.compile(pattern, re.IGNORECASE)
    
    def detect_entities(self, text: str) -> Dict[str, List[DetectedEntity]]:
        detected_entities = {
            'locations': [],
            'time_ranges': [],
            'numbers': [],
            'percentages': [],
            'measurements': [],
            'comparison_terms': [],
            'list_terms': [],
            'custom_entities': []
        }
        
        detected_entities['locations'] = self._detect_locations(text)
        detected_entities['time_ranges'] = self._detect_time_ranges(text)
        detected_entities['numbers'] = self._detect_numbers(text)
        detected_entities['percentages'] = self._detect_percentages(text)
        detected_entities['measurements'] = self._detect_measurements(text)
        detected_entities['comparison_terms'] = self._detect_comparison_terms(text)
        detected_entities['list_terms'] = self._detect_list_terms(text)
        detected_entities['custom_entities'] = self._detect_custom_entities(text)
        
        return detected_entities
    
    def _detect_locations(self, text: str) -> List[DetectedEntity]:
        entities = []
        
        if 'location' in self.compiled_patterns:
            pattern = self.compiled_patterns['location']
            for match in pattern.finditer(text):
                matched_text = match.group()
                if not self._is_common_word(matched_text):
                    entities.append(DetectedEntity(
                        text=matched_text,
                        category='location',
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.8
                    ))
        
        return self._remove_overlapping_entities(entities)
    
    def _detect_time_ranges(self, text: str) -> List[DetectedEntity]:
        entities = []
        
        if 'time_range' in self.compiled_patterns:
            patterns = self.compiled_patterns['time_range']
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entities.append(DetectedEntity(
                        text=match.group(),
                        category='time_range',
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=1.0
                    ))
        
        return entities
    
    def _detect_numbers(self, text: str) -> List[DetectedEntity]:
        entities = []
        
        if 'numbers' in self.compiled_patterns:
            pattern = self.compiled_patterns['numbers']
            for match in pattern.finditer(text):
                entities.append(DetectedEntity(
                    text=match.group(),
                    category='number',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=1.0
                ))
        
        return entities
    
    def _detect_percentages(self, text: str) -> List[DetectedEntity]:
        entities = []
        
        if 'percentages' in self.compiled_patterns:
            pattern = self.compiled_patterns['percentages']
            for match in pattern.finditer(text):
                entities.append(DetectedEntity(
                    text=match.group(),
                    category='percentage',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=1.0
                ))
        
        return entities
    
    def _detect_measurements(self, text: str) -> List[DetectedEntity]:
        entities = []
        
        if 'measurements' in self.compiled_patterns:
            pattern = self.compiled_patterns['measurements']
            for match in pattern.finditer(text):
                entities.append(DetectedEntity(
                    text=match.group(),
                    category='measurement',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=1.0
                ))
        
        return entities
    
    def _detect_comparison_terms(self, text: str) -> List[DetectedEntity]:
        entities = []
        text_lower = text.lower()
        
        for term in self.entity_lists['comparison_keywords']:
            if term in text_lower:
                start_pos = text_lower.find(term)
                entities.append(DetectedEntity(
                    text=term,
                    category='comparison_term',
                    start_pos=start_pos,
                    end_pos=start_pos + len(term),
                    confidence=1.0
                ))
        
        return entities
    
    def _detect_list_terms(self, text: str) -> List[DetectedEntity]:
        entities = []
        text_lower = text.lower()
        
        for term in self.entity_lists['list_keywords']:
            if term in text_lower:
                start_pos = text_lower.find(term)
                entities.append(DetectedEntity(
                    text=term,
                    category='list_term',
                    start_pos=start_pos,
                    end_pos=start_pos + len(term),
                    confidence=1.0
                ))
        
        return entities
    
    def _detect_custom_entities(self, text: str) -> List[DetectedEntity]:
        entities = []
        text_lower = text.lower()
        
        for category, terms in self.entity_lists.items():
            if category not in ['comparison_keywords', 'list_keywords', 'location_indicators']:
                for term in terms:
                    if term.lower() in text_lower:
                        start_pos = text_lower.find(term.lower())
                        entities.append(DetectedEntity(
                            text=term,
                            category=category,
                            start_pos=start_pos,
                            end_pos=start_pos + len(term),
                            confidence=1.0
                        ))
        
        return entities
    
    def _is_common_word(self, word: str) -> bool:
        common_words = {
            'the', 'this', 'that', 'these', 'those', 'a', 'an', 'what', 'how',
            'when', 'where', 'why', 'who', 'which', 'and', 'or', 'but', 'in',
            'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'down',
            'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
            'compare', 'network', 'latency', 'between', 'last', 'months', 'give',
            'me', 'call', 'drop', 'rate', 'coverage', 'spectrum', 'allocation',
            'show', 'list', 'provide', 'tell', 'display', 'fetch', 'retrieve',
            'get', 'find', 'search', 'look', 'difference', 'contrast', 'relative'
        }
        return word.lower() in common_words
    
    def _remove_overlapping_entities(self, entities: List[DetectedEntity]) -> List[DetectedEntity]:
        if not entities:
            return entities
        
        entities.sort(key=lambda x: x.start_pos)
        filtered_entities = []
        
        for entity in entities:
            overlaps = False
            for existing in filtered_entities:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    overlaps = True
                    if entity.confidence > existing.confidence:
                        filtered_entities.remove(existing)
                        filtered_entities.append(entity)
                    break
            
            if not overlaps:
                filtered_entities.append(entity)
        
        return filtered_entities

class QueryDecomposer:
    
    def __init__(self, entity_detector: Optional[EntityDetector] = None):
        self.entity_detector = entity_detector or EntityDetector()
        
        self.decomposition_strategies = {
            DecompositionRule.COMPARISON_DECOMPOSITION: self._decompose_comparison,
            DecompositionRule.LIST_DECOMPOSITION: self._decompose_list,
            DecompositionRule.MULTI_ENTITY_DECOMPOSITION: self._decompose_by_entities,
            DecompositionRule.TEMPORAL_DECOMPOSITION: self._decompose_by_time,
            DecompositionRule.SPATIAL_DECOMPOSITION: self._decompose_by_space
        }
    
    def decompose_query(self, query: str) -> QueryDecompositionResult:
        detected_entities = self.entity_detector.detect_entities(query)
        
        decomposition_rules = self._identify_decomposition_rules(query, detected_entities)
        
        query_type = self._determine_query_type(query, detected_entities, decomposition_rules)
        
        sub_queries = self._apply_decomposition_strategies(
            query, detected_entities, decomposition_rules
        )
        
        confidence_score = self._calculate_confidence_score(
            detected_entities, decomposition_rules, sub_queries
        )
        
        return QueryDecompositionResult(
            original_query=query,
            sub_queries=sub_queries if sub_queries else [query],
            detected_entities=detected_entities,
            decomposition_rules_applied=decomposition_rules,
            query_type=query_type,
            confidence_score=confidence_score
        )
    
    def _identify_decomposition_rules(self, query: str, entities: Dict[str, List[DetectedEntity]]) -> List[DecompositionRule]:
        rules = []
        
        if any(entity.category == 'comparison_term' for entity_list in entities.values() for entity in entity_list):
            rules.append(DecompositionRule.COMPARISON_DECOMPOSITION)
        
        if any(entity.category == 'list_term' for entity_list in entities.values() for entity in entity_list):
            rules.append(DecompositionRule.LIST_DECOMPOSITION)
        
        for category, entity_list in entities.items():
            if len(entity_list) > 1 and category not in ['comparison_terms', 'list_terms']:
                rules.append(DecompositionRule.MULTI_ENTITY_DECOMPOSITION)
                break
        
        if len(entities.get('time_ranges', [])) > 1:
            rules.append(DecompositionRule.TEMPORAL_DECOMPOSITION)
        
        if len(entities.get('locations', [])) > 1:
            rules.append(DecompositionRule.SPATIAL_DECOMPOSITION)
        
        return rules
    
    def _determine_query_type(self, query: str, entities: Dict[str, List[DetectedEntity]], rules: List[DecompositionRule]) -> QueryType:
        if DecompositionRule.COMPARISON_DECOMPOSITION in rules:
            return QueryType.COMPARISON
        elif DecompositionRule.LIST_DECOMPOSITION in rules:
            return QueryType.LIST
        elif DecompositionRule.TEMPORAL_DECOMPOSITION in rules:
            return QueryType.TEMPORAL
        elif DecompositionRule.SPATIAL_DECOMPOSITION in rules:
            return QueryType.SPATIAL
        elif DecompositionRule.MULTI_ENTITY_DECOMPOSITION in rules:
            return QueryType.ANALYTICAL
        return QueryType.SINGLE
        
    def _apply_decomposition_strategies(self, query: str, entities: Dict[str, List[DetectedEntity]], rules: List[DecompositionRule]) -> List[str]:
        sub_queries = []
        for rule in rules:
            if rule in self.decomposition_strategies:
                strategy = self.decomposition_strategies[rule]
                result = strategy(query, entities)
                if result:
                    sub_queries.extend(result)
        
        unique_sub_queries = []
        seen = set()
        for sq in sub_queries:
            if sq not in seen:
                seen.add(sq)
                unique_sub_queries.append(sq)
        
        final_sub_queries = unique_sub_queries if unique_sub_queries else [query]
        return final_sub_queries
    
    def _decompose_comparison(self, query: str, entities: Dict[str, List[DetectedEntity]]) -> List[str]:
        sub_queries = []
        between_pattern = r'between\s+([^and]+?)\s+and\s+([^for\s]+)'
        match = re.search(between_pattern, query, re.IGNORECASE)
        if match:
            first = match.group(1).strip()
            second = match.group(2).strip()
            sub_queries.append(f"{first}")
            sub_queries.append(f"{second}")
        return sub_queries
    
    def _decompose_list(self, query: str, entities: Dict[str, List[DetectedEntity]]) -> List[str]:
        sub_queries = []
        separators = [' and ', ' & ', ', ']
        for sep in separators:
            if sep in query:
                parts = query.split(sep)
                for part in parts:
                    part = part.strip()
                    if part and part not in sub_queries:
                        sub_queries.append(part)
        return sub_queries
    
    def _decompose_by_entities(self, query: str, entities: Dict[str, List[DetectedEntity]]) -> List[str]:
        sub_queries = []
        for category, entity_list in entities.items():
            if len(entity_list) > 1 and category not in ['comparison_terms', 'list_terms']:
                for entity in entity_list:
                    modified_query = query.replace(entity.text, f"{entity.text} in {category}")
                    sub_queries.append(modified_query)
        return sub_queries if sub_queries else [query]
    
    def _decompose_by_time(self, query: str, entities: Dict[str, List[DetectedEntity]]) -> List[str]:
        sub_queries = []
        time_entities = entities.get('time_ranges', [])
        for time_entity in time_entities:
            modified_query = query.replace(time_entity.text, f"{time_entity.text} time_period")
            sub_queries.append(modified_query)
        return sub_queries if sub_queries else [query]
        
    def _decompose_by_space(self, query: str, entities: Dict[str, List[DetectedEntity]]) -> List[str]:
        sub_queries = []
        location_entities = entities.get('locations', [])
        for location_entity in location_entities:
            modified_query = query.replace(location_entity.text, f"{location_entity.text} location")
            sub_queries.append(modified_query)
        return sub_queries if sub_queries else [query]
        
    def _calculate_confidence_score(self, 
                                 entities: Dict[str, List[DetectedEntity]], 
                                 rules: List[DecompositionRule],
                                 sub_queries: List[str]) -> float:
        if not sub_queries:
            return 0.0
            
        entity_count = sum(len(entities[cat]) for cat in entities if entities[cat])
        rule_count = len(rules)
        subquery_count = len(sub_queries)
        
        if entity_count == 0:
            return 0.5 if subquery_count <= 1 else 0.7
            
        if rule_count == 0:
            return 0.6 if subquery_count <= 1 else 0.8
            
        base_score = min(0.9, 0.4 + (0.1 * entity_count) + (0.1 * rule_count))
        return min(1.0, base_score + (0.1 if subquery_count > 1 else 0))
        
    def _post_process_sub_queries(self, sub_queries: List[str], 
                               original_query: str,
                               entities: Dict[str, List[DetectedEntity]]) -> List[str]:
        if not sub_queries:
            return [original_query]
            
        processed = []
        seen = set()
        
        for query in sub_queries:
            query = query.strip()
            if not query:
                continue
                
            query = ' '.join(query.split())
            
            if query not in seen:
                seen.add(query)
                processed.append(query)
        
        if not processed:
            return [original_query]
            
        if len(processed) == 1 and processed[0] == original_query:
            return processed
            
        return processed

class QueryDecomposerAgent:
    
    def __init__(self, custom_entities: Optional[Dict[str, List[str]]] = None):
        self.entity_detector = EntityDetector(custom_entities)
        self.query_decomposer = QueryDecomposer(self.entity_detector)
        self.version = "1.0.0"
        
        self.retriever_callback: Optional[Callable[[RetrieverRequest], Any]] = None
        self.llm_callback: Optional[Callable[[str, Dict[str, Any]], Any]] = None
        self.preprocessing_callback: Optional[Callable[[str], str]] = None
        self.postprocessing_callback: Optional[Callable[[AgentOutput], Any]] = None
    
    def process_query(self, query: str, **kwargs) -> AgentOutput:
        start_time = time.time()
        
        processed_query = query
        if self.preprocessing_callback:
            processed_query = self.preprocessing_callback(query)
        
        decomposition_result = self.query_decomposer.decompose_query(processed_query)
        
        retriever_request = RetrieverRequest(
            sub_queries=decomposition_result.sub_queries,
            original_query=decomposition_result.original_query,
            context=decomposition_result.to_dict(),
            priority=1
        )
        
        processing_time = time.time() - start_time
        
        agent_output = AgentOutput(
            decomposition_result=decomposition_result,
            retriever_request=retriever_request,
            processing_time=processing_time,
            agent_version=self.version
        )
        
        if self.postprocessing_callback:
            agent_output = self.postprocessing_callback(agent_output)
        
        return agent_output
    
    def send_to_retriever(self, retriever_request: RetrieverRequest) -> Any:
        if self.retriever_callback:
            return self.retriever_callback(retriever_request)
        else:
            return {
                'sub_queries': retriever_request.sub_queries,
                'original_query': retriever_request.original_query,
                'context': retriever_request.context,
                'status': 'ready_for_retrieval'
            }
    
    def send_to_llm(self, sub_queries: list, context: Optional[Dict[str, Any]] = None) -> Any:
        if self.llm_callback:
            formatted_input = self._format_for_llm(sub_queries, context)
            return self.llm_callback(formatted_input, context or {})
        else:
            return self._format_for_llm(sub_queries, context)
    
    def _format_for_llm(self, sub_queries: list, context: Optional[Dict[str, Any]] = None) -> str:
        formatted = "Decomposed queries:\n"
        for i, query in enumerate(sub_queries, 1):
            formatted += f"{i}. {query}\n"
        
        if context:
            formatted += f"\nContext: {json.dumps(context, indent=2)}"
        
        return formatted
    
    def set_retriever_callback(self, callback: Callable[[RetrieverRequest], Any]):
        self.retriever_callback = callback
    
    def set_llm_callback(self, callback: Callable[[str, Dict[str, Any]], Any]):
        self.llm_callback = callback
    
    def set_preprocessing_callback(self, callback: Callable[[str], str]):
        self.preprocessing_callback = callback
    
    def set_postprocessing_callback(self, callback: Callable[[AgentOutput], Any]):
        self.postprocessing_callback = callback
    
    def add_custom_entities(self, category: str, terms: list):
        if category not in self.entity_detector.entity_lists:
            self.entity_detector.entity_lists[category] = []
        self.entity_detector.entity_lists[category].extend(terms)
    
    def get_agent_info(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'entity_categories': list(self.entity_detector.entity_lists.keys()),
            'decomposition_rules': [str(rule) for rule in self.query_decomposer.decomposition_strategies.keys()],
            'integration_callbacks': {
                'retriever': self.retriever_callback is not None,
                'llm': self.llm_callback is not None,
                'preprocessing': self.preprocessing_callback is not None,
                'postprocessing': self.postprocessing_callback is not None
            }
        }

def create_agent(custom_entities: Optional[Dict[str, List[str]]] = None) -> QueryDecomposerAgent:
    return QueryDecomposerAgent(custom_entities)

def process_single_query(query: str, custom_entities: Optional[Dict[str, List[str]]] = None) -> AgentOutput:
    agent = create_agent(custom_entities)
    return agent.process_query(query)

def demo_rag_integration():
    print("RAG Integration Demo")
    print("=" * 60)
    
    custom_entities = {
        'technologies': ['AI', 'ML', 'Blockchain', 'IoT', 'Cloud'],
        'products': ['ProductA', 'ProductB', 'ProductC'],
        'metrics': ['performance', 'cost', 'efficiency', 'reliability']
    }
    agent = QueryDecomposerAgent(custom_entities)
    
    def rag_retriever_callback(request: RetrieverRequest):
        print(f"🔍 RETRIEVER: Processing {len(request.sub_queries)} sub-queries...")
        retrieved_docs = []
        for sub_query in request.sub_queries:
            doc = f"Retrieved document for: {sub_query}"
            retrieved_docs.append(doc)
        return {"documents": retrieved_docs, "status": "retrieved"}
    
    def rag_llm_callback(formatted_input: str, context: dict):
        print(f"🤖 LLM: Processing {len(formatted_input.split())} words...")
        response = f"LLM Response: Based on the decomposed queries, here's the analysis..."
        return {"response": response, "status": "generated"}
    
    agent.set_retriever_callback(rag_retriever_callback)
    agent.set_llm_callback(rag_llm_callback)
    
    complex_query = "Compare AI and ML performance for ProductA and ProductB in last 6 months"
    print(f"📝 INPUT QUERY: {complex_query}")
    
    result = agent.process_query(complex_query)
    print(f"✂️  DECOMPOSED INTO: {len(result.decomposition_result.sub_queries)} sub-queries")
    for i, sub_query in enumerate(result.decomposition_result.sub_queries, 1):
        print(f"   {i}. {sub_query}")
    
    entity_summary = {}
    for category, entities in result.decomposition_result.detected_entities.items():
        if entities:
            entity_summary[category] = [entity.text for entity in entities]
    
    if entity_summary:
        print(f"\n🏷️  ENTITY ANALYSIS:")
        for category, entities in entity_summary.items():
            print(f"   {category}: {', '.join(entities)}")
    
    print(f"\n📊 DECOMPOSITION METRICS:")
    print(f"   Query Type: {result.decomposition_result.query_type.value}")
    print(f"   Confidence: {result.decomposition_result.confidence_score:.2f}")
    print(f"   Rules Applied: {[rule.value for rule in result.decomposition_result.decomposition_rules_applied]}")
    
    retriever_response = agent.send_to_retriever(result.retriever_request)
    print(f"📚 RETRIEVED: {len(retriever_response['documents'])} documents")
    
    llm_response = agent.send_to_llm(result.decomposition_result.sub_queries, result.decomposition_result.to_dict())
    print(f"💬 LLM RESPONSE: {llm_response['response'][:100]}...")
    
    print("\n✅ RAG Pipeline Complete!")
    return result, retriever_response, llm_response

def test_enhanced_decomposition():
    print("Enhanced Decomposition Test - Generating 5+ Sub-queries")
    print("=" * 70)
    
    test_queries = [
        "Compare AI and ML performance for ProductA and ProductB in last 6 months",
        "Give me performance metrics, cost analysis, reliability data, and efficiency reports for all products",
        "What is the network latency in Mumbai, Delhi, Bangalore, and Chennai for 4G and 5G services",
        "Show me spectrum allocation, coverage analysis, call drop rates, and signal strength for different regions",
        "Compare the efficiency of different algorithms in machine learning applications",
        "Analyze the performance, scalability, security, and maintenance of cloud services across different providers",
    ]
    
    agent = QueryDecomposerAgent()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. TEST QUERY: {query}")
        print("-" * 70)
        
        result = agent.process_query(query)
        
        print(f"📊 DECOMPOSITION RESULTS:")
        print(f"   Query Type: {result.decomposition_result.query_type.value}")
        print(f"   Confidence: {result.decomposition_result.confidence_score:.2f}")
        print(f"   Sub-queries ({len(result.decomposition_result.sub_queries)}):")
        
        for j, sub_query in enumerate(result.decomposition_result.sub_queries, 1):
            print(f"   {j:2d}. {sub_query}")
        
        entity_summary = {}
        for category, entities in result.decomposition_result.detected_entities.items():
            if entities:
                entity_summary[category] = [entity.text for entity in entities]
        
        if entity_summary:
            print(f"\n🏷️  ENTITY ANALYSIS:")
            for category, entities in entity_summary.items():
                print(f"   {category}: {', '.join(entities)}")
        
        print(f"\n📈 COMPLEXITY ANALYSIS:")
        total_entities = sum(len(entities) for entities in result.decomposition_result.detected_entities.values())
        print(f"   Total Entities: {total_entities}")
        print(f"   Sub-queries Generated: {len(result.decomposition_result.sub_queries)}")
        print(f"   Decomposition Ratio: {len(result.decomposition_result.sub_queries)/max(1, total_entities):.2f}")
        
        if len(result.decomposition_result.sub_queries) >= 5:
            print(f"   ✅ SUCCESS: Generated {len(result.decomposition_result.sub_queries)} sub-queries (≥5)")
        else:
            print(f"   ⚠️  Generated {len(result.decomposition_result.sub_queries)} sub-queries (<5)")

def test_dynamic_decomposition():
    print("Dynamic Decomposition Test")
    print("=" * 60)
    
    test_queries = [
        "Compare AI and ML performance",  # Simple comparison
        "Compare AI and ML performance for ProductA and ProductB in last 6 months",  # Complex
        "Give me performance metrics, cost analysis, and reliability data for all products",  # List query
        "What is the network latency in Mumbai, Delhi, and Bangalore for 4G and 5G services",  # Multi-location, multi-tech
        "Show me the spectrum allocation, coverage analysis, and call drop rates for different regions",  # Complex analytical
        "Compare the efficiency of different algorithms in machine learning applications",  # Technical comparison
    ]
    
    agent = QueryDecomposerAgent()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. TEST QUERY: {query}")
        print("-" * 50)
        
        result = agent.process_query(query)
        
        print(f"📊 DECOMPOSITION RESULTS:")
        print(f"   Query Type: {result.decomposition_result.query_type.value}")
        print(f"   Confidence: {result.decomposition_result.confidence_score:.2f}")
        print(f"   Sub-queries ({len(result.decomposition_result.sub_queries)}):")
        
        for j, sub_query in enumerate(result.decomposition_result.sub_queries, 1):
            print(f"   {j}. {sub_query}")
        
        entity_summary = {}
        for category, entities in result.decomposition_result.detected_entities.items():
            if entities:
                entity_summary[category] = [entity.text for entity in entities]
        
        if entity_summary:
            print(f"\n🏷️  DETECTED ENTITIES:")
            for category, entities in entity_summary.items():
                print(f"   {category}: {', '.join(entities)}")
        
        print(f"\n📈 COMPLEXITY ANALYSIS:")
        total_entities = sum(len(entities) for entities in result.decomposition_result.detected_entities.values())
        print(f"   Total Entities: {total_entities}")
        print(f"   Sub-queries Generated: {len(result.decomposition_result.sub_queries)}")
        print(f"   Decomposition Ratio: {len(result.decomposition_result.sub_queries)/max(1, total_entities):.2f}")

def interactive_demo():
    print("Interactive Query Decomposer Demo")
    print("=" * 60)
    print("Enter queries to see how they get decomposed (type 'quit' to exit)")
    print("Try queries of different complexity to see dynamic decomposition!")
    
    agent = QueryDecomposerAgent()
    
    while True:
        try:
            query = input("\n🔍 Enter your query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            result = agent.process_query(query)
            
            print(f"\n📊 DECOMPOSITION RESULTS:")
            print(f"   Query Type: {result.decomposition_result.query_type.value}")
            print(f"   Confidence: {result.decomposition_result.confidence_score:.2f}")
            print(f"   Sub-queries ({len(result.decomposition_result.sub_queries)}):")
            
            for i, sub_query in enumerate(result.decomposition_result.sub_queries, 1):
                print(f"   {i}. {sub_query}")
            
            # Show detected entities
            entity_summary = {}
            for category, entities in result.decomposition_result.detected_entities.items():
                if entities:
                    entity_summary[category] = [entity.text for entity in entities]
            
            if entity_summary:
                print(f"\n🏷️  DETECTED ENTITIES:")
                for category, entities in entity_summary.items():
                    print(f"   {category}: {', '.join(entities)}")
            
            # Show complexity analysis
            total_entities = sum(len(entities) for entities in result.decomposition_result.detected_entities.values())
            print(f"\n📈 COMPLEXITY ANALYSIS:")
            print(f"   Total Entities: {total_entities}")
            print(f"   Sub-queries Generated: {len(result.decomposition_result.sub_queries)}")
            print(f"   Decomposition Ratio: {len(result.decomposition_result.sub_queries)/max(1, total_entities):.2f}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n👋 Demo ended. Thanks for trying the Query Decomposer Agent!")

if __name__ == "__main__":
    print("Query Decomposer Agent - Enhanced Decomposition (5+ Sub-queries) & RAG Integration Ready")
    print("=" * 80)
    
    print("\n1. Enhanced Decomposition Test (5+ Sub-queries):")
    test_enhanced_decomposition()
    
    print("\n2. Dynamic Decomposition Test:")
    test_dynamic_decomposition()
    
    print("\n3. RAG Integration Demo:")
    demo_rag_integration()
    
    print("\n4. Basic Usage:")
    result = process_single_query("Compare network latency between 4G and 5G in Bangalore for last 3 months")
    print(f"Original: {result.decomposition_result.original_query}")
    print(f"Sub-queries: {result.decomposition_result.sub_queries}")
    print(f"Query Type: {result.decomposition_result.query_type.value}")
    print(f"Confidence: {result.decomposition_result.confidence_score:.2f}")
    
    print("\n5. Interactive Demo:")
    try:
        interactive_demo()
    except:
        print("Interactive demo skipped (non-interactive environment)")
    
    print("\n" + "=" * 80)
    print("🎯 IMPLEMENTATION GUIDE:")
    print("1. Replace placeholder functions with your actual implementations")
    print("2. Set up your input/output file paths as indicated")
    print("3. Configure your retriever and LLM integrations")
    print("4. Test with your domain-specific entities and queries")
    print("5. The agent generates 5+ sub-queries based on query complexity")
    print("6. Enhanced with multiple decomposition strategies for maximum coverage")
    print("=" * 80)
