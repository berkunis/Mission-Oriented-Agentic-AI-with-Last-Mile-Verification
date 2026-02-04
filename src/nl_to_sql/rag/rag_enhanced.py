"""
RAG-Enhanced LLM
================

LLM wrapper that uses RAG for schema-aware SQL generation.
"""

from nl_to_sql.llm.base import LLMInterface
from nl_to_sql.models import LLMResponse
from nl_to_sql.rag.schema_retriever import SchemaRetriever


class RAGEnhancedLLM(LLMInterface):
    """
    LLM wrapper that enhances prompts with retrieved schema context.

    This wrapper:
    1. Retrieves relevant schema information for the query
    2. Augments the prompt with the retrieved context
    3. Forwards to the underlying LLM
    """

    def __init__(
        self,
        base_llm: LLMInterface,
        schema: dict,
        retriever: SchemaRetriever | None = None,
        top_k: int = 5,
        include_full_schema: bool = False,
    ):
        """
        Initialize RAG-enhanced LLM.

        Args:
            base_llm: The underlying LLM to use
            schema: Database schema to index
            retriever: Optional custom retriever
            top_k: Number of chunks to retrieve
            include_full_schema: Whether to also include the full schema
        """
        self.base_llm = base_llm
        self.schema = schema
        self.top_k = top_k
        self.include_full_schema = include_full_schema

        # Initialize retriever
        self.retriever = retriever or SchemaRetriever(use_embeddings=True)
        self.retriever.index_schema(schema)

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """
        Generate SQL with RAG-enhanced context.

        Args:
            prompt: User prompt (expected to contain the NL query)
            system_prompt: Optional system prompt

        Returns:
            LLMResponse from the underlying LLM
        """
        # Extract the query from the prompt
        # Assumes format: "... Question: {query}"
        query = self._extract_query(prompt)

        # Retrieve relevant context
        context = self.retriever.get_context_for_query(query, top_k=self.top_k)

        # Build enhanced prompt
        enhanced_prompt = self._build_enhanced_prompt(prompt, context)

        # Generate with base LLM
        return self.base_llm.generate(enhanced_prompt, system_prompt)

    def _extract_query(self, prompt: str) -> str:
        """Extract the natural language query from the prompt."""
        # Look for "Question:" marker
        if "Question:" in prompt:
            parts = prompt.split("Question:")
            if len(parts) > 1:
                return parts[-1].strip()

        # Fallback: use last line
        lines = prompt.strip().split("\n")
        return lines[-1].strip()

    def _build_enhanced_prompt(self, original_prompt: str, context: str) -> str:
        """Build the enhanced prompt with retrieved context."""
        # Insert context before the question
        if "Question:" in original_prompt:
            parts = original_prompt.split("Question:")
            base_prompt = parts[0].strip()
            question = parts[1].strip() if len(parts) > 1 else ""

            enhanced = f"""{base_prompt}

{context}

Question: {question}"""
        else:
            # Prepend context
            enhanced = f"""{context}

{original_prompt}"""

        # Optionally include full schema
        if self.include_full_schema:
            schema_str = self._format_full_schema()
            enhanced = f"{schema_str}\n\n{enhanced}"

        return enhanced

    def _format_full_schema(self) -> str:
        """Format the full schema as a string."""
        lines = ["Full Database Schema:"]
        for table_name, table_info in self.schema.items():
            columns = table_info.get("columns", [])
            types = table_info.get("types", {})
            col_defs = [f"{col} ({types.get(col, 'TEXT')})" for col in columns]
            lines.append(f"  {table_name}: {', '.join(col_defs)}")
        return "\n".join(lines)


class AdaptiveRAGLLM(RAGEnhancedLLM):
    """
    Adaptive RAG-enhanced LLM that adjusts retrieval based on query complexity.

    Features:
    - Dynamic top_k based on query complexity
    - Confidence-based context inclusion
    - Query decomposition for complex queries
    """

    def __init__(
        self,
        base_llm: LLMInterface,
        schema: dict,
        min_top_k: int = 3,
        max_top_k: int = 10,
        confidence_threshold: float = 0.5,
        **kwargs,
    ):
        """
        Initialize adaptive RAG LLM.

        Args:
            base_llm: The underlying LLM
            schema: Database schema
            min_top_k: Minimum chunks to retrieve
            max_top_k: Maximum chunks to retrieve
            confidence_threshold: Score threshold for inclusion
            **kwargs: Additional arguments for base class
        """
        super().__init__(base_llm, schema, **kwargs)
        self.min_top_k = min_top_k
        self.max_top_k = max_top_k
        self.confidence_threshold = confidence_threshold

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate with adaptive retrieval."""
        query = self._extract_query(prompt)

        # Estimate query complexity
        complexity = self._estimate_complexity(query)

        # Adjust top_k based on complexity
        top_k = self.min_top_k + int((self.max_top_k - self.min_top_k) * complexity)

        # Retrieve with adaptive top_k
        result = self.retriever.retrieve(query, top_k=top_k)

        # Filter by confidence
        confident_chunks = []
        for chunk, score in zip(result.chunks, result.scores):
            if score >= self.confidence_threshold:
                confident_chunks.append(chunk)

        # Build context from confident chunks
        if confident_chunks:
            context_lines = ["Relevant schema information:"]
            for i, chunk in enumerate(confident_chunks):
                context_lines.append(f"  {i+1}. {chunk.to_context_string()}")
            context = "\n".join(context_lines)
        else:
            context = "No highly relevant schema information found. Using general knowledge."

        # Build and send enhanced prompt
        enhanced_prompt = self._build_enhanced_prompt(prompt, context)
        return self.base_llm.generate(enhanced_prompt, system_prompt)

    def _estimate_complexity(self, query: str) -> float:
        """
        Estimate query complexity (0.0 to 1.0).

        Higher complexity = more context needed.
        """
        complexity = 0.0
        query_lower = query.lower()

        # Keywords indicating complexity
        complex_keywords = [
            "join",
            "multiple",
            "both",
            "and",
            "with",
            "along with",
            "combined",
            "together",
            "compare",
            "correlation",
            "relationship",
        ]
        for kw in complex_keywords:
            if kw in query_lower:
                complexity += 0.1

        # Aggregation keywords
        agg_keywords = ["total", "sum", "count", "average", "avg", "max", "min", "group"]
        for kw in agg_keywords:
            if kw in query_lower:
                complexity += 0.05

        # Question length
        word_count = len(query.split())
        if word_count > 10:
            complexity += 0.1
        if word_count > 20:
            complexity += 0.1

        return min(complexity, 1.0)
