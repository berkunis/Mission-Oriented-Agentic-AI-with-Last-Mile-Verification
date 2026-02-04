"""
Schema Retriever
================

Schema-aware retrieval for RAG-enhanced SQL generation.
"""

from dataclasses import dataclass, field
from typing import Any

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None
    np = None


@dataclass
class SchemaChunk:
    """A chunk of schema information for retrieval."""

    chunk_id: str
    table_name: str
    content: str
    chunk_type: str  # "table", "column", "relationship", "description"
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_context_string(self) -> str:
        """Convert to a string suitable for LLM context."""
        return f"[{self.table_name}] {self.content}"


@dataclass
class RetrievalResult:
    """Result of a retrieval query."""

    chunks: list[SchemaChunk]
    scores: list[float]
    query: str


class SchemaRetriever:
    """
    Retrieves relevant schema information based on natural language queries.

    Uses semantic similarity to find the most relevant tables, columns,
    and relationships for a given query.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_embeddings: bool = True,
    ):
        """
        Initialize the schema retriever.

        Args:
            model_name: Name of the sentence-transformers model
            use_embeddings: Whether to use embeddings (False = keyword matching)
        """
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE
        self.chunks: list[SchemaChunk] = []
        self._embeddings: list[list[float]] = []

        if self.use_embeddings:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None

    def index_schema(self, schema: dict) -> None:
        """
        Index a database schema for retrieval.

        Args:
            schema: Schema dictionary with table definitions
        """
        self.chunks = []
        self._embeddings = []

        for table_name, table_info in schema.items():
            # Create table-level chunk
            columns = table_info.get("columns", [])
            types = table_info.get("types", {})

            table_description = f"Table '{table_name}' with columns: {', '.join(columns)}"
            self.chunks.append(
                SchemaChunk(
                    chunk_id=f"{table_name}_table",
                    table_name=table_name,
                    content=table_description,
                    chunk_type="table",
                    metadata={"column_count": len(columns)},
                )
            )

            # Create column-level chunks
            for col_name in columns:
                col_type = types.get(col_name, "TEXT")
                col_description = f"Column '{col_name}' in table '{table_name}', type: {col_type}"

                # Add semantic hints based on column name
                hints = self._generate_column_hints(col_name, col_type)
                if hints:
                    col_description += f". {hints}"

                self.chunks.append(
                    SchemaChunk(
                        chunk_id=f"{table_name}_{col_name}",
                        table_name=table_name,
                        content=col_description,
                        chunk_type="column",
                        metadata={"column_name": col_name, "column_type": col_type},
                    )
                )

        # Generate embeddings if available
        if self.use_embeddings:
            texts = [chunk.content for chunk in self.chunks]
            embeddings = self.model.encode(texts)
            self._embeddings = embeddings.tolist()

            for i, chunk in enumerate(self.chunks):
                chunk.embedding = self._embeddings[i]

    def _generate_column_hints(self, col_name: str, col_type: str) -> str:
        """Generate semantic hints for common column patterns."""
        hints = []
        col_lower = col_name.lower()

        # Common patterns
        if "id" in col_lower:
            hints.append("identifier field")
        if "name" in col_lower:
            hints.append("used for display and filtering")
        if "email" in col_lower:
            hints.append("contact information")
        if "date" in col_lower or "created" in col_lower or "updated" in col_lower:
            hints.append("temporal data for date-based queries")
        if "amount" in col_lower or "price" in col_lower or "cost" in col_lower:
            hints.append("monetary value, useful for SUM/AVG calculations")
        if "status" in col_lower or "tier" in col_lower or "type" in col_lower:
            hints.append("categorical field for filtering and grouping")
        if "count" in col_lower or "quantity" in col_lower or "stock" in col_lower:
            hints.append("numeric count, useful for inventory queries")

        return " ".join(hints)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> RetrievalResult:
        """
        Retrieve relevant schema chunks for a query.

        Args:
            query: Natural language query
            top_k: Maximum number of chunks to return
            min_score: Minimum similarity score threshold

        Returns:
            RetrievalResult with ranked chunks
        """
        if not self.chunks:
            return RetrievalResult(chunks=[], scores=[], query=query)

        if self.use_embeddings:
            return self._retrieve_with_embeddings(query, top_k, min_score)
        else:
            return self._retrieve_with_keywords(query, top_k)

    def _retrieve_with_embeddings(
        self,
        query: str,
        top_k: int,
        min_score: float,
    ) -> RetrievalResult:
        """Retrieve using semantic similarity."""
        query_embedding = self.model.encode([query])[0]

        # Calculate cosine similarities
        scores = []
        for chunk_embedding in self._embeddings:
            score = self._cosine_similarity(query_embedding, chunk_embedding)
            scores.append(score)

        # Sort by score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Filter and limit
        result_chunks = []
        result_scores = []
        for idx, score in indexed_scores[:top_k]:
            if score >= min_score:
                result_chunks.append(self.chunks[idx])
                result_scores.append(score)

        return RetrievalResult(
            chunks=result_chunks,
            scores=result_scores,
            query=query,
        )

    def _retrieve_with_keywords(
        self,
        query: str,
        top_k: int,
    ) -> RetrievalResult:
        """Retrieve using keyword matching (fallback when embeddings unavailable)."""
        query_words = set(query.lower().split())

        scores = []
        for chunk in self.chunks:
            chunk_words = set(chunk.content.lower().split())
            # Jaccard similarity
            intersection = len(query_words & chunk_words)
            union = len(query_words | chunk_words)
            score = intersection / union if union > 0 else 0.0
            scores.append(score)

        # Sort by score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        result_chunks = []
        result_scores = []
        for idx, score in indexed_scores[:top_k]:
            result_chunks.append(self.chunks[idx])
            result_scores.append(score)

        return RetrievalResult(
            chunks=result_chunks,
            scores=result_scores,
            query=query,
        )

    @staticmethod
    def _cosine_similarity(vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        if not EMBEDDINGS_AVAILABLE:
            return 0.0

        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get_context_for_query(
        self,
        query: str,
        top_k: int = 5,
        include_scores: bool = False,
    ) -> str:
        """
        Get formatted context string for a query.

        Args:
            query: Natural language query
            top_k: Maximum number of chunks to include
            include_scores: Whether to include relevance scores

        Returns:
            Formatted context string for LLM prompt
        """
        result = self.retrieve(query, top_k=top_k)

        if not result.chunks:
            return "No relevant schema information found."

        lines = ["Relevant schema information:"]
        for i, (chunk, score) in enumerate(zip(result.chunks, result.scores)):
            if include_scores:
                lines.append(f"  {i+1}. [{score:.2f}] {chunk.to_context_string()}")
            else:
                lines.append(f"  {i+1}. {chunk.to_context_string()}")

        return "\n".join(lines)
