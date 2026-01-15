"""
Retrieval Pipeline for Financial RAG Chatbot
Handles query classification, semantic search, and relevance validation.
"""

from typing import List, Dict, Tuple, Optional
from vector_db import VectorDatabase
from config import config


class RetrievalPipeline:
    """Manages the retrieval pipeline for RAG queries."""

    def __init__(self, vector_db: VectorDatabase):
        """Initialize with a vector database instance."""
        self.vector_db = vector_db

    def classify_query(self, query: str) -> Tuple[Dict, Optional[str]]:
        """
        Classify query to determine filtering and namespace preference.

        Returns:
            filters: Dictionary of metadata filters
            preferred_namespace: 'holdings', 'trades', or None (search both)
        """
        query_lower = query.lower()
        filters = {}
        preferred_namespace = None

        # Check for P&L related queries
        pl_keywords = ['performance', 'p&l', 'profit', 'loss', 'pl_ytd', 'yearly', 'return']
        if any(keyword in query_lower for keyword in pl_keywords):
            filters['has_pl'] = True
            preferred_namespace = 'holdings'  # P&L data is in holdings

        # Check for trade-specific queries
        trade_keywords = ['trade', 'buy', 'sell', 'transaction', 'purchase']
        if any(keyword in query_lower for keyword in trade_keywords):
            preferred_namespace = 'trades'

        # Check for holdings-specific queries
        holdings_keywords = ['holding', 'position', 'quantity held', 'owns']
        if any(keyword in query_lower for keyword in holdings_keywords):
            preferred_namespace = 'holdings'

        return filters, preferred_namespace

    def retrieve_chunks(
        self,
        query: str,
        top_k: int = None,
        filters: Dict = None,
        namespace: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User query text
            top_k: Number of top chunks to retrieve
            filters: Metadata filters
            namespace: Specific namespace to search, or None for both

        Returns:
            List of relevant chunks with scores
        """
        top_k = top_k or config.TOP_K_CHUNKS

        if namespace:
            # Search specific namespace
            results = self.vector_db.query(
                query_text=query,
                namespace=namespace,
                top_k=top_k,
                filter_dict=filters
            )
        else:
            # Search both namespaces
            results = self.vector_db.query_both_namespaces(
                query_text=query,
                top_k=top_k,
                filter_dict=filters
            )

        return results

    def validate_retrieval(
        self,
        results: List[Dict],
        min_score: float = None,
        min_chunks: int = None
    ) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """
        Validate that retrieved chunks are relevant enough.

        Returns:
            (valid_chunks, error_message)
            If validation fails, returns (None, error_message)
        """
        min_score = min_score or config.MIN_RELEVANCE_SCORE
        min_chunks = min_chunks or config.MIN_CHUNKS_REQUIRED

        if not results:
            return None, "Sorry, I cannot find the answer in the provided data"

        # Filter by minimum score
        relevant_chunks = [r for r in results if r['score'] >= min_score]

        if len(relevant_chunks) < min_chunks:
            return None, "Sorry, I cannot find the answer in the provided data"

        return relevant_chunks, None

    def deduplicate_chunks(self, chunks: List[Dict], similarity_threshold: float = 0.95) -> List[Dict]:
        """
        Remove near-duplicate chunks based on score similarity.

        This prevents returning multiple very similar chunks that don't add value.
        """
        if not chunks:
            return chunks

        deduplicated = [chunks[0]]  # Always keep the best match

        for chunk in chunks[1:]:
            # Check if this chunk is too similar to any already selected chunk
            is_duplicate = False

            for selected in deduplicated:
                # If from the same fund and scores are very close, consider duplicate
                if (chunk['metadata'].get('fund') == selected['metadata'].get('fund') and
                    abs(chunk['score'] - selected['score']) < 0.05):
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(chunk)

        return deduplicated

    def format_chunks_for_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context string for LLM.

        Returns nicely formatted text with chunk metadata and content.
        """
        if not chunks:
            return "No relevant data found."

        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            metadata = chunk['metadata']
            fund = metadata.get('fund', 'Unknown')
            file_type = metadata.get('file', 'Unknown')
            score = chunk['score']

            # Get chunk text
            text = metadata.get('text', '')

            chunk_header = f"=== Chunk {i} (Fund: {fund}, Source: {file_type}, Relevance: {score:.2f}) ==="
            context_parts.append(f"{chunk_header}\n{text}\n")

        return "\n".join(context_parts)

    def retrieve_and_validate(
        self,
        query: str,
        top_k: int = None,
        auto_classify: bool = True
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Complete retrieval pipeline: classify, retrieve, validate, format.

        Returns:
            (formatted_context, error_message)
            If successful, returns (context_string, None)
            If failed, returns (None, error_message)
        """
        # Auto-classify query if enabled
        filters = None
        namespace = None

        if auto_classify:
            filters, namespace = self.classify_query(query)
            print(f"📋 Query classification: filters={filters}, namespace={namespace}")

        # Retrieve chunks
        print(f"🔍 Retrieving chunks for query: '{query[:50]}...'")
        chunks = self.retrieve_chunks(
            query=query,
            top_k=top_k,
            filters=filters,
            namespace=namespace
        )

        if not chunks:
            return None, "Sorry, I cannot find the answer in the provided data"

        print(f"  Found {len(chunks)} chunks")

        # Validate retrieval
        valid_chunks, error = self.validate_retrieval(chunks)

        if error:
            print(f"  ⚠️ Validation failed: {error}")
            return None, error

        print(f"  ✅ {len(valid_chunks)} chunks passed validation")

        # Deduplicate
        deduplicated_chunks = self.deduplicate_chunks(valid_chunks)
        print(f"  📦 {len(deduplicated_chunks)} chunks after deduplication")

        # Format for context
        formatted_context = self.format_chunks_for_context(deduplicated_chunks)

        return formatted_context, None

    def get_retrieval_stats(self, chunks: List[Dict]) -> Dict:
        """Get statistics about retrieved chunks."""
        if not chunks:
            return {}

        funds = set()
        sources = set()
        scores = []

        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            funds.add(metadata.get('fund', 'Unknown'))
            sources.add(metadata.get('file', 'Unknown'))
            scores.append(chunk.get('score', 0))

        return {
            'total_chunks': len(chunks),
            'unique_funds': len(funds),
            'funds': list(funds),
            'sources': list(sources),
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'min_score': min(scores) if scores else 0
        }


class HybridRetrievalPipeline:
    """
    Hybrid retrieval system that routes queries intelligently.

    - Aggregation queries → pandas (ALL funds data)
    - Specific queries → RAG (semantic search)
    """

    def __init__(self, vector_db: VectorDatabase, holdings_df, trades_df):
        """Initialize with vector DB and DataFrames for aggregation."""
        from query_router import QueryRouter

        self.vector_db = vector_db
        self.router = QueryRouter(holdings_df, trades_df)
        self.rag_retrieval = RetrievalPipeline(vector_db)

    def retrieve_context(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Route query and retrieve appropriate context.

        Returns:
            (context, error_message)
        """
        # Classify query type
        query_type = self.router.classify_query_type(query)

        if query_type == 'aggregation':
            # Use pandas for accurate all-fund aggregation
            print(f"🔀 Routing to AGGREGATION (pandas) - ensures ALL funds included")

            try:
                stats = self.router.compute_fund_aggregations()
                context = self.router.format_aggregation_context(stats, query)
                return context, None
            except Exception as e:
                print(f"❌ Aggregation error: {str(e)}")
                return None, "Sorry, I cannot find the answer in the provided data"

        else:
            # Use RAG for specific queries
            print(f"🔀 Routing to RAG (semantic search)")
            return self.rag_retrieval.retrieve_and_validate(query)


if __name__ == "__main__":
    # Test the retrieval pipeline
    print("🧪 Testing Retrieval Pipeline...")

    from vector_db import VectorDatabase
    from data_processor import DataProcessor

    # Load data for hybrid testing
    processor = DataProcessor('holdings.csv', 'trades.csv')
    processor.load_data()
    processor.clean_data()

    # Initialize vector database
    vdb = VectorDatabase()

    if vdb.initialize_pinecone():
        # Initialize HYBRID retrieval pipeline
        hybrid_retrieval = HybridRetrievalPipeline(
            vdb,
            processor.holdings_df,
            processor.trades_df
        )

        # Test queries (aggregation + specific)
        test_queries = [
            ("Which fund performed best based on yearly P&L?", "aggregation"),
            ("How many holdings does MNC Investment Fund have?", "specific"),
            ("Compare all funds", "aggregation"),
            ("Total number of trades for HoldCo 1", "specific"),
            ("What is Apple's market cap?", "specific")  # Should fail
        ]

        for query, expected_type in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"Expected routing: {expected_type}")
            print(f"{'='*60}")

            context, error = hybrid_retrieval.retrieve_context(query)

            if error:
                print(f"❌ {error}")
            else:
                print(f"✅ Retrieved context ({len(context)} characters)")
                print(f"\nContext preview:\n{context[:500]}...")

        print("\n✅ Hybrid retrieval pipeline test complete")
    else:
        print("\n❌ Could not initialize vector database")
