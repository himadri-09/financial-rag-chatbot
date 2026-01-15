"""
Test Suite for Financial RAG Chatbot
Validates data processing, retrieval, and generation.
"""

import pandas as pd
from data_processor import DataProcessor
from vector_db import VectorDatabase
from retrieval import RetrievalPipeline
from llm_handler import LLMHandler, RAGPipeline
from config import config


class TestRAGSystem:
    """Test suite for the RAG chatbot system."""

    def __init__(self):
        """Initialize test suite."""
        self.processor = None
        self.vdb = None
        self.retrieval = None
        self.llm = None
        self.rag = None
        self.test_results = []

    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """Log a test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")

        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'message': message
        })

    def test_data_loading(self):
        """Test 1: Data loading and validation."""
        print("\n" + "="*60)
        print("Test 1: Data Loading")
        print("="*60)

        try:
            self.processor = DataProcessor(config.HOLDINGS_CSV, config.TRADES_CSV)
            holdings_df, trades_df = self.processor.load_data()

            # Validate row counts
            holdings_count = len(holdings_df)
            trades_count = len(trades_df)

            self.log_result(
                "Load CSVs",
                holdings_count == 1022 and trades_count == 649,
                f"Holdings: {holdings_count}, Trades: {trades_count}"
            )

            # Clean data
            self.processor.clean_data()
            self.log_result("Clean data", True, "Data cleaning completed")

            return True

        except Exception as e:
            self.log_result("Load CSVs", False, str(e))
            return False

    def test_chunking(self):
        """Test 2: Data chunking."""
        print("\n" + "="*60)
        print("Test 2: Data Chunking")
        print("="*60)

        try:
            holdings_chunks, trades_chunks = self.processor.process_all_data()

            total_chunks = len(holdings_chunks) + len(trades_chunks)

            self.log_result(
                "Create chunks",
                total_chunks > 0,
                f"Created {len(holdings_chunks)} holdings + {len(trades_chunks)} trades = {total_chunks} total"
            )

            # Validate chunk structure
            if holdings_chunks:
                sample = holdings_chunks[0]
                has_text = 'text' in sample
                has_metadata = 'metadata' in sample
                has_fund = 'fund' in sample.get('metadata', {})

                self.log_result(
                    "Chunk structure",
                    has_text and has_metadata and has_fund,
                    "Chunks have required fields"
                )

            return holdings_chunks, trades_chunks

        except Exception as e:
            self.log_result("Create chunks", False, str(e))
            return [], []

    def test_vector_db(self, holdings_chunks, trades_chunks):
        """Test 3: Vector database operations."""
        print("\n" + "="*60)
        print("Test 3: Vector Database")
        print("="*60)

        try:
            self.vdb = VectorDatabase()

            # Initialize
            init_success = self.vdb.initialize_pinecone()
            self.log_result("Initialize Pinecone", init_success)

            if not init_success:
                return False

            # Test embedding
            test_text = "MNC Investment Fund MSFT holdings"
            embedding = self.vdb.embed_text(test_text)

            self.log_result(
                "Generate embedding",
                len(embedding) == 384,
                f"Embedding dimension: {len(embedding)}"
            )

            # Upsert chunks (optional in test - may skip if already done)
            # Uncomment to test upsert:
            # self.vdb.upsert_chunks(holdings_chunks[:5], config.PINECONE_NAMESPACE_HOLDINGS)
            # self.vdb.upsert_chunks(trades_chunks[:5], config.PINECONE_NAMESPACE_TRADES)

            # Get stats
            stats = self.vdb.get_index_stats()
            self.log_result(
                "Index stats",
                'total_vectors' in stats,
                f"Total vectors: {stats.get('total_vectors', 0)}"
            )

            return True

        except Exception as e:
            self.log_result("Vector database", False, str(e))
            return False

    def test_retrieval(self):
        """Test 4: Retrieval pipeline."""
        print("\n" + "="*60)
        print("Test 4: Retrieval Pipeline")
        print("="*60)

        try:
            self.retrieval = RetrievalPipeline(self.vdb)

            # Test query classification
            test_queries = [
                ("How many holdings does MNC Investment Fund have?", "holdings"),
                ("Total trades for HoldCo 1", "trades"),
                ("Which fund has best P&L?", "holdings")
            ]

            for query, expected_preference in test_queries:
                filters, namespace = self.retrieval.classify_query(query)
                self.log_result(
                    f"Classify: '{query[:30]}...'",
                    namespace == expected_preference or namespace is None,
                    f"Namespace: {namespace}, Filters: {filters}"
                )

            # Test retrieval
            test_query = "MNC Investment Fund holdings"
            chunks = self.retrieval.retrieve_chunks(test_query, top_k=5)

            self.log_result(
                "Retrieve chunks",
                len(chunks) > 0,
                f"Retrieved {len(chunks)} chunks"
            )

            # Test validation
            if chunks:
                valid, error = self.retrieval.validate_retrieval(chunks)
                self.log_result(
                    "Validate retrieval",
                    error is None,
                    f"Valid chunks: {len(valid) if valid else 0}"
                )

            return True

        except Exception as e:
            self.log_result("Retrieval pipeline", False, str(e))
            return False

    def test_rag_pipeline(self):
        """Test 5: Complete RAG pipeline."""
        print("\n" + "="*60)
        print("Test 5: RAG Pipeline (End-to-End)")
        print("="*60)

        try:
            self.llm = LLMHandler()
            self.rag = RAGPipeline(self.retrieval, self.llm)

            # Test questions
            test_cases = [
                {
                    'question': "How many holdings does Garfield have?",
                    'should_find_answer': True,
                    'expected_content': ['221', 'Garfield']
                },
                {
                    'question': "What is Apple's stock price today?",
                    'should_find_answer': False,
                    'expected_content': ['Sorry', 'cannot find']
                }
            ]

            for i, test_case in enumerate(test_cases, 1):
                question = test_case['question']
                should_find = test_case['should_find_answer']
                expected = test_case['expected_content']

                print(f"\n--- Test Case {i} ---")
                print(f"Q: {question}")

                result = self.rag.query(question)
                answer = result['answer']

                print(f"A: {answer[:150]}...")

                if should_find:
                    # Should get a real answer
                    passed = not any(word in answer.lower() for word in ['sorry', 'cannot find'])
                    self.log_result(
                        f"RAG Test {i} (expects answer)",
                        passed,
                        f"Got {'valid' if passed else 'invalid'} answer"
                    )
                else:
                    # Should get "Sorry" message
                    passed = any(word in answer.lower() for word in ['sorry', 'cannot find'])
                    self.log_result(
                        f"RAG Test {i} (expects rejection)",
                        passed,
                        f"Correctly rejected out-of-scope question"
                    )

            return True

        except Exception as e:
            self.log_result("RAG Pipeline", False, str(e))
            return False

    def test_ground_truth_validation(self):
        """Test 6: Validate answers against ground truth."""
        print("\n" + "="*60)
        print("Test 6: Ground Truth Validation")
        print("="*60)

        try:
            # Calculate ground truth from data
            holdings_df = self.processor.holdings_df

            # Test 1: Count holdings for Garfield
            garfield_count = len(holdings_df[holdings_df['PortfolioName'] == 'Garfield'])
            print(f"Ground truth: Garfield has {garfield_count} holdings")

            result = self.rag.query("How many holdings does Garfield have?")
            answer = result['answer']

            contains_count = str(garfield_count) in answer
            self.log_result(
                "Garfield holdings count",
                contains_count,
                f"Answer mentions {garfield_count}: {contains_count}"
            )

            # Test 2: Top fund by P&L
            top_fund_pl = holdings_df.groupby('PortfolioName')['PL_YTD'].sum().sort_values(ascending=False)
            top_fund = top_fund_pl.index[0]
            top_pl = top_fund_pl.values[0]

            print(f"Ground truth: Top fund by P&L is {top_fund} with ${top_pl:,.2f}")

            result = self.rag.query("Which fund performed best based on yearly P&L?")
            answer = result['answer']

            mentions_top_fund = top_fund in answer
            self.log_result(
                "Top fund by P&L",
                mentions_top_fund,
                f"Answer mentions {top_fund}: {mentions_top_fund}"
            )

            return True

        except Exception as e:
            self.log_result("Ground truth validation", False, str(e))
            return False

    def run_all_tests(self):
        """Run all tests in sequence."""
        print("\n" + "="*60)
        print("🧪 FINANCIAL RAG CHATBOT - TEST SUITE")
        print("="*60)

        # Test 1: Data loading
        if not self.test_data_loading():
            print("\n❌ Data loading failed - stopping tests")
            return False

        # Test 2: Chunking
        holdings_chunks, trades_chunks = self.test_chunking()
        if not holdings_chunks and not trades_chunks:
            print("\n❌ Chunking failed - stopping tests")
            return False

        # Test 3: Vector DB
        if not self.test_vector_db(holdings_chunks, trades_chunks):
            print("\n❌ Vector DB setup failed - stopping tests")
            return False

        # Test 4: Retrieval
        if not self.test_retrieval():
            print("\n❌ Retrieval failed - stopping tests")
            return False

        # Test 5: RAG Pipeline
        if not self.test_rag_pipeline():
            print("\n❌ RAG pipeline failed - continuing to validation")

        # Test 6: Ground truth validation
        self.test_ground_truth_validation()

        # Summary
        self.print_summary()

        return True

    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)

        passed = sum(1 for r in self.test_results if r['passed'])
        total = len(self.test_results)
        pass_rate = (passed / total * 100) if total > 0 else 0

        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Pass Rate: {pass_rate:.1f}%")

        if pass_rate == 100:
            print("\n🎉 ALL TESTS PASSED!")
        elif pass_rate >= 80:
            print("\n✅ Most tests passed - system functional")
        else:
            print("\n⚠️ Many tests failed - review errors above")

        print("\n" + "="*60)


if __name__ == "__main__":
    # Run the test suite
    tester = TestRAGSystem()
    tester.run_all_tests()
