"""
LLM Handler for Financial RAG Chatbot
Integrates with Google Gemini for answer generation.
"""

import time
import re
from typing import Optional, Dict
from google import genai
from google.genai import types
from config import config
from retrieval import RetrievalPipeline


class LLMHandler:
    """Handles LLM interactions for answer generation."""

    def __init__(self, api_key: str = None, model_name: str = None):
        """Initialize LLM handler with Gemini."""
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model_name = model_name or config.LLM_MODEL

        # Configure Gemini client
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            print(f"✅ Gemini API configured")
        else:
            self.client = None

        # Initialize model
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize the Gemini model."""
        try:
            if self.client:
                self.generation_config = types.GenerateContentConfig(
                    temperature=config.LLM_TEMPERATURE,
                    max_output_tokens=config.LLM_MAX_TOKENS,
                )
                self.model = self.model_name
                print(f"✅ Initialized {self.model_name}")
            else:
                self.model = None

        except Exception as e:
            print(f"❌ Error initializing model: {str(e)}")
            self.model = None

    def generate_answer(
        self,
        query: str,
        context_chunks: str,
        include_sources: bool = True
    ) -> str:
        """
        Generate an answer using the LLM with retrieved context.

        Args:
            query: User question
            context_chunks: Formatted context from retrieval
            include_sources: Whether to include source attribution

        Returns:
            Generated answer text
        """
        if not self.model or not self.client:
            return "❌ LLM model not initialized. Please check your API key."

        try:
            # Build prompt using template
            prompt = config.get_prompt_template(context_chunks, query)

            # Generate response
            print(f"🤖 Generating answer with {self.model_name}...")
            start_time = time.time()

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self.generation_config
            )

            elapsed = time.time() - start_time
            print(f"  ✅ Generated in {elapsed:.2f}s")

            # Extract text from response
            if response.text:
                answer = response.text
            else:
                answer = "Sorry, I could not generate an answer."

            return answer

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error generating answer: {error_msg}")

            # Handle common errors
            if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                return "⚠️ API rate limit exceeded. Please wait a moment and try again."
            elif "api key" in error_msg.lower():
                return "❌ Invalid API key. Please check your Gemini API key configuration."
            else:
                return f"❌ Error generating answer: {error_msg}"

    def validate_answer(self, answer: str, context_chunks: str) -> bool:
        """
        Validate that the answer appears to use only the provided context.

        Returns True if answer seems valid, False if suspicious.
        """
        answer_lower = answer.lower()

        # Check for "sorry" message (valid failure case)
        if "sorry" in answer_lower and "cannot find" in answer_lower:
            return True

        # Check if answer has actual content
        if len(answer.strip()) < 10:
            return False

        # Could add more sophisticated validation here
        # e.g., checking if numbers in answer appear in context

        return True


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation."""

    def __init__(self, retrieval_pipeline: RetrievalPipeline, llm_handler: LLMHandler):
        """Initialize with retrieval and LLM components."""
        self.retrieval = retrieval_pipeline
        self.llm = llm_handler

    def query(
        self,
        user_question: str,
        top_k: int = None,
        auto_classify: bool = True
    ) -> Dict:
        """
        Complete RAG query: retrieve context and generate answer.

        Returns:
            Dict with 'answer', 'context', 'error', and 'stats' keys
        """
        result = {
            'answer': '',
            'context': '',
            'error': None,
            'stats': {}
        }

        # Step 1: Retrieve and validate context
        print(f"\n{'='*60}")
        print(f"🔍 Processing query: {user_question}")
        print(f"{'='*60}")

        context, error = self.retrieval.retrieve_and_validate(
            query=user_question,
            top_k=top_k,
            auto_classify=auto_classify
        )

        if error:
            result['error'] = error
            result['answer'] = error
            return result

        result['context'] = context

        # Step 2: Generate answer with LLM
        answer = self.llm.generate_answer(
            query=user_question,
            context_chunks=context
        )

        result['answer'] = answer

        # Step 3: Validate answer
        is_valid = self.llm.validate_answer(answer, context)

        if not is_valid:
            print("⚠️ Answer validation failed")
            result['answer'] = "Sorry, I cannot find the answer in the provided data"

        print(f"\n✅ Query complete")
        print(f"{'='*60}\n")

        return result


def sanitize_text_for_json(text: str) -> str:
    """
    Clean text to ensure JSON serialization succeeds.
    Removes control characters while preserving newlines and formatting.
    """
    if not text:
        return text

    # Remove control characters except newline and tab
    # This regex removes characters in the range 0x00-0x1F except \n (0x0A) and \t (0x09)
    text = re.sub(r'[\x00-\x09\x0B-\x1F\x7F]', '', text)

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove any null bytes
    text = text.replace('\x00', '')

    return text.strip()


class HybridRAGPipeline:
    """
    Hybrid RAG pipeline that intelligently routes queries.

    - Aggregation queries → pandas + aggregation prompt
    - Specific queries → RAG + semantic search prompt
    """

    # Prompt template for aggregation queries (pandas-computed stats)
    AGGREGATION_PROMPT = """You are a financial data analyst. Answer the question using ONLY the aggregated statistics below computed from ALL funds in the dataset.

CRITICAL RULES:
1. Use ONLY the statistics in the CONTEXT below
2. The data already includes ALL funds - no missing funds
3. Cite specific numbers and rankings from context
4. Format currency values clearly with $ and commas
5. When comparing funds, show multiple funds, not just the top one

FORMATTING INSTRUCTIONS:
- Use **bold** for fund names, key numbers, and important terms
- Use bullet points (*) for listing multiple items
- Use numbered lists (1., 2., 3.) for rankings
- Add blank lines between sections for readability
- Structure the answer with clear sections

CONTEXT (Complete Fund Statistics):
{context}

QUESTION: {question}

ANSWER (well-formatted with bold text, lists, and clear structure):"""

    # Prompt template for specific queries (RAG chunks)
    RAG_PROMPT = """You are a financial data analyst. Answer the question using ONLY the provided context chunks from holdings and trades data.

CRITICAL RULES:
1. Use ONLY the data in the CONTEXT below
2. For counts: COUNT non-zero Qty rows per fund
3. If insufficient data: respond exactly "Sorry, I cannot find the answer in the provided data"
4. Cite specific numbers and fund names from context
5. Be precise with values from the data

FORMATTING INSTRUCTIONS:
- Use **bold** for fund names, security names, and key numbers
- Use bullet points (*) for listing holdings, securities, or multiple items
- Use numbered lists (1., 2., 3.) for rankings or ordered items
- Add blank lines between different funds or sections
- Keep the answer clear and well-structured

CONTEXT (Retrieved Chunks):
{context}

QUESTION: {question}

ANSWER (well-formatted with bold text, lists, and clear structure):"""

    def __init__(self, hybrid_retrieval, llm_handler: LLMHandler):
        """Initialize with hybrid retrieval and LLM handler."""
        self.retrieval = hybrid_retrieval
        self.llm = llm_handler

    def query(self, user_question: str) -> Dict:
        """
        Complete hybrid query: route, retrieve context, and generate answer.

        Returns:
            Dict with 'answer', 'context', 'query_type', 'error' keys
        """
        result = {
            'answer': '',
            'context': '',
            'query_type': '',
            'error': None
        }

        print(f"\n{'='*60}")
        print(f"🔍 Processing query: {user_question}")
        print(f"{'='*60}")

        # Step 1: Get context (router decides: aggregation vs RAG)
        context, error = self.retrieval.retrieve_context(user_question)

        if error:
            result['error'] = error
            result['answer'] = error
            return result

        result['context'] = context

        # Step 2: Determine query type and select prompt template
        query_type = self.retrieval.router.classify_query_type(user_question)
        result['query_type'] = query_type

        if query_type == 'aggregation':
            prompt_template = self.AGGREGATION_PROMPT
            print(f"📝 Using AGGREGATION prompt")
        else:
            prompt_template = self.RAG_PROMPT
            print(f"📝 Using RAG prompt")

        # Step 3: Fill prompt template
        filled_prompt = prompt_template.format(
            context=context,
            question=user_question
        )

        # Step 4: Generate answer with LLM
        try:
            response = self.llm.client.models.generate_content(
                model=self.llm.model,
                contents=filled_prompt,
                config=self.llm.generation_config
            )
            raw_answer = response.text if response.text else "Sorry, I could not generate an answer."
            answer = sanitize_text_for_json(raw_answer)
        except Exception as e:
            print(f"❌ Generation error: {str(e)}")
            answer = f"Error generating answer: {str(e)}"

        result['answer'] = answer

        print(f"\n✅ Hybrid query complete")
        print(f"{'='*60}\n")

        return result

    def batch_query(self, questions: list) -> list:
        """Process multiple questions in batch."""
        results = []

        for i, question in enumerate(questions, 1):
            print(f"\n📝 Processing question {i}/{len(questions)}")
            result = self.query(question)
            results.append(result)

            # Add small delay to respect rate limits
            if i < len(questions):
                time.sleep(0.5)

        return results


if __name__ == "__main__":
    # Test the LLM handler
    print("🧪 Testing LLM Handler...")

    from vector_db import VectorDatabase

    # Initialize components
    vdb = VectorDatabase()

    if vdb.initialize_pinecone():
        retrieval = RetrievalPipeline(vdb)
        llm = LLMHandler()

        # Create RAG pipeline
        rag = RAGPipeline(retrieval, llm)

        # Test query
        test_question = "How many holdings does MNC Investment Fund have?"

        print(f"\n{'='*60}")
        print(f"Testing RAG Pipeline")
        print(f"{'='*60}")

        result = rag.query(test_question)

        print(f"\n📊 Result:")
        print(f"  Answer: {result['answer'][:200]}...")
        print(f"  Error: {result['error']}")
        print(f"  Context length: {len(result['context'])} characters")

        print("\n✅ LLM handler test complete")
    else:
        print("\n❌ Could not initialize vector database")
