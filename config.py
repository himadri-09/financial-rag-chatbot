"""
Configuration settings for Financial RAG Chatbot
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the RAG chatbot system."""

    # File paths
    HOLDINGS_CSV = "holdings.csv"
    TRADES_CSV = "trades.csv"

    # API Keys (loaded from environment)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")

    # Pinecone settings
    PINECONE_INDEX_NAME = "financial-chatbot"
    PINECONE_DIMENSION = 384  # for all-MiniLM-L6-v2
    PINECONE_METRIC = "cosine"
    PINECONE_NAMESPACE_HOLDINGS = "holdings"
    PINECONE_NAMESPACE_TRADES = "trades"

    # Embedding settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE_TOKENS = 500
    CHUNK_OVERLAP_TOKENS = 50

    # Retrieval settings
    TOP_K_CHUNKS = 10
    MIN_RELEVANCE_SCORE = 0.3
    MIN_CHUNKS_REQUIRED = 3

    # LLM settings
    LLM_MODEL = "gemini-2.5-flash"
    LLM_TEMPERATURE = 0.1  
    LLM_MAX_TOKENS = 1024

    # System prompt template
    SYSTEM_PROMPT = """You are a financial data analyst assistant. Answer questions using ONLY the provided context chunks from holdings and trades data.

CRITICAL RULES:
1. Use ONLY the data in the CONTEXT below - do not use external knowledge
2. For fund performance: GROUP by PortfolioName, SUM(PL_YTD), rank DESC
3. For counts: COUNT non-zero Qty rows per fund
4. If insufficient data: respond exactly "Sorry, I cannot find the answer in the provided data"
5. Cite specific numbers and fund names from context
6. Be precise with numbers - use exact values from the data
7. Format currency values with $ and proper formatting

CONTEXT (Retrieved Chunks):
{context_chunks}

QUESTION: {question}

ANSWER (with specific numbers from context):"""

    # Streamlit UI settings
    APP_TITLE = "💰 Financial Data Chatbot"
    APP_CAPTION = "Trained on holdings.csv (1,022 rows) & trades.csv (649 rows)"

    EXAMPLE_QUESTIONS = [
        "How many holdings does MNC Investment Fund have?",
        "Which fund performed best based on yearly P&L?",
        "Total number of trades for HoldCo 1",
        "What is the total P&L for Garfield fund?",
        "Which funds hold MSFT equity?",
        "Compare P&L of Garfield vs Heather",
        "What types of securities does Platpot hold?",
        "How many Buy vs Sell trades are there?"
    ]

    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def validate(cls) -> bool:
        """Validate that required settings are present."""
        required = {
            "GEMINI_API_KEY": cls.GEMINI_API_KEY,
            "PINECONE_API_KEY": cls.PINECONE_API_KEY,
        }

        missing = [key for key, value in required.items() if not value]

        if missing:
            print(f"⚠️  Missing required configuration: {', '.join(missing)}")
            return False

        return True

    @classmethod
    def get_prompt_template(cls, context_chunks: str, question: str) -> str:
        """Get filled prompt template."""
        return cls.SYSTEM_PROMPT.format(
            context_chunks=context_chunks,
            question=question
        )


# Export config instance
config = Config()
