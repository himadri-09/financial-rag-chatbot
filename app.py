"""
FastAPI Application for Financial RAG Chatbot
Simple REST API with minimal HTML UI and comprehensive logging.
"""

import logging
import sys
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import uvicorn

from data_processor import DataProcessor
from vector_db import VectorDatabase
from retrieval import HybridRetrievalPipeline
from llm_handler import LLMHandler, HybridRAGPipeline, sanitize_text_for_json
from config import config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Data Chatbot API",
    description="API for querying fund holdings, trades, and performance data",
    version="1.0.0"
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Global variables for the RAG system
rag_pipeline = None
initialization_error = None


# Request/Response models
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    message: str
    system_ready: bool


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global rag_pipeline, initialization_error

    logger.info("="*60)
    logger.info("Starting Financial RAG Chatbot API")
    logger.info("="*60)

    try:
        logger.info("Loading CSV data...")
        processor = DataProcessor(config.HOLDINGS_CSV, config.TRADES_CSV)
        processor.load_data()
        processor.clean_data()
        logger.info("CSV data loaded successfully")

        logger.info("Connecting to vector database...")
        vdb = VectorDatabase()
        if not vdb.initialize_pinecone():
            raise Exception("Failed to connect to Pinecone. Make sure it's initialized.")
        logger.info("Vector database connected")

        logger.info("Setting up hybrid retrieval pipeline...")
        hybrid_retrieval = HybridRetrievalPipeline(
            vdb,
            processor.holdings_df,
            processor.trades_df
        )
        logger.info("Hybrid retrieval pipeline ready")

        logger.info("Initializing LLM...")
        llm = LLMHandler()
        logger.info("LLM initialized")

        rag_pipeline = HybridRAGPipeline(hybrid_retrieval, llm)
        logger.info("="*60)
        logger.info("RAG System Initialized Successfully!")
        logger.info("="*60)

    except Exception as e:
        initialization_error = str(e)
        logger.error(f"Initialization Error: {initialization_error}", exc_info=True)
        logger.info("Make sure you've run: python setup_pinecone.py")


@app.get("/")
async def read_root():
    """Serve the minimal HTML UI."""
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    else:
        return HTMLResponse(
            content="<h1>UI not found</h1><p>Please ensure the static/index.html file exists.</p>",
            status_code=404
        )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if initialization_error:
        logger.warning(f"Health check failed: {initialization_error}")
        return HealthResponse(
            status="error",
            message=f"System initialization failed: {initialization_error}",
            system_ready=False
        )

    if rag_pipeline is None:
        return HealthResponse(
            status="initializing",
            message="System is still initializing",
            system_ready=False
        )

    return HealthResponse(
        status="healthy",
        message="System is ready",
        system_ready=True
    )


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query endpoint for asking questions."""
    logger.info(f"Received query: {request.question}")

    if initialization_error:
        logger.error(f"Query rejected - system not initialized: {initialization_error}")
        raise HTTPException(
            status_code=500,
            detail=f"System not initialized: {initialization_error}"
        )

    if rag_pipeline is None:
        logger.warning("Query rejected - system still initializing")
        raise HTTPException(
            status_code=503,
            detail="System is still initializing. Please try again in a moment."
        )

    try:
        result = rag_pipeline.query(request.question)
        logger.info(f"Query processed successfully | Type: {result.get('query_type', 'unknown')}")

        # Triple sanitize to ensure JSON compatibility
        raw_answer = result.get('answer', '')
        logger.info(f"Raw answer length: {len(raw_answer)}")
        
        # First sanitization - use the imported function
        sanitized_answer = sanitize_text_for_json(raw_answer)
        logger.info(f"After sanitize_text_for_json: {len(sanitized_answer)}")
        
        # Second pass - remove any remaining problematic characters
        sanitized_answer = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', sanitized_answer)
        logger.info(f"After regex cleanup: {len(sanitized_answer)}")
        
        # Third pass - ensure proper string encoding
        sanitized_answer = sanitized_answer.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        logger.info(f"Final answer length: {len(sanitized_answer)}")

        return QueryResponse(
            answer=sanitized_answer,
            query_type=result.get('query_type', 'unknown'),
            error=result.get('error')
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing query: {error_msg}", exc_info=True)
        
        # Try to return a safe error message
        safe_error = sanitize_text_for_json(f"Error processing query: {error_msg}")
        
        return QueryResponse(
            answer=safe_error,
            query_type="error",
            error=safe_error
        )


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Starting Financial Data Chatbot Server")
    logger.info("="*60)
    logger.info("Server: http://localhost:8000")
    logger.info("API Docs: http://localhost:8000/docs")
    logger.info("="*60)

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_config=None  # Use our custom logging
    )