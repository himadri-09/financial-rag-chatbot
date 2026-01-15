"""
Streamlit Application for Financial RAG Chatbot
Simplified chat interface - assumes Pinecone is already initialized.
"""

import streamlit as st
from dotenv import load_dotenv
from data_processor import DataProcessor
from vector_db import VectorDatabase
from retrieval import HybridRetrievalPipeline
from llm_handler import LLMHandler, HybridRAGPipeline
from config import config

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Financial Data Chatbot",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None


@st.cache_resource(show_spinner="🔄 Initializing chatbot...")
def initialize_rag_system():
    """Initialize RAG system once and cache it."""
    try:
        # Load data (needed for pandas aggregations)
        processor = DataProcessor(config.HOLDINGS_CSV, config.TRADES_CSV)
        processor.load_data()
        processor.clean_data()

        # Initialize vector database
        vdb = VectorDatabase()
        if not vdb.initialize_pinecone():
            st.error("❌ Failed to connect to Pinecone. Make sure it's initialized with setup_pinecone.py")
            return None

        # Initialize HYBRID retrieval pipeline
        hybrid_retrieval = HybridRetrievalPipeline(
            vdb,
            processor.holdings_df,
            processor.trades_df
        )

        # Initialize LLM
        llm = LLMHandler()

        # Create HYBRID RAG pipeline
        rag_pipeline = HybridRAGPipeline(hybrid_retrieval, llm)

        return rag_pipeline

    except Exception as e:
        st.error(f"❌ Initialization error: {str(e)}")
        st.info("💡 Make sure you've run: python setup_pinecone.py")
        return None


# Header
st.markdown('<div class="main-header">💰 Financial Data Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about fund holdings, trades, and performance</div>', unsafe_allow_html=True)

# Initialize system
if st.session_state.rag_pipeline is None:
    st.session_state.rag_pipeline = initialize_rag_system()

# Check if system is ready
if st.session_state.rag_pipeline is None:
    st.stop()

# Example questions
with st.expander("📝 Example Questions", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Aggregation Queries:**")
        if st.button("Which fund performed best?", key="ex1"):
            st.session_state.user_input = "Which fund performed best based on yearly P&L?"
        if st.button("Compare all funds", key="ex2"):
            st.session_state.user_input = "Compare P&L performance of all funds"
        if st.button("Top 3 funds by P&L", key="ex3"):
            st.session_state.user_input = "What are the top 3 funds by yearly P&L?"

    with col2:
        st.markdown("**Specific Queries:**")
        if st.button("Garfield holdings count", key="ex4"):
            st.session_state.user_input = "How many holdings does Garfield have?"
        if st.button("MNC Fund securities", key="ex5"):
            st.session_state.user_input = "What securities does MNC Investment Fund hold?"
        if st.button("HoldCo 1 trades", key="ex6"):
            st.session_state.user_input = "Total number of trades for HoldCo 1"

st.divider()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about funds, holdings, or trades...", key="chat_input"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag_pipeline.query(prompt)
                answer = result['answer']
                query_type = result.get('query_type', 'unknown')

                # Show routing info (subtle)
                route_emoji = "📊" if query_type == 'aggregation' else "🔍"
                st.caption(f"{route_emoji} {query_type.title()} Query")

                # Show answer
                st.markdown(answer)

            except Exception as e:
                answer = f"Sorry, an error occurred: {str(e)}"
                st.error(answer)

    # Add bot response to chat
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Clear chat button in sidebar
with st.sidebar:
    st.markdown("### 🗑️ Chat Controls")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption("This chatbot uses a **Hybrid RAG + Pandas** approach:")
    st.caption("- **Aggregation queries** → Pandas (all funds)")
    st.caption("- **Specific queries** → RAG (semantic search)")
    st.caption("- **Data:** Holdings (1,022 rows), Trades (649 rows)")
