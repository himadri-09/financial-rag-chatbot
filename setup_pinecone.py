"""
One-time setup script to initialize Pinecone with embeddings.
Run this once: python setup_pinecone.py
"""

import os
from dotenv import load_dotenv
from data_processor import DataProcessor
from vector_db import VectorDatabase
from config import config

# Load environment variables
load_dotenv()

def main():
    print("=" * 60)
    print("🚀 ONE-TIME SETUP: Initializing Pinecone")
    print("=" * 60)

    # Step 1: Load and process data
    print("\n📂 Step 1: Loading CSV files...")
    processor = DataProcessor(config.HOLDINGS_CSV, config.TRADES_CSV)
    processor.load_data()
    processor.clean_data()

    # Step 2: Create chunks
    print("\n✂️ Step 2: Creating data chunks...")
    holdings_chunks, trades_chunks = processor.process_all_data()
    print(f"✅ Created {len(holdings_chunks)} holdings chunks")
    print(f"✅ Created {len(trades_chunks)} trades chunks")

    # Step 3: Initialize Pinecone
    print("\n🔌 Step 3: Connecting to Pinecone...")
    vdb = VectorDatabase()

    if not vdb.initialize_pinecone():
        print("❌ Failed to initialize Pinecone. Check your API key.")
        return False

    # Step 4: Upload embeddings
    print("\n📤 Step 4: Uploading embeddings to Pinecone...")
    print("   (This will take 2-3 minutes...)")

    print(f"\n   Uploading {len(holdings_chunks)} holdings chunks...")
    vdb.upsert_chunks(holdings_chunks, config.PINECONE_NAMESPACE_HOLDINGS)

    print(f"\n   Uploading {len(trades_chunks)} trades chunks...")
    vdb.upsert_chunks(trades_chunks, config.PINECONE_NAMESPACE_TRADES)

    # Step 5: Verify
    print("\n✅ Step 5: Verifying setup...")
    stats = vdb.get_index_stats()
    print(f"   Total vectors in Pinecone: {stats.get('total_vectors', 0)}")
    print(f"   Namespaces: {list(stats.get('namespaces', {}).keys())}")

    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print("\nYou can now run the chatbot:")
    print("   streamlit run app.py")
    print("\nNote: You only need to run this setup once.")
    print("=" * 60)

    return True

if __name__ == "__main__":
    main()
