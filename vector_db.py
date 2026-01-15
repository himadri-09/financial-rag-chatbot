"""
Vector Database Module for Financial RAG Chatbot
Handles Pinecone initialization, embedding, and vector storage.
"""

import time
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from config import config


class VectorDatabase:
    """Manages vector embeddings and Pinecone operations."""

    def __init__(self, api_key: str = None, environment: str = None):
        """Initialize vector database with Pinecone."""
        self.api_key = api_key or config.PINECONE_API_KEY
        self.environment = environment or config.PINECONE_ENVIRONMENT
        self.index_name = config.PINECONE_INDEX_NAME

        # Initialize Pinecone client
        self.pc = None
        self.index = None

        # Initialize embedding model
        print(f"📥 Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print(f"✅ Embedding model loaded (dimension: {config.PINECONE_DIMENSION})")

    def initialize_pinecone(self):
        """Initialize Pinecone connection and create/connect to index."""
        print("🔌 Initializing Pinecone...")

        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)

            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                print(f"📦 Creating new index: {self.index_name}")

                # Create index with serverless spec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=config.PINECONE_DIMENSION,
                    metric=config.PINECONE_METRIC,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )

                # Wait for index to be ready
                print("⏳ Waiting for index to be ready...")
                time.sleep(5)

            else:
                print(f"✅ Index '{self.index_name}' already exists")

            # Connect to index
            self.index = self.pc.Index(self.index_name)
            print(f"✅ Connected to Pinecone index: {self.index_name}")

            # Get index stats
            stats = self.index.describe_index_stats()
            print(f"📊 Index stats: {stats.get('total_vector_count', 0)} vectors")

            return True

        except Exception as e:
            print(f"❌ Error initializing Pinecone: {str(e)}")
            return False

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        print(f"🔄 Embedding {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=False
        )
        return embeddings.tolist()

    def upsert_chunks(self, chunks: List[Dict], namespace: str, batch_size: int = 100):
        """Upsert chunks to Pinecone index."""
        if not self.index:
            print("❌ Index not initialized. Call initialize_pinecone() first.")
            return False

        print(f"📤 Upserting {len(chunks)} chunks to namespace '{namespace}'...")

        try:
            # Prepare vectors for upsert
            vectors = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{namespace}_{i}"
                text = chunk['text']
                metadata = chunk['metadata']

                # Generate embedding
                embedding = self.embed_text(text)

                # Add text to metadata (Pinecone allows storing original text)
                metadata['text'] = text[:1000]  # Limit text length in metadata

                # Prepare vector tuple (id, embedding, metadata)
                vectors.append((chunk_id, embedding, metadata))

                # Upsert in batches
                if len(vectors) >= batch_size:
                    self.index.upsert(vectors=vectors, namespace=namespace)
                    print(f"  ✓ Upserted batch of {len(vectors)} vectors")
                    vectors = []

            # Upsert remaining vectors
            if vectors:
                self.index.upsert(vectors=vectors, namespace=namespace)
                print(f"  ✓ Upserted final batch of {len(vectors)} vectors")

            print(f"✅ Successfully upserted {len(chunks)} chunks to '{namespace}'")
            return True

        except Exception as e:
            print(f"❌ Error upserting chunks: {str(e)}")
            return False

    def query(self, query_text: str, namespace: str, top_k: int = None,
              filter_dict: Dict = None) -> List[Dict]:
        """Query the vector database."""
        if not self.index:
            print("❌ Index not initialized. Call initialize_pinecone() first.")
            return []

        top_k = top_k or config.TOP_K_CHUNKS

        try:
            # Generate query embedding
            query_embedding = self.embed_text(query_text)

            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict,
                include_metadata=True
            )

            # Format results
            matches = []
            for match in results.get('matches', []):
                matches.append({
                    'id': match.get('id'),
                    'score': match.get('score'),
                    'metadata': match.get('metadata', {})
                })

            return matches

        except Exception as e:
            print(f"❌ Error querying index: {str(e)}")
            return []

    def query_both_namespaces(self, query_text: str, top_k: int = None,
                              filter_dict: Dict = None) -> List[Dict]:
        """Query both holdings and trades namespaces and combine results."""
        top_k = top_k or config.TOP_K_CHUNKS

        # Query both namespaces
        holdings_results = self.query(
            query_text,
            namespace=config.PINECONE_NAMESPACE_HOLDINGS,
            top_k=top_k,
            filter_dict=filter_dict
        )

        trades_results = self.query(
            query_text,
            namespace=config.PINECONE_NAMESPACE_TRADES,
            top_k=top_k,
            filter_dict=filter_dict
        )

        # Combine and sort by score
        all_results = holdings_results + trades_results
        all_results.sort(key=lambda x: x['score'], reverse=True)

        # Return top K from combined results
        return all_results[:top_k]

    def delete_namespace(self, namespace: str):
        """Delete all vectors in a namespace."""
        if not self.index:
            print("❌ Index not initialized.")
            return False

        try:
            self.index.delete(delete_all=True, namespace=namespace)
            print(f"✅ Deleted all vectors from namespace '{namespace}'")
            return True
        except Exception as e:
            print(f"❌ Error deleting namespace: {str(e)}")
            return False

    def get_index_stats(self) -> Dict:
        """Get statistics about the index."""
        if not self.index:
            return {}

        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'namespaces': stats.get('namespaces', {})
            }
        except Exception as e:
            print(f"❌ Error getting index stats: {str(e)}")
            return {}


if __name__ == "__main__":
    # Test the vector database
    print("🧪 Testing Vector Database...")

    # Initialize
    vdb = VectorDatabase()

    if vdb.initialize_pinecone():
        # Test embedding
        test_text = "MNC Investment Fund has significant holdings in MSFT equity"
        embedding = vdb.embed_text(test_text)
        print(f"\n✅ Test embedding generated: {len(embedding)} dimensions")

        # Get stats
        stats = vdb.get_index_stats()
        print(f"\n📊 Index Statistics:")
        print(f"  Total vectors: {stats.get('total_vectors', 0)}")
        print(f"  Dimension: {stats.get('dimension', 0)}")
        print(f"  Namespaces: {list(stats.get('namespaces', {}).keys())}")

        print("\n✅ Vector database test complete")
    else:
        print("\n❌ Vector database initialization failed")
