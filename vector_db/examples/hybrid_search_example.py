"""
Example implementation of hybrid search combining dense and sparse vectors.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from src.config import create_qdrant_client, create_collection, get_device
from qdrant_client import models
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

def create_sparse_vectors(texts, max_features=384):
    """Create sparse vectors using TF-IDF"""
    vectorizer = TfidfVectorizer(max_features=max_features)
    sparse_matrix = vectorizer.fit_transform(texts)
    return sparse_matrix.toarray()

def hybrid_search(client, collection_name, query, model, vectorizer, alpha=0.5):
    """
    Perform hybrid search combining dense and sparse vectors
    alpha: weight for dense vectors (1-alpha for sparse vectors)
    """
    # Generate dense embedding for query
    dense_vector = model.encode(query)
    
    # Generate sparse vector for query
    sparse_vector = vectorizer.transform([query]).toarray()[0]
    
    # Pad sparse vector if needed
    if len(sparse_vector) < 384:
        padding = np.zeros(384 - len(sparse_vector))
        sparse_vector = np.concatenate([sparse_vector, padding])
    
    # Normalize vectors
    dense_norm = np.linalg.norm(dense_vector)
    sparse_norm = np.linalg.norm(sparse_vector)
    
    if dense_norm > 0:
        dense_vector = dense_vector / dense_norm
    if sparse_norm > 0:
        sparse_vector = sparse_vector / sparse_norm
    
    # Combine vectors with weights
    combined_vector = alpha * dense_vector + (1 - alpha) * sparse_vector
    
    return client.search(
        collection_name=collection_name,
        query_vector=combined_vector.tolist(),
        with_payload=True,
        with_vectors=True,
        limit=3
    )

def main():
    # Initialize models
    model = SentenceTransformer('all-MiniLM-L6-v2', device=get_device())
    
    # Create Qdrant client and collection
    client = create_qdrant_client()
    collection_name = "hybrid_collection"
    
    # Check if collection exists and delete if it does
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    
    # Create collection with vector parameters
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )
    
    # Example data
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps across a sleeping hound",
        "The lazy dog sleeps peacefully in the sun",
        "A document about machine learning and AI",
        "Another text about artificial intelligence",
        "Something completely different about databases"
    ]
    
    # Generate dense embeddings
    dense_embeddings = model.encode(texts)
    
    # Generate sparse embeddings using TF-IDF with fixed vocabulary size
    vectorizer = TfidfVectorizer(max_features=384)
    # First fit the vectorizer on all texts to build vocabulary
    vectorizer.fit(texts)
    # Then transform each text
    sparse_embeddings = vectorizer.transform(texts).toarray()
    
    # Pad sparse vectors if needed (in case they're smaller than 384)
    if sparse_embeddings.shape[1] < 384:
        padding = np.zeros((sparse_embeddings.shape[0], 384 - sparse_embeddings.shape[1]))
        sparse_embeddings = np.hstack([sparse_embeddings, padding])
    
    # Upload points with combined vectors
    points = []
    for idx, (text, dense_emb, sparse_emb) in enumerate(zip(texts, dense_embeddings, sparse_embeddings)):
        # Normalize vectors
        dense_norm = np.linalg.norm(dense_emb)
        sparse_norm = np.linalg.norm(sparse_emb)
        
        if dense_norm > 0:
            dense_emb = dense_emb / dense_norm
        if sparse_norm > 0:
            sparse_emb = sparse_emb / sparse_norm
        
        # Combine vectors (50% dense, 50% sparse)
        combined_vector = 0.5 * dense_emb + 0.5 * sparse_emb
        
        point = models.PointStruct(
            id=idx,
            vector=combined_vector.tolist(),
            payload={"text": text}
        )
        points.append(point)
    
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    # Example queries to demonstrate hybrid search effectiveness
    queries = [
        "fox jumping",  # Should match semantic meaning
        "AI intelligence",  # Should match keywords
        "quick brown",  # Should match exact phrases
    ]
    
    print("\nHybrid Search Results:")
    print("=" * 50)
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = hybrid_search(client, collection_name, query, model, vectorizer)
        
        for hit in results:
            print(f"\nText: {hit.payload['text']}")
            print(f"Score: {hit.score:.4f}")

if __name__ == "__main__":
    main()
