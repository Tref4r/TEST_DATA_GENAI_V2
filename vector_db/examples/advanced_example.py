"""
Advanced implementation showcasing Qdrant's hybrid search and filtering capabilities.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from src.config import create_qdrant_client, create_collection, get_device
from qdrant_client import models
import numpy as np
from datetime import datetime

def main():
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2', device=get_device())
    
    # Create Qdrant client and collection with named vectors
    client = create_qdrant_client()
    collection_name = "advanced_collection"
    
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
    
    # Example data with metadata for filtering
    texts = [
        {
            "text": "The quick brown fox jumps over the lazy dog",
            "category": "animal_action",
            "created_at": "2025-08-18"
        },
        {
            "text": "A fast brown fox leaps across a sleeping hound",
            "category": "animal_action",
            "created_at": "2025-08-17"
        },
        {
            "text": "The lazy dog sleeps peacefully in the sun",
            "category": "animal_rest",
            "created_at": "2025-08-16"
        }
    ]
    
    # Generate embeddings
    dense_embeddings = model.encode([doc["text"] for doc in texts])
    # Simulate sparse embeddings (in real case, use proper sparse encoding)
    sparse_embeddings = dense_embeddings  # Just for demonstration
    
    # Upload points with vectors and metadata
    points = []
    for idx, (doc, embedding, _) in enumerate(zip(texts, dense_embeddings, sparse_embeddings)):
        point = models.PointStruct(
            id=idx,
            vector=embedding.tolist(),  # Using single vector field
            payload=doc
        )
        points.append(point)
    
    client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points
    )
    
    # Example search with filtering using newer API
    query = "fox jumping"
    query_vector = model.encode(query)
    
    # Using search with filtering
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        query_filter={
            "must": [
                {
                    "key": "category",
                    "match": {"value": "animal_action"}
                },
                {
                    "key": "created_at",
                    "match": {"value": "2025-08-17"}
                }
            ]
        },
        with_payload=True,
        score_threshold=0.5,
        limit=2
    )
    
    print("\nHybrid Search Results for query:", query)
    print("With category='animal_action' and created_at >= '2025-08-17'")
    for hit in search_result:
        print(f"\nText: {hit.payload['text']}")
        print(f"Category: {hit.payload['category']}")
        print(f"Created: {hit.payload['created_at']}")
        print(f"Score: {hit.score}")

if __name__ == "__main__":
    main()
