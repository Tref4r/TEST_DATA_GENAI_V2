"""
Example implementation of vector database operations.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from src.config import create_qdrant_client, create_collection, get_device
from qdrant_client import models
import numpy as np

def main():
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2', device=get_device())
    
    # Create Qdrant client and collection
    client = create_qdrant_client()
    collection_name = "example_collection"
    
    # Delete collection if it exists
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    
    # Create collection with vector size matching the model
    create_collection(client, collection_name, vector_size=384)
    
    # Example data
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps across a sleeping hound",
        "The lazy dog sleeps peacefully in the sun"
    ]
    
    # Generate embeddings
    embeddings = model.encode(texts)
    
    # Upload points to Qdrant
    client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": idx,
                "vector": embedding.tolist(),
                "payload": {"text": text}
            }
            for idx, (text, embedding) in enumerate(zip(texts, embeddings))
        ]
    )
    
    # Example search using basic vector search
    query = "fox jumping"
    query_vector = model.encode(query)
    
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        with_payload=True,
        limit=2
    )
    
    print("\nSearch Results for query:", query)
    for hit in search_result:
        print(f"Text: {hit.payload['text']}")
        print(f"Score: {hit.score}\n")

if __name__ == "__main__":
    main()
