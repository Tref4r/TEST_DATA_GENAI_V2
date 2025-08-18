"""
Vector database configuration and setup utilities.
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
import torch

def get_device():
    """Get the appropriate device for computations."""
    if torch.cuda.is_available():
        # For RTX 4050 with 6GB VRAM, we'll use CUDA
        return torch.device("cuda")
    return torch.device("cpu")

def create_qdrant_client(host="localhost", port=6333):
    """Create and return a Qdrant client instance."""
    return QdrantClient(host=host, port=port)

def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 384,
    distance: models.Distance = models.Distance.COSINE
):
    """Create a new collection in Qdrant."""
    return client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=distance
        )
    )
