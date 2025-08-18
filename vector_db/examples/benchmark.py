"""
Benchmarking utilities for vector database operations.
"""
import time
import statistics
from functools import wraps
import logging
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import create_qdrant_client, create_collection, get_device
from sentence_transformers import SentenceTransformer
from qdrant_client import models

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper

class VectorDBBenchmark:
    def __init__(self, collection_name="benchmark_collection"):
        self.client = create_qdrant_client()
        self.collection_name = collection_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=get_device())
        
        # Reset collection
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
        
        # Create the collection with proper vector size
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "size": 384,  # Size from all-MiniLM-L6-v2 model
                "distance": "Cosine"
            }
        )
        
        self.results = {
            'insertion': [],
            'search': [],
            'batch_search': [],
            'update': [],
            'delete': []
        }
    
    @timer_decorator
    def insert_point(self, text, id):
        vector = self.model.encode(text)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[{
                'id': id,
                'vector': vector.tolist(),
                'payload': {'text': text}
            }]
        )
    
    @timer_decorator
    def search_point(self, query):
        vector = self.model.encode(query)
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=vector.tolist(),
            limit=5
        )
    
    @timer_decorator
    def batch_search(self, queries):
        vectors = self.model.encode(queries)
        requests = [
            models.SearchRequest(
                vector=vector.tolist(),
                limit=5
            )
            for vector in vectors
        ]
        return self.client.search_batch(
            collection_name=self.collection_name,
            requests=requests
        )
    
    @timer_decorator
    def update_point(self, id, new_text):
        vector = self.model.encode(new_text)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[{
                'id': id,
                'vector': vector.tolist(),
                'payload': {'text': new_text}
            }]
        )
    
    @timer_decorator
    def delete_point(self, id):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[id]
        )
    
    def run_benchmark(self, num_points=1000):
        logger.info(f"Starting benchmark with {num_points} points...")
        
        # Test data
        texts = [
            f"This is test document {i} with some random words for embedding"
            for i in range(num_points)
        ]
        
        # Test insertions
        logger.info("Testing insertions...")
        for i, text in enumerate(texts):
            _, time_taken = self.insert_point(text, i)
            self.results['insertion'].append(time_taken)
        
        # Test searches
        logger.info("Testing single searches...")
        queries = ["test document", "random words", "embedding search"]
        for query in queries:
            _, time_taken = self.search_point(query)
            self.results['search'].append(time_taken)
        
        # Test batch searches
        logger.info("Testing batch searches...")
        _, time_taken = self.batch_search(queries)
        self.results['batch_search'].append(time_taken)
        
        # Test updates
        logger.info("Testing updates...")
        for i in range(min(10, num_points)):
            _, time_taken = self.update_point(
                i, f"Updated document {i} with new content"
            )
            self.results['update'].append(time_taken)
        
        # Test deletions
        logger.info("Testing deletions...")
        for i in range(min(10, num_points)):
            _, time_taken = self.delete_point(i)
            self.results['delete'].append(time_taken)
        
        self._print_results()
    
    def _print_results(self):
        # Create results string
        current_time = time.strftime("%Y%m%d_%H%M%S")
        results = ["\nBenchmark Results:"]
        results.append("=" * 50)
        
        for operation, times in self.results.items():
            if times:
                avg_time = statistics.mean(times)
                std_dev = statistics.stdev(times) if len(times) > 1 else 0
                results.extend([
                    f"\n{operation.capitalize()}:",
                    f"  Average time: {avg_time:.4f} seconds",
                    f"  Standard deviation: {std_dev:.4f} seconds",
                    f"  Number of operations: {len(times)}"
                ])
        
        # Print to console
        print("\n".join(results))
        
        # Save to file
        filename = f"benchmark_results_{self.collection_name}_{current_time}.txt"
        filepath = Path(__file__).parent.parent / "benchmark_results" / filename
        
        # Create benchmark_results directory if it doesn't exist
        filepath.parent.mkdir(exist_ok=True)
        
        # Write results to file
        with open(filepath, "w") as f:
            f.write("\n".join([
                f"Benchmark Results for collection: {self.collection_name}",
                f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "=" * 50,
                ""
            ]))
            f.write("\n".join(results))
            
        print(f"\nResults saved to: {filepath}")

def main():
    # Run benchmark with different sizes
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\nRunning benchmark with {size} points")
        print("=" * 50)
        benchmark = VectorDBBenchmark(f"benchmark_{size}")
        benchmark.run_benchmark(size)

if __name__ == "__main__":
    main()
