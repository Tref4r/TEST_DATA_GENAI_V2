"""
Visualization and analysis utilities for vector database benchmarks
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import create_qdrant_client, get_device
from qdrant_client import models
from sklearn.feature_extraction.text import TfidfVectorizer

def plot_benchmark_results(results_file):
    """Generate visualizations from benchmark results"""
    # Read benchmark results
    with open(results_file, 'r') as f:
        lines = f.readlines()
    
    # Parse the results
    metrics = {}
    current_op = None
    
    for line in lines:
        if 'Average time:' in line:
            time = float(line.split(':')[1].split()[0])
            metrics[current_op] = {'avg_time': time}
        elif 'Standard deviation:' in line:
            std = float(line.split(':')[1].split()[0])
            metrics[current_op]['std'] = std
        elif ':' in line and 'Results' not in line and 'Date' not in line:
            current_op = line.strip(':').strip()
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Set style to a default clean style
    plt.style.use('default')
    
    # Set custom colors
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Set background style
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_facecolor('#f8f9fa')
    
    # Plot 1: Average Operation Times with Error Bars
    operations = df.index
    avg_times = df['avg_time']
    std_times = df['std']
    
    bars1 = ax1.bar(operations, avg_times, yerr=std_times, capsize=5)
    ax1.set_title('Operation Latency', pad=20)
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s', ha='center', va='bottom')
    
    # Plot 2: Operations per Second (Throughput)
    throughput = 1 / avg_times
    bars2 = ax2.bar(operations, throughput)
    ax2.set_title('Operation Throughput', pad=20)
    ax2.set_ylabel('Operations per Second')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}/s', ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    output_file = results_file.replace('.txt', '_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def compare_search_methods(collection_size=100):
    """Compare pure dense search vs hybrid search performance"""
    # Initialize models
    model = SentenceTransformer('all-MiniLM-L6-v2', device=get_device())
    client = create_qdrant_client()
    
    # Test data
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps across a sleeping hound",
        "The lazy dog sleeps peacefully in the sun",
        "A document about machine learning and AI",
        "Another text about artificial intelligence",
        "Something completely different about databases",
        "Deep learning models perform well on NLP tasks",
        "Neural networks are revolutionizing AI research",
        "Big data analytics helps business decisions",
        "Cloud computing enables scalable solutions"
    ]
    
    # Create collections for both methods
    collections = {
        "dense": "dense_search",
        "hybrid": "hybrid_search"
    }
    
    for name, collection in collections.items():
        if client.collection_exists(collection):
            client.delete_collection(collection)
        
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )
    
    # Generate embeddings
    dense_embeddings = model.encode(texts)
    
    # Generate sparse embeddings
    vectorizer = TfidfVectorizer(max_features=384)
    vectorizer.fit(texts)
    sparse_embeddings = vectorizer.transform(texts).toarray()
    
    # Pad sparse vectors if needed
    if sparse_embeddings.shape[1] < 384:
        padding = np.zeros((sparse_embeddings.shape[0], 384 - sparse_embeddings.shape[1]))
        sparse_embeddings = np.hstack([sparse_embeddings, padding])
    
    # Upload points for dense search
    dense_points = []
    for idx, (text, dense_emb) in enumerate(zip(texts, dense_embeddings)):
        point = models.PointStruct(
            id=idx,
            vector=dense_emb.tolist(),
            payload={"text": text}
        )
        dense_points.append(point)
    
    client.upsert(
        collection_name=collections["dense"],
        points=dense_points
    )
    
    # Upload points for hybrid search
    hybrid_points = []
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
        hybrid_points.append(point)
    
    client.upsert(
        collection_name=collections["hybrid"],
        points=hybrid_points
    )
    
    # Test queries
    test_queries = [
        "machine learning AI",  # Semantic + keyword match
        "quick animal jumping",  # Semantic understanding
        "neural networks deep learning",  # Technical terms
        "peaceful sleep",  # Contextual understanding
        "data analysis business",  # Mixed concepts
    ]
    
    results = {
        "dense": {},
        "hybrid": {}
    }
    
    # Run searches
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("=" * 50)
        
        # Dense search
        query_vector = model.encode(query)
        dense_results = client.search(
            collection_name=collections["dense"],
            query_vector=query_vector.tolist(),
            limit=3
        )
        
        # Hybrid search
        sparse_vector = vectorizer.transform([query]).toarray()[0]
        if len(sparse_vector) < 384:
            padding = np.zeros(384 - len(sparse_vector))
            sparse_vector = np.concatenate([sparse_vector, padding])
        
        dense_norm = np.linalg.norm(query_vector)
        sparse_norm = np.linalg.norm(sparse_vector)
        
        if dense_norm > 0:
            query_vector = query_vector / dense_norm
        if sparse_norm > 0:
            sparse_vector = sparse_vector / sparse_norm
        
        combined_vector = 0.5 * query_vector + 0.5 * sparse_vector
        
        hybrid_results = client.search(
            collection_name=collections["hybrid"],
            query_vector=combined_vector.tolist(),
            limit=3
        )
        
        # Store and print results
        print("\nDense Search Results:")
        for hit in dense_results:
            print(f"Score: {hit.score:.4f} - {hit.payload['text']}")
        
        print("\nHybrid Search Results:")
        for hit in hybrid_results:
            print(f"Score: {hit.score:.4f} - {hit.payload['text']}")
        
        # Store average scores
        results["dense"][query] = sum(hit.score for hit in dense_results) / len(dense_results)
        results["hybrid"][query] = sum(hit.score for hit in hybrid_results) / len(hybrid_results)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    queries = list(results["dense"].keys())
    dense_scores = [results["dense"][q] for q in queries]
    hybrid_scores = [results["hybrid"][q] for q in queries]
    
    x = np.arange(len(queries))
    width = 0.35
    
    plt.bar(x - width/2, dense_scores, width, label='Dense Search')
    plt.bar(x + width/2, hybrid_scores, width, label='Hybrid Search')
    
    plt.ylabel('Average Similarity Score')
    plt.title('Dense vs Hybrid Search Performance Comparison')
    plt.xticks(x, [q[:20] + '...' if len(q) > 20 else q for q in queries], rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    comparison_file = f'benchmark_results/search_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_file

if __name__ == "__main__":
    # Process all benchmark result files
    results_dir = Path(__file__).parent.parent / "benchmark_results"
    for results_file in results_dir.glob("benchmark_results_*.txt"):
        if not results_file.name.endswith('_visualization.png'):
            print(f"Generating visualization for {results_file.name}")
            viz_file = plot_benchmark_results(str(results_file))
            print(f"Visualization saved to {viz_file}")
    
    # Run search comparison
    print("\nRunning search method comparison...")
    comparison_file = compare_search_methods()
    print(f"Search comparison visualization saved to {comparison_file}")
