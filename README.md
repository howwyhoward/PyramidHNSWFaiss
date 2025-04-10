# Pyramid Search with FAISS

## Overview
Pyramid Search is a hierarchical, partitioned similarity search framework that enhances FAISS-based Approximate Nearest Neighbor (ANN) search by:
- Constructing a Meta-HNSW Graph to route queries efficiently.
- Partitioning the dataset into sub-indexes for faster local searches.
- Utilizing FAISS’s HNSW (Hierarchical Navigable Small World) graphs for optimized local queries.

This approach balances speed, scalability, and accuracy for large-scale nearest neighbor searches.

---

# How Pyramid Search Works

## Step 1: Load Dataset
- The dataset is loaded into an N x D matrix.
- All vectors are L2-normalized for cosine similarity.

## Step 2: Build Meta-HNSW Graph
- FAISS’s Hierarchical Navigable Small World (HNSW) index is used.
- The meta-graph routes queries efficiently.

## Step 3: Partition Dataset
- FAISS k-means clustering is used to create partitions.
- Each data point is assigned to its nearest cluster center.

## Step 4: Build Local Sub-HNSWs
- Each partition has its own HNSW index for fast local searches.

## Step 5: Query Execution
- Query first searches the Meta-HNSW.
- Top-2 partitions are selected for localized k-NN search.
- The best matches are aggregated and returned.

#To run current main.cpp from home directory
rm -rf build/
cmake -B build -G "Unix Makefiles"
make -C build -j$(nproc)
./build/pyramid_search

