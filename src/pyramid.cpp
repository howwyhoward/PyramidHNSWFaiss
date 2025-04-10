#include "../include/pyramid.h"
#include "../include/partition.h"
#include "../include/search.h"
#include <faiss/IndexFlat.h>
#include <faiss/Clustering.h>
#include <iostream>
#include <algorithm>
#include <memory>

namespace pyramid {

PyramidGraph::PyramidGraph(int dim, int num_clusters, int m, int ef_construction, int ef_search)
    : dim_(dim), num_clusters_(num_clusters), 
      m_(m), ef_construction_(ef_construction), ef_search_(ef_search),
      total_vectors_(0) {
    
    // Initialize the meta-graph
    meta_graph_ = std::make_unique<faiss::IndexHNSWFlat>(dim_, m_);
    meta_graph_->hnsw.efConstruction = ef_construction_;
    meta_graph_->hnsw.efSearch = ef_search_;
    
    // Initialize partition structures
    sub_graphs_.resize(num_clusters_);
    partition_indices_.resize(num_clusters_);
}

PyramidGraph::~PyramidGraph() = default;

void PyramidGraph::build(const float* dataset, size_t n) {
    // IMPLEMENTATION OF ALGORITHM 3: Pyramid Index Construction
    total_vectors_ = n;
    
    // Step 3-4: Partition the dataset using k-means clustering
    std::vector<int> cluster_assignments = partition_data(dataset, n);
    
    // Step 5: Extract cluster centers and build the meta-HNSW graph
    std::vector<float> centers = extract_cluster_centers(dataset, n, cluster_assignments);
    meta_graph_->add(num_clusters_, centers.data());
    
    // Step 6-10: Partition dataset and assign items to sub-datasets
    for (size_t i = 0; i < n; i++) {
        int cluster = cluster_assignments[i];
        partition_indices_[cluster].push_back(i);
    }
    
    // Step 11-12: Build sub-HNSW graphs for each partition
    for (int c = 0; c < num_clusters_; c++) {
        // Skip empty partitions
        if (partition_indices_[c].empty()) {
            continue;
        }
        
        // Create sub-graph for this partition
        sub_graphs_[c] = std::make_unique<faiss::IndexHNSWFlat>(dim_, m_);
        sub_graphs_[c]->hnsw.efConstruction = ef_construction_;
        sub_graphs_[c]->hnsw.efSearch = ef_search_;
        
        // Extract vectors for this cluster
        const size_t cluster_size = partition_indices_[c].size();
        std::vector<float> cluster_data(cluster_size * dim_);
        
        for (size_t i = 0; i < cluster_size; i++) {
            const size_t idx = partition_indices_[c][i];
            std::copy(dataset + idx * dim_, dataset + (idx + 1) * dim_, cluster_data.data() + i * dim_);
        }
        
        // Add vectors to the sub-graph
        sub_graphs_[c]->add(cluster_size, cluster_data.data());
    }
}

void PyramidGraph::search(const float* query, int k, int* indices, float* distances) const {
    // IMPLEMENTATION OF ALGORITHM 4: Pyramid Query Processing
    
    // Step 3-4: Find the top partitions using the meta-HNSW graph
    const int num_partitions_to_search = std::min(2, num_clusters_); // Search in top-2 partitions
    
    std::vector<float> partition_distances(num_partitions_to_search);
    std::vector<faiss::idx_t> partition_indices(num_partitions_to_search);
    
    meta_graph_->search(1, query, num_partitions_to_search, 
                       partition_distances.data(), partition_indices.data());
    
    // Prepare for merging results (Initialize resSet)
    std::vector<faiss::idx_t> all_indices;
    std::vector<float> all_distances;
    
    // Step 5-8: Search in each selected partition that contains neighbors
    for (int p = 0; p < num_partitions_to_search; p++) {
        int partition_idx = partition_indices[p];
        
        // Skip if partition is empty or doesn't exist
        if (partition_idx >= num_clusters_ || !sub_graphs_[partition_idx] || 
            partition_indices_[partition_idx].empty()) {
            continue;
        }
        
        // Step 7: Search within this partition's sub-HNSW graph
        const int local_k = std::min(k, static_cast<int>(partition_indices_[partition_idx].size()));
        std::vector<float> local_distances(local_k);
        std::vector<faiss::idx_t> local_indices(local_k);
        
        sub_graphs_[partition_idx]->search(1, query, local_k, 
                                          local_distances.data(), local_indices.data());
        
        // Step 8: Add results to resSet
        for (int i = 0; i < local_k; i++) {
            all_indices.push_back(partition_indices_[partition_idx][local_indices[i]]);
            all_distances.push_back(local_distances[i]);
        }
    }
    
    // If we have no results, return empty
    if (all_indices.empty()) {
        for (int i = 0; i < k; i++) {
            indices[i] = -1;
            distances[i] = std::numeric_limits<float>::max();
        }
        return;
    }
    
    // Step 9: Extract the top k neighbors from resSet
    std::vector<std::pair<float, faiss::idx_t>> sorted_results;
    for (size_t i = 0; i < all_indices.size(); i++) {
        sorted_results.emplace_back(all_distances[i], all_indices[i]);
    }
    
    std::sort(sorted_results.begin(), sorted_results.end());
    
    // Copy results to output arrays
    const int result_k = std::min(k, static_cast<int>(sorted_results.size()));
    for (int i = 0; i < result_k; i++) {
        distances[i] = sorted_results[i].first;
        indices[i] = sorted_results[i].second;
    }
    
    // Fill any remaining slots with -1
    for (int i = result_k; i < k; i++) {
        indices[i] = -1;
        distances[i] = std::numeric_limits<float>::max();
    }
}

std::vector<int> PyramidGraph::partition_data(const float* dataset, size_t n) {
    std::vector<int> assignments(n);
    std::vector<float> centroids(num_clusters_ * dim_);
    
    // Perform k-means clustering
    bool success = kmeans_cluster(dataset, n, dim_, num_clusters_, 
                               centroids.data(), assignments.data());
    
    if (!success) {
        std::cerr << "K-means clustering failed!" << std::endl;
        // Fall back to simple assignment if k-means fails
        for (size_t i = 0; i < n; i++) {
            assignments[i] = i % num_clusters_;
        }
    }
    
    return assignments;
}

std::vector<float> PyramidGraph::extract_cluster_centers(const float* dataset, size_t n, 
                                                      const std::vector<int>& cluster_assign) {
    std::vector<float> centers(num_clusters_ * dim_, 0.0f);
    std::vector<int> counts(num_clusters_, 0);
    
    // Sum vectors in each cluster
    for (size_t i = 0; i < n; i++) {
        int cluster = cluster_assign[i];
        if (cluster < 0 || cluster >= num_clusters_) {
            continue;  // Skip invalid assignments
        }
        
        for (int d = 0; d < dim_; d++) {
            centers[cluster * dim_ + d] += dataset[i * dim_ + d];
        }
        counts[cluster]++;
    }
    
    // Compute average for each cluster
    for (int c = 0; c < num_clusters_; c++) {
        if (counts[c] > 0) {
            for (int d = 0; d < dim_; d++) {
                centers[c * dim_ + d] /= counts[c];
            }
        }
    }
    
    return centers;
}

} // namespace pyramid 