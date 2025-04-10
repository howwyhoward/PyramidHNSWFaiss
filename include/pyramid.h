#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>

namespace pyramid {

/**
 * PyramidGraph - Main class for the Pyramid HNSW implementation
 * 
 * This class implements a hierarchical search structure:
 * - A meta-HNSW graph for routing queries to relevant partitions
 * - Multiple sub-HNSW graphs for local search within partitions
 */
class PyramidGraph {
public:
    /**
     * Initialize a new PyramidGraph
     * 
     * @param dim Dimension of the feature vectors
     * @param num_clusters Number of partitions to create
     * @param m Number of connections per node in HNSW graph (default: 32)
     * @param ef_construction Size of the dynamic candidate list during construction (default: 40)
     * @param ef_search Size of the candidate list during search (default: 16)
     */
    PyramidGraph(int dim, int num_clusters, 
                int m = 32, int ef_construction = 40, int ef_search = 16);
    
    /**
     * Destructor
     */
    ~PyramidGraph();
    
    /**
     * Build the pyramid structure from a dataset
     * 
     * @param dataset Pointer to the dataset vectors
     * @param n Number of vectors in the dataset
     */
    void build(const float* dataset, size_t n);
    
    /**
     * Search for k nearest neighbors to the query vector
     * 
     * @param query Pointer to the query vector
     * @param k Number of neighbors to return
     * @param indices Output array for the indices of neighbors
     * @param distances Output array for the distances to neighbors
     */
    void search(const float* query, int k, int* indices, float* distances) const;

    /**
     * Get the number of vectors indexed
     */
    size_t ntotal() const {
        return total_vectors_;
    }

private:
    int dim_;                    // Dimension of feature vectors
    int num_clusters_;           // Number of partitions
    int m_;                      // Number of connections per node in HNSW graph
    int ef_construction_;        // Dynamic candidate list size during construction
    int ef_search_;              // Dynamic candidate list size during search
    size_t total_vectors_;       // Total number of vectors indexed
    
    std::unique_ptr<faiss::IndexHNSWFlat> meta_graph_;  // Top-level HNSW graph
    std::vector<std::unique_ptr<faiss::IndexHNSWFlat>> sub_graphs_;  // Sub-HNSW graphs for each partition
    std::vector<std::vector<faiss::idx_t>> partition_indices_;  // Mapping of which vectors belong to which partition
    
    /**
     * Partition the dataset using k-means clustering
     * 
     * @param dataset Pointer to the dataset vectors
     * @param n Number of vectors in the dataset
     * @return Vector of cluster assignments for each data point
     */
    std::vector<int> partition_data(const float* dataset, size_t n);
    
    /**
     * Extract cluster centers from k-means result
     * 
     * @param dataset Pointer to the dataset vectors
     * @param n Number of vectors in the dataset
     * @param cluster_assign Cluster assignments for each data point
     * @return Matrix of cluster centers
     */
    std::vector<float> extract_cluster_centers(const float* dataset, size_t n, 
                                             const std::vector<int>& cluster_assign);
};

} // namespace pyramid 