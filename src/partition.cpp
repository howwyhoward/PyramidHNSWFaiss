#include "../include/partition.h"
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <vector>
#include <iostream>
#include <random>
#include <unordered_set>

namespace pyramid {

// Perform k-means clustering on a dataset using FAISS
bool kmeans_cluster(const float* dataset, size_t n, int dim, int k,
                   float* cluster_centers, int* assignments, int niter, bool verbose) {
    // Handle edge cases
    if (n < k) {
        std::cerr << "Error: Number of points (" << n << ") is less than k (" << k << ")" << std::endl;
        return false;
    }
    
    try {
        // Create FAISS clustering object
        faiss::ClusteringParameters params;
        params.niter = niter;
        params.verbose = verbose;
        
        // Use L2 distance for clustering
        faiss::IndexFlatL2 index(dim);
        faiss::Clustering clustering(dim, k, params);
        
        // Prepare input data
        std::vector<float> data_copy(dataset, dataset + n * dim);
        
        // Run k-means clustering
        clustering.train(n, data_copy.data(), index);
        
        // Copy cluster centers
        std::copy(clustering.centroids.data(), clustering.centroids.data() + k * dim, cluster_centers);
        
        // Assign points to clusters
        assign_to_clusters(dataset, n, dim, cluster_centers, k, assignments);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during clustering: " << e.what() << std::endl;
        return false;
    }
}

void copy_idx_to_int(const faiss::idx_t* src, int* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = static_cast<int>(src[i]);
    }
}

void assign_to_clusters(const float* dataset, size_t n, int dim, 
                      const float* cluster_centers, int k, int* assignments) {
    // Create a FAISS index for the cluster centers
    faiss::IndexFlatL2 center_index(dim);
    center_index.add(k, cluster_centers);
    
    // Find nearest center for each point - need to use faiss::idx_t for labels
    std::vector<float> distances(n);
    std::vector<faiss::idx_t> idx_assignments(n);
    
    // Perform search with idx_t type for labels
    center_index.search(n, dataset, 1, distances.data(), idx_assignments.data());
    
    // Convert faiss::idx_t to int using our helper
    copy_idx_to_int(idx_assignments.data(), assignments, n);
}

std::vector<std::vector<int>> extract_cluster_members(const float* dataset, size_t n, 
                                                    const int* assignments, int k) {
    std::vector<std::vector<int>> clusters(k);
    
    // Assign each point to its cluster
    for (size_t i = 0; i < n; i++) {
        int cluster = assignments[i];
        if (cluster >= 0 && cluster < k) {
            clusters[cluster].push_back(i);
        }
    }
    
    return clusters;
}

} // namespace pyramid 