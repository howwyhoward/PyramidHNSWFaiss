#pragma once

#include <vector>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

namespace pyramid {

/**
 * Perform k-means clustering on a dataset using FAISS
 *
 * @param dataset Pointer to the dataset vectors
 * @param n Number of vectors in the dataset
 * @param dim Dimension of feature vectors
 * @param k Number of clusters to create
 * @param cluster_centers Output array for cluster centers (size: k * dim)
 * @param assignments Output array for cluster assignments (size: n)
 * @param niter Number of iterations for k-means (default: 25)
 * @param verbose Whether to print progress information (default: false)
 * @return True if clustering succeeded, false otherwise
 */
bool kmeans_cluster(const float* dataset, size_t n, int dim, int k,
                    float* cluster_centers, int* assignments, 
                    int niter = 25, bool verbose = false);

/**
 * Helper function to copy faiss::idx_t values to int array
 *
 * @param src Source array of faiss::idx_t values
 * @param dst Destination array of int values
 * @param n Number of elements to copy
 */
void copy_idx_to_int(const faiss::idx_t* src, int* dst, size_t n);

/**
 * Assign data points to their nearest cluster
 *
 * @param dataset Pointer to the dataset vectors
 * @param n Number of vectors in the dataset
 * @param dim Dimension of feature vectors
 * @param cluster_centers Array of cluster centers
 * @param k Number of clusters
 * @param assignments Output array for cluster assignments (size: n)
 */
void assign_to_clusters(const float* dataset, size_t n, int dim, 
                      const float* cluster_centers, int k, int* assignments);

/**
 * Extract vectors belonging to each cluster
 *
 * @param dataset Pointer to the dataset vectors
 * @param n Number of vectors in the dataset
 * @param dim Dimension of feature vectors
 * @param assignments Cluster assignments for each data point
 * @param k Number of clusters
 * @return Vector of vectors containing the indices of points in each cluster
 */
std::vector<std::vector<int>> extract_cluster_members(const float* dataset, size_t n, 
                                                     const int* assignments, int k);

} // namespace pyramid 