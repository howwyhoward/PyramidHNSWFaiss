#pragma once

#include <vector>
#include <memory>
#include <faiss/Index.h>
#include "pyramid.h"

namespace pyramid {

/**
 * Search result structure containing indices and distances
 */
struct SearchResult {
    std::vector<faiss::idx_t> indices;  // Indices of nearest neighbors
    std::vector<float> distances;       // Distances to nearest neighbors
    
    SearchResult(int k) : indices(k), distances(k) {}
};

/**
 * Perform k-NN search in a FAISS index
 *
 * @param index FAISS index to search in
 * @param query Query vector
 * @param k Number of neighbors to return
 * @return SearchResult containing indices and distances
 */
SearchResult search_index(const faiss::Index* index, const float* query, int k);

/**
 * Perform k-NN search in the Meta-HNSW to find relevant partitions
 *
 * @param meta_index Meta-HNSW index
 * @param query Query vector
 * @param num_partitions Number of partitions to return
 * @return SearchResult containing partition indices and distances
 */
SearchResult find_partitions(const faiss::Index* meta_index, const float* query, int num_partitions);

/**
 * Merge multiple search results and sort by distance
 *
 * @param results Vector of search results to merge
 * @param k Number of neighbors to keep in the final result
 * @return Merged SearchResult containing the top-k neighbors
 */
SearchResult merge_results(const std::vector<SearchResult>& results, int k);

} // namespace pyramid 