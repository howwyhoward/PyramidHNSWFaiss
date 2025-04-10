#include "../include/search.h"
#include <faiss/Index.h>
#include <algorithm>
#include <vector>
#include <limits>

namespace pyramid {

// Perform k-NN search in a FAISS index
SearchResult search_index(const faiss::Index* index, const float* query, int k) {
    // Validate inputs
    if (!index || !query || k <= 0) {
        return SearchResult(0);
    }
    
    // Resize result containers
    SearchResult result(k);
    
    // Perform search
    index->search(1, query, k, result.distances.data(), result.indices.data());
    
    return result;
}

// Perform k-NN search in the Meta-HNSW to find relevant partitions
SearchResult find_partitions(const faiss::Index* meta_index, const float* query, int num_partitions) {
    // Validate inputs
    if (!meta_index || !query || num_partitions <= 0) {
        return SearchResult(0);
    }
    
    // Resize result containers
    SearchResult result(num_partitions);
    
    // Find closest partition centers
    meta_index->search(1, query, num_partitions, result.distances.data(), result.indices.data());
    
    return result;
}

// Combines multiple search results and sorts by distance
SearchResult merge_results(const std::vector<SearchResult>& results, int k) {
    // Count total number of results to merge
    size_t total_results = 0;
    for (const auto& result : results) {
        total_results += result.indices.size();
    }
    
    if (total_results == 0) {
        return SearchResult(0);
    }
    
    // Collect all indices and distances
    std::vector<std::pair<float, faiss::idx_t>> all_results;
    all_results.reserve(total_results);
    
    for (const auto& result : results) {
        for (size_t i = 0; i < result.indices.size(); i++) {
            // Skip invalid indices (marked as -1)
            if (result.indices[i] != -1) {
                all_results.emplace_back(result.distances[i], result.indices[i]);
            }
        }
    }
    
    // Sort by distance (ascending)
    std::sort(all_results.begin(), all_results.end());
    
    // Create final result with top-k
    int final_k = std::min(k, static_cast<int>(all_results.size()));
    SearchResult merged_result(final_k);
    
    for (int i = 0; i < final_k; i++) {
        merged_result.distances[i] = all_results[i].first;
        merged_result.indices[i] = all_results[i].second;
    }
    
    return merged_result;
}

} // namespace pyramid 