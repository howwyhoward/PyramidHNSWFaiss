#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <iomanip>

#include "../include/pyramid.h"
#include "../include/similarity.h"

// Function to read .fvecs file (base/query vectors)
std::vector<float> read_fvecs(const std::string& filename, int& num_vectors, int& dim) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    file.read(reinterpret_cast<char*>(&dim), sizeof(int)); // Read dimension

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    num_vectors = file_size / ((1 + dim) * sizeof(float)); // Calculate number of vectors

    std::vector<float> data(num_vectors * dim);
    file.seekg(0, std::ios::beg);
    
    for (int i = 0; i < num_vectors; i++) {
        int d;
        file.read(reinterpret_cast<char*>(&d), sizeof(int)); // Read dimension (should match)
        if (d != dim) {
            std::cerr << "Dimension mismatch in " << filename << std::endl;
            exit(1);
        }
        file.read(reinterpret_cast<char*>(&data[i * dim]), dim * sizeof(float)); // Read vector data
    }

    file.close();
    return data;
}

// Function to read .ivecs file (ground truth neighbors)
std::vector<std::vector<int>> read_ivecs(const std::string& filename, int num_queries, int k) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening ground truth file: " << filename << std::endl;
        exit(1);
    }

    std::vector<std::vector<int>> ground_truth(num_queries);

    for (int i = 0; i < num_queries; i++) {
        int d;
        file.read(reinterpret_cast<char*>(&d), sizeof(int)); // Read the number of neighbors stored

        if (i == 0) {
            std::cout << "Ground truth file contains k = " << d << " neighbors per query" << std::endl;
        }

        if (d < k) {
            std::cerr << "Error: Ground truth file contains only " << d << " neighbors, but we need " << k << std::endl;
            exit(1);
        }

        ground_truth[i].resize(d);
        file.read(reinterpret_cast<char*>(ground_truth[i].data()), d * sizeof(int));
    }

    file.close();
    return ground_truth;
}

// Function to compute recall@k (accuracy comparison between results & ground truth)
float compute_recall(const std::vector<int>& result_indices,
                     const std::vector<std::vector<int>>& ground_truth,
                     int num_queries, int k) {
    int correct = 0;
    
    for (int i = 0; i < num_queries; i++) {
        std::unordered_set<int> true_neighbors(ground_truth[i].begin(), ground_truth[i].end());

        for (int j = 0; j < k; j++) {
            if (true_neighbors.find(result_indices[i * k + j]) != true_neighbors.end()) {
                correct++;
            }
        }
    }

    return static_cast<float>(correct) / (num_queries * k);
}

int main() {
    int num_base, num_queries, dim;
    int k = 100; // Number of nearest neighbors to find
    int num_clusters = 10; // Number of partitions to create

    // Load dataset
    std::vector<float> base_vectors = read_fvecs("data/siftsmall/siftsmall_base.fvecs", num_base, dim);
    std::cout << "Loaded " << num_base << " base vectors with dimension " << dim << std::endl;

    std::vector<float> query_vectors = read_fvecs("data/siftsmall/siftsmall_query.fvecs", num_queries, dim);
    std::cout << "Loaded " << num_queries << " query vectors" << std::endl;

    std::vector<std::vector<int>> ground_truth = read_ivecs("data/siftsmall/siftsmall_groundtruth.ivecs", num_queries, k);
    std::cout << "Loaded ground truth for " << num_queries << " queries" << std::endl;

    // Optional: Normalize vectors for angular similarity
    // pyramid::normalize_dataset(base_vectors.data(), num_base, dim);
    // pyramid::normalize_dataset(query_vectors.data(), num_queries, dim);

    // Create and build Pyramid index
    std::cout << "\nBuilding Pyramid index with " << num_clusters << " partitions..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    pyramid::PyramidGraph pyramid(dim, num_clusters);
    // Call to Algorithm 3: Pyramid Index Construction
    pyramid.build(base_vectors.data(), num_base);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Indexed " << pyramid.ntotal() << " vectors in " << build_time << " ms\n";

    // Perform k-NN search with Pyramid
    std::vector<int> result_indices(num_queries * k);
    std::vector<float> result_distances(num_queries * k);
    
    std::cout << "\nPerforming " << num_queries << " queries..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_queries; i++) {
        // Call to Algorithm 4: Pyramid Query Processing
        pyramid.search(query_vectors.data() + i * dim, k, 
                      result_indices.data() + i * k, 
                      result_distances.data() + i * k);
    }
    
    end_time = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double qps = num_queries / (search_time / 1000.0);
    
    std::cout << "Search completed in " << search_time << " ms (" 
              << std::fixed << std::setprecision(2) << qps << " queries per second)\n";

    // Display Pyramid query results
    std::cout << "\nQuery Results (Top-" << k << " neighbors for first 5 queries):\n";
    for (int i = 0; i < std::min(5, num_queries); i++) {
        std::cout << "Query " << i << ": ";
        for (int j = 0; j < std::min(5, k); j++) { // Show only first 5 neighbors
            std::cout << "(" << result_indices[i * k + j] << ", " 
                     << result_distances[i * k + j] << ") ";
        }
        std::cout << "..." << std::endl;
    }

    // Compute recall@k
    float recall = compute_recall(result_indices, ground_truth, num_queries, k);
    std::cout << "\nRecall@" << k << " = " << recall * 100 << "%" << std::endl;

    return 0;
}

