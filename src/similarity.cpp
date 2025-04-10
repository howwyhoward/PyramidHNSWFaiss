#include "../include/similarity.h"
#include <cmath>
#include <algorithm>

namespace pyramid {

float euclidean_distance(const float* a, const float* b, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

float angular_distance(const float* a, const float* b, int dim) {
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (int i = 0; i < dim; i++) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 1.0f;  // Maximum distance for zero vectors
    }
    
    // Compute cosine similarity and convert to distance
    float cosine = dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
    
    // Clamp to [-1, 1] to handle floating point errors
    cosine = std::max(-1.0f, std::min(1.0f, cosine));
    
    // Return angular distance (1 - cosine similarity)
    return 1.0f - cosine;
}

void normalize_vector(float* vec, int dim) {
    float norm = 0.0f;
    
    // Compute L2 norm
    for (int i = 0; i < dim; i++) {
        norm += vec[i] * vec[i];
    }
    
    norm = std::sqrt(norm);
    
    // Avoid division by zero
    if (norm > 0.0f) {
        for (int i = 0; i < dim; i++) {
            vec[i] /= norm;
        }
    }
}

void normalize_dataset(float* data, size_t n, int dim) {
    // Normalize each vector
    for (size_t i = 0; i < n; i++) {
        normalize_vector(data + i * dim, dim);
    }
}

} // namespace pyramid 