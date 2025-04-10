#pragma once

#include <vector>
#include <cmath>

namespace pyramid {

/**
 * Compute Euclidean distance between two vectors
 *
 * @param a First vector
 * @param b Second vector
 * @param dim Dimension of vectors
 * @return Squared Euclidean distance
 */
float euclidean_distance(const float* a, const float* b, int dim);

/**
 * Compute Angular distance (cosine similarity) between two vectors
 *
 * @param a First vector
 * @param b Second vector
 * @param dim Dimension of vectors
 * @return Angular distance (1 - cosine similarity)
 */
float angular_distance(const float* a, const float* b, int dim);

/**
 * Normalize a vector to unit length
 *
 * @param vec Vector to normalize
 * @param dim Dimension of the vector
 */
void normalize_vector(float* vec, int dim);

/**
 * Normalize a dataset of vectors to unit length
 *
 * @param data Dataset to normalize (modified in-place)
 * @param n Number of vectors
 * @param dim Dimension of each vector
 */
void normalize_dataset(float* data, size_t n, int dim);

} // namespace pyramid 