#!/usr/bin/env python3
# To run this script:
# python utils/view_fvecs.py data/siftsmall/siftsmall_base.fvecs 5
# python utils/view_fvecs.py data/siftsmall/siftsmall_groundtruth.ivecs 5
# 
# Arguments:
#   1. Path to the .fvecs or .ivecs file
#   2. (Optional) Number of vectors to display (default: 5)

import numpy as np
import struct
import sys

def read_fvecs_vectors(filename, num_vectors=5):
    """Read the first num_vectors from a .fvecs file and return them as a list of numpy arrays"""
    vectors = []
    with open(filename, 'rb') as f:
        for i in range(num_vectors):
            try:
                # Read dimension (int32)
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break  # End of file
                dim = struct.unpack('i', dim_bytes)[0]
                
                # Read vector data (dim * float32)
                vector_bytes = f.read(dim * 4)
                if len(vector_bytes) < dim * 4:
                    break  # Incomplete vector
                
                # Unpack all floats
                vector = list(struct.unpack(f'{dim}f', vector_bytes))
                vectors.append((dim, vector))
            except Exception as e:
                print(f"Error reading vector {i}: {e}")
                break
    return vectors

def read_ivecs_vectors(filename, num_vectors=5):
    """Read the first num_vectors from a .ivecs file and return them as a list of numpy arrays"""
    vectors = []
    with open(filename, 'rb') as f:
        for i in range(num_vectors):
            try:
                # Read count (int32)
                count_bytes = f.read(4)
                if not count_bytes:
                    break  # End of file
                count = struct.unpack('i', count_bytes)[0]
                
                # Read vector data (count * int32)
                vector_bytes = f.read(count * 4)
                if len(vector_bytes) < count * 4:
                    break  # Incomplete vector
                
                # Unpack all integers
                vector = list(struct.unpack(f'{count}i', vector_bytes))
                vectors.append((count, vector))
            except Exception as e:
                print(f"Error reading vector {i}: {e}")
                break
    return vectors

def print_vectors(vectors, is_float=True):
    """Print vectors in a readable format"""
    for i, (dim, vector) in enumerate(vectors):
        print(f"Vector {i} (dim={dim}):")
        if is_float:
            # Truncate long float vectors
            if dim > 20:
                print(f"  [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}, ..., {vector[dim-1]:.4f}]")
                # Print some statistics
                print(f"  Min: {min(vector):.4f}, Max: {max(vector):.4f}, Mean: {sum(vector)/len(vector):.4f}")
            else:
                print("  [" + ", ".join([f"{x:.4f}" for x in vector]) + "]")
        else:
            # For integer vectors (like groundtruth), don't truncate as much
            if dim > 20:
                print(f"  [{vector[0]}, {vector[1]}, {vector[2]}, ..., {vector[dim-1]}]")
            else:
                print("  [" + ", ".join([str(x) for x in vector]) + "]")
        print()

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <filename.fvecs|filename.ivecs> [num_vectors]")
        return
    
    filename = sys.argv[1]
    num_vectors = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    if filename.endswith('.fvecs'):
        vectors = read_fvecs_vectors(filename, num_vectors)
        print_vectors(vectors, is_float=True)
    elif filename.endswith('.ivecs'):
        vectors = read_ivecs_vectors(filename, num_vectors)
        print_vectors(vectors, is_float=False)
    else:
        print("Unknown file format. Please use .fvecs or .ivecs files.")

if __name__ == "__main__":
    main() 