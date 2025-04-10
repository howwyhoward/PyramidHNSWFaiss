# Pyramid Search with FAISS - Setup and Usage Guide

This guide details how to set up and run the Pyramid Search with FAISS on macOS. Pyramid Search is a hierarchical, partitioned similarity search framework that enhances FAISS-based Approximate Nearest Neighbor (ANN) search.

## Prerequisites

- macOS (tested on macOS Sequoia/Sonoma)
- Homebrew package manager
- Miniconda or Anaconda installed

## Setup Instructions

### Step 1: Create a Conda Environment

First, create a dedicated conda environment for FAISS:

```bash
conda create -n faiss_env python=3.10 -y
conda activate faiss_env
```

### Step 2: Install FAISS

Install FAISS from the PyTorch channel:

```bash
conda install -c pytorch faiss-cpu -y
```

### Step 3: Install OpenMP

OpenMP is required for parallel processing capabilities:

```bash
brew install libomp
```

### Step 4: Set Environment Variables

Set the necessary environment variables for OpenMP:

```bash
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
```

### Step 5: Configure and Build with CMake

Remove any existing build directory and configure the project with CMake:

```bash
rm -rf build/
cmake -B build -G "Unix Makefiles" \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
    -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp" \
    -DOpenMP_C_LIB_NAMES="omp" \
    -DCMAKE_CXX_FLAGS="-I/opt/homebrew/opt/libomp/include" \
    -DCMAKE_C_FLAGS="-I/opt/homebrew/opt/libomp/include" \
    .
```

### Step 6: Compile the Project

Build the project using Make:

```bash
make -C build -j4  # Adjust the number based on your CPU cores
```

## Running the HNSW Pyramid Search

Execute the compiled binary:

```bash
./build/pyramid_search
```

## Expected Output

The program will:

1. Load base vectors and query vectors from the SIFT dataset
2. Build a Pyramid index with the specified number of partitions
3. Perform k-NN search for each query vector
4. Display search results and performance metrics

Sample output:

```
Loaded 10000 base vectors with dimension 128
Loaded 100 query vectors
Ground truth file contains k = 100 neighbors per query
Loaded ground truth for 100 queries

Building Pyramid index with 10 partitions...
Indexed 10000 vectors in 110 ms

Performing 100 queries...
Search completed in 9 ms (11111.11 queries per second)

Query Results (Top-100 neighbors for first 5 queries):
Query 0: (2176, 76608.00) (3752, 77004.00) (3615, 95662.00) (1884, 98017.00) (3013, 98568.00) ...
Query 1: (2781, 65973.00) (9574, 67555.00) (2492, 69114.00) (1322, 69130.00) (3136, 70189.00) ...
Query 2: (2707, 42740.00) (9938, 48163.00) (2698, 49529.00) (9972, 51859.00) (6995, 58445.00) ...
Query 3: (9843, 46301.00) (9825, 47021.00) (9574, 55761.00) (9582, 58664.00) (4097, 62311.00) ...
Query 4: (4719, 62755.00) (5164, 66326.00) (1671, 70261.00) (1538, 71437.00) (5897, 78055.00) ...

Recall@100 = 86.75%
```

## Performance Analysis

- **Index Building Time**: The time it takes to build the Pyramid index (typically ~100-200ms for the sample dataset)
- **Query Time**: The time to complete all queries (typically under 10ms for 100 queries)
- **Queries per Second**: The number of queries that can be processed each second (typically 10,000+ QPS)
- **Recall@k**: The accuracy of the search results compared to ground truth (~85-90% on the sample dataset)

## Customization

You can modify the following parameters in `src/main.cpp`:

- `k`: Number of nearest neighbors to find
- `num_clusters`: Number of partitions to create

## Troubleshooting

- If you encounter FAISS not found errors during cmake, verify the FAISS installation in your conda environment.
- For OpenMP issues, ensure libomp is properly installed and environment variables are set correctly.
- For dataset issues, confirm that the SIFT dataset files are in the correct location (`data/siftsmall/` directory). 