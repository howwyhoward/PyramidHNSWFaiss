cmake_minimum_required(VERSION 3.10)
project(PyramidSearch)

set(CMAKE_CXX_STANDARD 17)

# Find conda environment
execute_process(
    COMMAND bash -c "conda info --base"
    OUTPUT_VARIABLE CONDA_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Conda prefix: ${CONDA_PREFIX}")

# Find FAISS in conda environment
set(FAISS_INCLUDE_DIR "${CONDA_PREFIX}/include")
set(FAISS_LIBRARY_DIR "${CONDA_PREFIX}/lib")

# Check if FAISS exists
if(NOT EXISTS "${FAISS_INCLUDE_DIR}/faiss")
    message(FATAL_ERROR "FAISS headers not found in ${FAISS_INCLUDE_DIR}")
endif()

# Find BLAS
find_package(BLAS REQUIRED)

# Find OpenMP (optional)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "OpenMP not found - building without OpenMP support")
endif()

# Set include paths
include_directories(${FAISS_INCLUDE_DIR})
include_directories(src include)
link_directories(${FAISS_LIBRARY_DIR})

# Find all source files
file(GLOB SOURCES
    src/*.cpp
)

# Create library for our implementation
add_library(pyramid_lib STATIC ${SOURCES})
target_link_libraries(pyramid_lib faiss)

# Create executable
add_executable(pyramid_search src/main.cpp)
target_link_libraries(pyramid_search pyramid_lib faiss ${BLAS_LIBRARIES})

# Add tests if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()

# Enable testing
enable_testing()

