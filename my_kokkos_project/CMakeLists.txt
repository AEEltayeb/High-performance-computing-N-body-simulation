cmake_minimum_required(VERSION 3.16)
project(HPC_NBody_Project CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set path to your Kokkos installation
set(Kokkos_DIR "${CMAKE_SOURCE_DIR}/../kokkos/install/lib/cmake/Kokkos")

# Find necessary packages
find_package(Kokkos REQUIRED)

# --- Find OpenMP only if the Kokkos OpenMP backend is enabled ---
if(Kokkos_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# --- Serial Version ---
add_executable(serial_nbody n_body_serial.cpp)

# --- OpenMP Version ---
add_executable(parallel_nbody_openmp n_body_parallel_openmp.cpp)
target_link_libraries(parallel_nbody_openmp PRIVATE OpenMP::OpenMP_CXX)

# --- Kokkos Versions ---
add_executable(parallel_nbody_kokkos n_body_parallel_v3.cpp)

# --- FIX 1: Add compile option for icx compiler ---
if(MSVC OR CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    target_compile_options(parallel_nbody_kokkos PRIVATE "-DKOKKOS_IMPL_ALIGN_PTR(size)=")
endif()

# --- FIX 2: Link Kokkos AND its required dependencies ---
if(Kokkos_ENABLE_OPENMP)
    target_link_libraries(parallel_nbody_kokkos PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX)
else()
    target_link_libraries(parallel_nbody_kokkos PRIVATE Kokkos::kokkos)
endif()

# --- Optional: Add a check for the SYCL backend ---
if (Kokkos_ENABLE_SYCL)
    message(STATUS "Kokkos SYCL backend is enabled for GPU execution.")
endif()