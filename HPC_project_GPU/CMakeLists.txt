cmake_minimum_required(VERSION 3.16)
project(NBody_GPU_Project CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to SYCL-enabled Kokkos
set(Kokkos_DIR "C:/Users/ahmad/Desktop/HPC_GPU_Project/kokkos/install_sycl/lib/cmake/Kokkos")
find_package(Kokkos REQUIRED)

# Print backend info
if(Kokkos_ENABLE_SYCL)
    message(STATUS "SYCL backend enabled. GPU version will run on Intel GPU.")
endif()

# GPU version only
add_executable(n_body_gpu src/n_body_gpu.cpp)
target_link_libraries(n_body_gpu PRIVATE Kokkos::kokkos)
