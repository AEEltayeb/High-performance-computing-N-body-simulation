# HPC Project – N-Body Simulation (Serial, OpenMP, Kokkos, GPU)

This project implements an N-Body simulation in multiple versions:
- **Serial**
- **OpenMP**
- **Kokkos (CPU)**
- **GPU (SYCL with Kokkos, runs under WSL)**

Each version has its own build and execution steps.  
The **GPU version is in its own file** (`n_body_gpu.cpp`) and must be copied to WSL before compilation and execution.  

---

## Project Setup
Navigate to the project directory:
cd <project_root>/my_kokkos_project

---

## Serial / OpenMP Versions (Windows + oneAPI)
1. Remove any old build:
rmdir /s /q build

2. Recreate build directory:
mkdir build && cd build

3. Compile:
cmake .. -G "Ninja" -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release
ninja

4. Set number of OpenMP threads:
set OMP_NUM_THREADS=12

5. Run executables:
.\parallel_nbody_openmp.exe
.\serial_nbody.exe

---

## Kokkos CPU Version (Windows + oneAPI)
1. Remove old build:
cd <project_root>/my_kokkos_project
rmdir /s /q build

2. Recreate build and navigate:
mkdir build && cd build

3. Compile with Kokkos support:
cmake .. -G "Ninja" ^
-DCMAKE_CXX_COMPILER=icx ^
-DCMAKE_PREFIX_PATH="<path_to_kokkos_install>" ^
-DKokkos_ENABLE_OPENMP=ON ^
-DCMAKE_BUILD_TYPE=Release

ninja parallel_nbody_kokkos

4. Run:
set OMP_NUM_THREADS=12
.\parallel_nbody_kokkos.exe

---

## GPU Version (WSL + SYCL + Kokkos)
Note: The GPU code is located in `n_body_gpu.cpp`.  
It must be copied from the Windows project folder into your WSL project directory before compiling.

1. Open WSL:
wsl -d Ubuntu-22.04

2. Navigate to project directory inside WSL.

3. Source Intel oneAPI environment:
source /opt/intel/oneapi/setvars.sh

4. Compile GPU version:
icpx -fsycl -std=c++17 -O3 -g \
    -I<path_to_kokkos_install>/include \
    <project_root>/src/n_body_gpu.cpp \
    <path_to_kokkos_install>/lib/libkokkoscore.a \
    <path_to_kokkos_install>/lib/libkokkoscontainers.a \
    <path_to_kokkos_install>/lib/libkokkosalgorithms.a \
    <path_to_kokkos_install>/lib/libkokkossimd.a \
    -o <project_root>/n_body_gpu

5. Run GPU executable:
<project_root>/n_body_gpu

---

License
This project is licensed under the MIT License.
See LICENSE for details.

Author
A.E. Eltayeb
GitHub: @AEEltayeb

## Copying Code Between Windows and WSL
To copy the GPU source file from Windows to WSL:
cp /mnt/c/<Windows_project_root>/src/n_body_gpu.cpp ~/HPC_GPU_Project/src/

Check the copied files:
ls -l ~/HPC_GPU_Project/src/
