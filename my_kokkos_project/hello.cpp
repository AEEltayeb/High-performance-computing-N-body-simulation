#include <Kokkos_Core.hpp>
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        std::cout << "Kokkos initialized with " << Kokkos::DefaultExecutionSpace().name() << "\n";

        const int N = 1'000'000;

        // Allocate an array of N elements
        Kokkos::View<double*> data("data", N);

        // Start timer
        auto start = std::chrono::high_resolution_clock::now();

        // Parallel loop: fill array with values
        Kokkos::parallel_for("FillArray", N, KOKKOS_LAMBDA(int i) {
            data(i) = i * 0.5;
        });

        // Fence to ensure computation completes
        Kokkos::fence();

        // End timer
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Filled array of size " << N << " in " << elapsed.count() << " seconds.\n";

        // Check a few elements
        std::cout << "Sample values: data[0] = " << data(0) 
                  << ", data[N/2] = " << data(N / 2)
                  << ", data[N-1] = " << data(N - 1) << "\n";
    }
    Kokkos::finalize();
    return 0;
}
