#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        std::cout << "Hello from main thread!" << std::endl;

        Kokkos::parallel_for("HelloWorld", 4, KOKKOS_LAMBDA(int i) {
            printf("Hello from Kokkos thread %d\n", i);
        });
    }
    Kokkos::finalize();
    return 0;
}
