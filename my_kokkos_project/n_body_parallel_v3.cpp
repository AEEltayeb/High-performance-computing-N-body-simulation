#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
#include <random>
#include <chrono>

constexpr int N = 10000;
constexpr double dt = 0.01;
constexpr int steps = 100;
constexpr double G = 6.67430e-11;

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        std::cout << "Running on: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
        Kokkos::View<double*> x("x", N);
        Kokkos::View<double*> y("y", N);
        Kokkos::View<double*> z("z", N);
        Kokkos::View<double*> vx("vx", N);
        Kokkos::View<double*> vy("vy", N);
        Kokkos::View<double*> vz("vz", N);
        Kokkos::View<double*> mass("mass", N);

        // Host mirrors for initialization
        auto x_h = Kokkos::create_mirror_view(x);
        auto y_h = Kokkos::create_mirror_view(y);
        auto z_h = Kokkos::create_mirror_view(z);
        auto vx_h = Kokkos::create_mirror_view(vx);
        auto vy_h = Kokkos::create_mirror_view(vy);
        auto vz_h = Kokkos::create_mirror_view(vz);
        auto mass_h = Kokkos::create_mirror_view(mass);

        // Random number generation
        std::mt19937 rng(42); // Seed for reproducibility
        std::uniform_real_distribution<double> pos_dist(0.0, 100.0);
        std::uniform_real_distribution<double> mass_dist(1.0, 1e5);

        for (int i = 0; i < N; ++i) {
            x_h(i) = pos_dist(rng);
            y_h(i) = pos_dist(rng);
            z_h(i) = pos_dist(rng);
            vx_h(i) = vy_h(i) = vz_h(i) = 0.0;
            mass_h(i) = mass_dist(rng);
        }

        // Deep copy
        Kokkos::deep_copy(x, x_h);
        Kokkos::deep_copy(y, y_h);
        Kokkos::deep_copy(z, z_h);
        Kokkos::deep_copy(vx, vx_h);
        Kokkos::deep_copy(vy, vy_h);
        Kokkos::deep_copy(vz, vz_h);
        Kokkos::deep_copy(mass, mass_h);

        auto start = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < steps; ++step) {
            Kokkos::parallel_for("update", N, KOKKOS_LAMBDA(const int i) {
                double fx = 0.0, fy = 0.0, fz = 0.0;
                for (int j = 0; j < N; ++j) {
                    if (i == j) continue;
                    double dx = x(j) - x(i);
                    double dy = y(j) - y(i);
                    double dz = z(j) - z(i);
                    double dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9;
                    double inv_dist = 1.0 / sqrt(dist_sqr);
                    double inv_dist3 = inv_dist * inv_dist * inv_dist;
                    double f = G * mass(i) * mass(j) * inv_dist3;
                    fx += f * dx;
                    fy += f * dy;
                    fz += f * dz;
                }
                vx(i) += fx / mass(i) * dt;
                vy(i) += fy / mass(i) * dt;
                vz(i) += fz / mass(i) * dt;
            });

            Kokkos::parallel_for("move", N, KOKKOS_LAMBDA(const int i) {
                x(i) += vx(i) * dt;
                y(i) += vy(i) * dt;
                z(i) += vz(i) * dt;
            });
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "\nOptimizations Applied:\n";
        std::cout << "- SoA layout for better memory access.\n";
        std::cout << "- Flat parallel_for instead of nested teams.\n";
        std::cout << "- Reduced overhead by removing nested lambdas.\n";
        std::cout << "- Efficient memory access patterns.\n\n";
        std::cout << "number of steps: " << steps << ".\n";
        std::cout << "number of particles: " << N << ".\n";

        std::cout << "Simulation time: " << elapsed.count() << " seconds\n";

        // Copy back result to host and print
        Kokkos::deep_copy(x_h, x);
        Kokkos::deep_copy(y_h, y);
        Kokkos::deep_copy(z_h, z);
        for (int i = 0; i < 5; ++i) {
            std::cout << "Particle " << i << ": ("
                      << x_h(i) << ", " << y_h(i) << ", " << z_h(i) << ")\n";
        }
    }
    Kokkos::finalize();
    return 0;
}
