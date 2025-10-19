#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

constexpr int N = 10000;
constexpr double dt = 0.01;
constexpr int steps = 100;
constexpr double G = 6.67430e-11;

int main() {
    std::vector<double> x(N), y(N), z(N);
    std::vector<double> vx(N, 0.0), vy(N, 0.0), vz(N, 0.0);
    std::vector<double> mass(N);

    std::mt19937 rng(42); 
    std::uniform_real_distribution<double> pos_dist(0.0, 100.0);
    std::uniform_real_distribution<double> mass_dist(1.0, 1e5);

    for (int i = 0; i < N; ++i) {
        x[i] = pos_dist(rng);
        y[i] = pos_dist(rng);
        z[i] = pos_dist(rng);
        mass[i] = mass_dist(rng);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < steps; ++step) {
        for (int i = 0; i < N; ++i) {
            double fx = 0.0, fy = 0.0, fz = 0.0;
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;

                double dx = x[j] - x[i];
                double dy = y[j] - y[i];
                double dz = z[j] - z[i];

                double dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9;
                double inv_dist = 1.0 / std::sqrt(dist_sqr);
                double inv_dist3 = inv_dist * inv_dist * inv_dist;

                double f = G * mass[i] * mass[j] * inv_dist3;
                fx += f * dx;
                fy += f * dy;
                fz += f * dz;
            }

            vx[i] += fx / mass[i] * dt;
            vy[i] += fy / mass[i] * dt;
            vz[i] += fz / mass[i] * dt;
        }

        for (int i = 0; i < N; ++i) {
            x[i] += vx[i] * dt;
            y[i] += vy[i] * dt;
            z[i] += vz[i] * dt;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "\nSerial Simulation Complete\n";
    std::cout << "Simulation time: " << elapsed << " seconds\n";
        std::cout << "number of steps: " << steps << ".\n";
        std::cout << "number of particles: " << N << ".\n";

    for (int i = 0; i < 5; ++i) {
        std::cout << "Particle " << i << ": ("
                  << x[i] << ", " << y[i] << ", " << z[i] << ")\n";
    }

    return 0;
}
