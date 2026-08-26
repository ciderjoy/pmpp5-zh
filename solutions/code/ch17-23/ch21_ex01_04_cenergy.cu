#include "common/cuda_check.hpp"
#include "ch17_23_cpu_algorithms.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int atom_chunk_size = 4;
constexpr int coarsening_factor = 4;
__constant__ float atom_chunk[atom_chunk_size * 4];

__device__ float atom_energy(float x, float y, float z, int atom_index) {
    const float atom_x = atom_chunk[atom_index * 4];
    const float atom_y = atom_chunk[atom_index * 4 + 1];
    const float atom_z = atom_chunk[atom_index * 4 + 2];
    const float charge = atom_chunk[atom_index * 4 + 3];
    const float dx = x - atom_x;
    const float dy = y - atom_y;
    const float dz = z - atom_z;
    return charge * rsqrtf(dx * dx + dy * dy + dz * dz);
}

__global__ void cenergy(float* energy, int size_x, int size_y, int size_z,
                        float spacing, int slice, int atom_count) {
    const int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_index >= size_x || y_index >= size_y || slice < 0 ||
        slice >= size_z || atom_count < 0 || atom_count > atom_chunk_size) {
        return;
    }
    const float x = static_cast<float>(x_index) * spacing;
    const float y = static_cast<float>(y_index) * spacing;
    const float z = static_cast<float>(slice) * spacing;
    float sum = 0.0f;
    for (int atom = 0; atom < atom_count; ++atom) {
        sum += atom_energy(x, y, z, atom);
    }
    const std::size_t index =
        (static_cast<std::size_t>(slice) * size_y + y_index) * size_x +
        x_index;
    energy[index] += sum;
}

__global__ void cenergy_vectorized(float* energy, int size_x, int size_y,
                                   int size_z, float spacing, int slice,
                                   int atom_count) {
    const int linear_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int first_x = linear_x * coarsening_factor;
    const int y_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (first_x >= size_x || y_index >= size_y || slice < 0 ||
        slice >= size_z || atom_count < 0 || atom_count > atom_chunk_size) {
        return;
    }
    float contributions[coarsening_factor] = {};
    const float y = static_cast<float>(y_index) * spacing;
    const float z = static_cast<float>(slice) * spacing;
    for (int atom = 0; atom < atom_count; ++atom) {
#pragma unroll
        for (int offset = 0; offset < coarsening_factor; ++offset) {
            if (first_x + offset < size_x) {
                const float x =
                    static_cast<float>(first_x + offset) * spacing;
                contributions[offset] += atom_energy(x, y, z, atom);
            }
        }
    }
    const std::size_t row_base =
        (static_cast<std::size_t>(slice) * size_y + y_index) * size_x;
    const std::size_t first = row_base + first_x;
    if (first_x + 3 < size_x && (first & 3U) == 0) {
        float4* pointer = reinterpret_cast<float4*>(energy + first);
        float4 previous = *pointer;
        previous.x += contributions[0];
        previous.y += contributions[1];
        previous.z += contributions[2];
        previous.w += contributions[3];
        *pointer = previous;
    } else {
#pragma unroll
        for (int offset = 0; offset < coarsening_factor; ++offset) {
            if (first_x + offset < size_x) {
                energy[first + offset] += contributions[offset];
            }
        }
    }
}

enum class KernelMode { scalar, vectorized };

std::vector<float> run_cenergy_gpu(int size_x, int size_y, int size_z,
                                   float spacing,
                                   const std::vector<pmpp_examples::Atom>& atoms,
                                   KernelMode mode) {
    const std::vector<float> validated =
        pmpp_examples::cenergy_cpu(size_x, size_y, size_z, spacing, atoms);
    std::vector<float> packed(atoms.size() * 4);
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        packed[i * 4] = atoms[i].x;
        packed[i * 4 + 1] = atoms[i].y;
        packed[i * 4 + 2] = atoms[i].z;
        packed[i * 4 + 3] = atoms[i].charge;
    }
    device_buffer<float> device_energy(validated.size());
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaMemsetAsync(device_energy.get(), 0,
                               validated.size() * sizeof(float), stream));
    const dim3 threads(32, 8, 1);
    const unsigned logical_x = mode == KernelMode::scalar
                                   ? static_cast<unsigned>(size_x)
                                   : static_cast<unsigned>(
                                         (size_x + coarsening_factor - 1) /
                                         coarsening_factor);
    const dim3 blocks((logical_x + threads.x - 1) / threads.x,
                      (static_cast<unsigned>(size_y) + threads.y - 1) /
                          threads.y,
                      1);
    constexpr std::size_t dynamic_shared_memory = 0;
    for (int slice = 0; slice < size_z; ++slice) {
        for (std::size_t first_atom = 0; first_atom < atoms.size();
             first_atom += atom_chunk_size) {
            const int count = static_cast<int>(std::min<std::size_t>(
                atom_chunk_size, atoms.size() - first_atom));
            CUDA_CHECK(cudaMemcpyToSymbolAsync(
                atom_chunk, packed.data() + first_atom * 4,
                static_cast<std::size_t>(count) * 4 * sizeof(float), 0,
                cudaMemcpyHostToDevice, stream));
            if (mode == KernelMode::scalar) {
                cenergy<<<blocks, threads, dynamic_shared_memory, stream>>>(
                    device_energy.get(), size_x, size_y, size_z, spacing, slice,
                    count);
            } else {
                cenergy_vectorized<<<blocks, threads, dynamic_shared_memory,
                                     stream>>>(device_energy.get(), size_x,
                                              size_y, size_z, spacing, slice,
                                              count);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> result(validated.size());
    CUDA_CHECK(cudaMemcpy(result.data(), device_energy.get(),
                          result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return result;
}

void run_cpu_assertions(int size_x, int size_y, int size_z, float spacing,
                        const std::vector<pmpp_examples::Atom>& atoms) {
    const std::vector<float> energy =
        pmpp_examples::cenergy_cpu(size_x, size_y, size_z, spacing, atoms);
    pmpp_examples::expect(
        energy.size() == static_cast<std::size_t>(size_x) * size_y * size_z,
        "CPU cenergy output shape");
    pmpp_examples::expect(
        std::all_of(energy.begin(), energy.end(),
                    [](float value) { return std::isfinite(value); }),
        "CPU cenergy finite output");
    const std::vector<float> no_atoms =
        pmpp_examples::cenergy_cpu(size_x, size_y, size_z, spacing, {});
    pmpp_examples::expect(
        std::all_of(no_atoms.begin(), no_atoms.end(),
                    [](float value) { return value == 0.0f; }),
        "CPU cenergy empty atom list");
    const std::vector<float> singular = pmpp_examples::cenergy_cpu(
        1, 1, 1, spacing, {{0.0f, 0.0f, 0.0f, 1.0f}});
    pmpp_examples::expect(std::isinf(singular[0]) && singular[0] > 0.0f,
                          "CPU cenergy preserves the q/r singularity");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool cpu_only = pmpp_examples::validate_cli(argc, argv);
        constexpr int size_x = 11;
        constexpr int size_y = 5;
        constexpr int size_z = 3;
        constexpr float spacing = 0.5f;
        const std::vector<pmpp_examples::Atom> atoms{
            {0.25f, 0.50f, 0.75f, 1.00f},
            {2.25f, 1.50f, 0.25f, -0.40f},
            {4.50f, 3.25f, 1.50f, 0.70f},
            {1.10f, 0.90f, 2.20f, 0.35f},
            {3.40f, 2.70f, 1.10f, -0.20f}};
        run_cpu_assertions(size_x, size_y, size_z, spacing, atoms);
        const std::vector<float> reference = pmpp_examples::cenergy_cpu(
            size_x, size_y, size_z, spacing, atoms);
        if (cpu_only || !has_cuda_device()) {
            report_cpu_only("ch21_ex01_04_cenergy");
            return EXIT_SUCCESS;
        }
        const std::vector<float> scalar = run_cenergy_gpu(
            size_x, size_y, size_z, spacing, atoms, KernelMode::scalar);
        const std::vector<float> vectorized = run_cenergy_gpu(
            size_x, size_y, size_z, spacing, atoms, KernelMode::vectorized);
        pmpp_examples::expect_near(scalar, reference, 2.0e-5f, 4.0e-5f,
                                   "GPU scalar cenergy");
        pmpp_examples::expect_near(vectorized, reference, 2.0e-5f, 4.0e-5f,
                                   "GPU vector cenergy");
        pmpp_examples::expect_near(vectorized, scalar, 2.0e-5f, 4.0e-5f,
                                   "GPU cenergy implementations");
        std::cout << "ch21_ex01_04_cenergy: GPU comparisons passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ch21_ex01_04_cenergy: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
