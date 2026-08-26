#include "common/cuda_check.hpp"
#include "ch10-16/reference_algorithms.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

template <int block_side>
__global__ void floyd_warshall_square(std::int64_t* distances, int vertices,
                                      int k, std::int64_t infinity) {
    __shared__ std::int64_t column_values[block_side];
    __shared__ std::int64_t row_values[block_side];
    const int row = blockIdx.y * block_side + threadIdx.y;
    const int column = blockIdx.x * block_side + threadIdx.x;
    if (threadIdx.x == 0) {
        column_values[threadIdx.y] =
            row < vertices
                ? distances[static_cast<std::size_t>(row) * vertices + k]
                : infinity;
    }
    if (threadIdx.y == 0) {
        row_values[threadIdx.x] =
            column < vertices
                ? distances[static_cast<std::size_t>(k) * vertices + column]
                : infinity;
    }
    __syncthreads();
    if (row < vertices && column < vertices && row != k && column != k &&
        column_values[threadIdx.y] != infinity &&
        row_values[threadIdx.x] != infinity) {
        const std::int64_t via =
            column_values[threadIdx.y] + row_values[threadIdx.x];
        std::int64_t& current =
            distances[static_cast<std::size_t>(row) * vertices + column];
        if (via < current) {
            current = via;
        }
    }
}

void validate_floyd_input(const std::vector<std::int64_t>& input,
                          int vertices, std::int64_t infinity) {
    if (vertices < 0 ||
        input.size() != static_cast<std::size_t>(vertices) * vertices) {
        throw std::invalid_argument("invalid Floyd-Warshall dimensions");
    }
    if (infinity <= 0 ||
        infinity > std::numeric_limits<std::int64_t>::max() / 2) {
        throw std::invalid_argument(
            "infinity must be positive and at most INT64_MAX/2");
    }
    for (const std::int64_t value : input) {
        if (value != infinity && (value < 0 || value >= infinity)) {
            throw std::invalid_argument(
                "finite distances must lie in [0, infinity)");
        }
    }
}

std::vector<std::int64_t> floyd_warshall_cuda(
    const std::vector<std::int64_t>& input, int vertices,
    std::int64_t infinity) {
    validate_floyd_input(input, vertices, infinity);
    if (vertices == 0) {
        return {};
    }
    constexpr int block_side = 16;
    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (block_side * block_side > properties.maxThreadsPerBlock ||
        block_side > properties.maxThreadsDim[0] ||
        block_side > properties.maxThreadsDim[1]) {
        throw std::runtime_error("device cannot launch a 16x16 block");
    }
    device_buffer<std::int64_t> device_distances(input.size());
    CUDA_CHECK(cudaMemcpy(device_distances.get(), input.data(),
                          input.size() * sizeof(std::int64_t),
                          cudaMemcpyHostToDevice));
    const dim3 block(block_side, block_side);
    const dim3 grid((vertices + block_side - 1) / block_side,
                    (vertices + block_side - 1) / block_side);
    for (int k = 0; k < vertices; ++k) {
        floyd_warshall_square<block_side><<<grid, block>>>(
            device_distances.get(), vertices, k, infinity);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::vector<std::int64_t> output(input.size());
    CUDA_CHECK(cudaMemcpy(output.data(), device_distances.get(),
                          output.size() * sizeof(std::int64_t),
                          cudaMemcpyDeviceToHost));
    return output;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        pmpp::reference::test_ch16();
        constexpr int vertices = 17;
        constexpr std::int64_t infinity = 1'000'000;
        std::vector<std::int64_t> input(
            static_cast<std::size_t>(vertices) * vertices, infinity);
        for (int vertex = 0; vertex < vertices; ++vertex) {
            input[static_cast<std::size_t>(vertex) * vertices + vertex] = 0;
        }
        for (int vertex = 0; vertex + 1 < vertices; ++vertex) {
            input[static_cast<std::size_t>(vertex) * vertices + vertex + 1] =
                1 + vertex % 5;
        }
        input[0 * vertices + 8] = 50;
        input[3 * vertices + 12] = 11;
        input[6 * vertices + 2] = 4;
        validate_floyd_input(input, vertices, infinity);
        bool rejected_negative_weight = false;
        try {
            validate_floyd_input({0, -1, infinity, 0}, 2, infinity);
        } catch (const std::invalid_argument&) {
            rejected_negative_weight = true;
        }
        pmpp::reference::require(
            rejected_negative_weight,
            "Floyd-Warshall range policy accepted a negative edge");
        const std::vector<std::int64_t> expected =
            pmpp::reference::floyd_warshall(input, vertices, infinity);
        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            pmpp::reference::verify_exact(
                pmpp::reference::floyd_warshall(input, vertices, infinity),
                expected, "chapter 16 CPU Floyd repeat");
            report_cpu_only("ch16_ex01_floyd_warshall");
            return EXIT_SUCCESS;
        }
        pmpp::reference::verify_exact(
            floyd_warshall_cuda(input, vertices, infinity), expected,
            "CUDA square Floyd-Warshall");
        std::cout << "ch16_ex01_floyd_warshall: CUDA table passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
