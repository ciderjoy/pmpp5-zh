#include "common/cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int block_width = 8;

__global__ void block_transpose(float* matrix, int width, int height) {
    __shared__ float tile[block_width][block_width + 1];
    const int x = blockIdx.x * block_width + threadIdx.x;
    const int y = blockIdx.y * block_width + threadIdx.y;
    const bool input_valid = x < width && y < height;
    tile[threadIdx.y][threadIdx.x] =
        input_valid ? matrix[y * width + x] : 0.0f;
    __syncthreads();

    const int source_x = blockIdx.x * block_width + threadIdx.y;
    const int source_y = blockIdx.y * block_width + threadIdx.x;
    if (input_valid && source_x < width && source_y < height) {
        matrix[y * width + x] = tile[threadIdx.x][threadIdx.y];
    }
}

std::vector<float> block_transpose_cpu(const std::vector<float>& input,
                                       int width, int height) {
    if (width <= 0 || height <= 0 ||
        input.size() != static_cast<std::size_t>(width) * height) {
        throw std::invalid_argument("invalid matrix dimensions");
    }
    std::vector<float> output = input;
    for (int base_y = 0; base_y < height; base_y += block_width) {
        for (int base_x = 0; base_x < width; base_x += block_width) {
            const int tile_height = std::min(block_width, height - base_y);
            const int tile_width = std::min(block_width, width - base_x);
            const int transposed_side = std::min(tile_width, tile_height);
            for (int y = 0; y < transposed_side; ++y) {
                for (int x = 0; x < transposed_side; ++x) {
                    output[(base_y + y) * width + base_x + x] =
                        input[(base_y + x) * width + base_x + y];
                }
            }
        }
    }
    return output;
}

void verify(const std::vector<float>& actual,
            const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("result size mismatch");
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > 1.0e-6f) {
            throw std::runtime_error("mismatch at index " +
                                     std::to_string(i));
        }
    }
}

int main(int argc, char** argv) {
    try {
        constexpr int width = 22;
        constexpr int height = 14;
        const std::size_t elements = static_cast<std::size_t>(width) * height;
        std::vector<float> input(elements);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                input[y * width + x] = static_cast<float>(100 * y + x);
            }
        }
        const std::vector<float> expected =
            block_transpose_cpu(input, width, height);
        if (expected[1] != 100.0f || expected[width] != 1.0f) {
            throw std::runtime_error("CPU transpose fixture failed");
        }

        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            verify(block_transpose_cpu(input, width, height), expected);
            report_cpu_only("ch05_ex10_block_transpose");
            return EXIT_SUCCESS;
        }

        device_buffer<float> d_matrix(elements);
        const std::size_t bytes = elements * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_matrix.get(), input.data(), bytes,
                              cudaMemcpyHostToDevice));
        const dim3 block(block_width, block_width);
        const dim3 grid((width + block_width - 1) / block_width,
                        (height + block_width - 1) / block_width);
        block_transpose<<<grid, block>>>(d_matrix.get(), width, height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> actual(elements);
        CUDA_CHECK(cudaMemcpy(actual.data(), d_matrix.get(), bytes,
                              cudaMemcpyDeviceToHost));
        verify(actual, expected);
        std::cout << "ch05_ex10_block_transpose: GPU and CPU agree.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
