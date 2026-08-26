#include "common/cuda_check.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int tile_size = 32;

__global__ void matmul_corner(const float* left, const float* right_column,
                              float* output, int m, int inner, int n) {
    __shared__ float left_tile[tile_size][tile_size];
    __shared__ float right_tile[tile_size][tile_size + 1];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * tile_size + ty;
    const int col = blockIdx.x * tile_size + tx;
    float sum = 0.0f;

    for (int phase = 0; phase < (inner + tile_size - 1) / tile_size;
         ++phase) {
        const int left_col = phase * tile_size + tx;
        left_tile[ty][tx] = row < m && left_col < inner
                                ? left[row * inner + left_col]
                                : 0.0f;

        const int right_row = phase * tile_size + tx;
        const int loaded_col = blockIdx.x * tile_size + ty;
        right_tile[tx][ty] = right_row < inner && loaded_col < n
                                 ? right_column[right_row + loaded_col * inner]
                                 : 0.0f;
        __syncthreads();

        for (int k = 0; k < tile_size; ++k) {
            sum += left_tile[ty][k] * right_tile[k][tx];
        }
        __syncthreads();
    }
    if (row < m && col < n) {
        output[row * n + col] = sum;
    }
}

std::vector<float> matmul_cpu(const std::vector<float>& left,
                              const std::vector<float>& right_column, int m,
                              int inner, int n) {
    if (m <= 0 || inner <= 0 || n <= 0 ||
        left.size() != static_cast<std::size_t>(m) * inner ||
        right_column.size() != static_cast<std::size_t>(inner) * n) {
        throw std::invalid_argument("invalid matrix dimensions");
    }
    std::vector<float> output(static_cast<std::size_t>(m) * n, 0.0f);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double sum = 0.0;
            for (int k = 0; k < inner; ++k) {
                sum += static_cast<double>(left[row * inner + k]) *
                       right_column[k + col * inner];
            }
            output[row * n + col] = static_cast<float>(sum);
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
        const float tolerance = 2.0e-4f + 2.0e-4f * std::abs(expected[i]);
        if (std::abs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error("mismatch at index " +
                                     std::to_string(i));
        }
    }
}

int main(int argc, char** argv) {
    try {
        constexpr int m = 37;
        constexpr int inner = 45;
        constexpr int n = 35;
        std::vector<float> left(static_cast<std::size_t>(m) * inner);
        std::vector<float> right_column(
            static_cast<std::size_t>(inner) * n);
        for (int row = 0; row < m; ++row) {
            for (int k = 0; k < inner; ++k) {
                left[row * inner + k] =
                    static_cast<float>((row * 5 + k * 3) % 19 - 9) / 16.0f;
            }
        }
        for (int col = 0; col < n; ++col) {
            for (int k = 0; k < inner; ++k) {
                right_column[k + col * inner] =
                    static_cast<float>((k * 7 + col * 2) % 17 - 8) / 16.0f;
            }
        }
        const std::vector<float> expected =
            matmul_cpu(left, right_column, m, inner, n);
        if (!std::isfinite(expected.front()) ||
            !std::isfinite(expected.back())) {
            throw std::runtime_error("CPU reference produced non-finite data");
        }

        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            verify(matmul_cpu(left, right_column, m, inner, n), expected);
            report_cpu_only("ch06_ex02_corner_matmul");
            return EXIT_SUCCESS;
        }

        device_buffer<float> d_left(left.size()),
            d_right(right_column.size()), d_output(expected.size());
        CUDA_CHECK(cudaMemcpy(d_left.get(), left.data(),
                              left.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_right.get(), right_column.data(),
                              right_column.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        const dim3 block(tile_size, tile_size);
        const dim3 grid((n + tile_size - 1) / tile_size,
                        (m + tile_size - 1) / tile_size);
        matmul_corner<<<grid, block>>>(d_left.get(), d_right.get(),
                                      d_output.get(), m, inner, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> actual(expected.size());
        CUDA_CHECK(cudaMemcpy(actual.data(), d_output.get(),
                              actual.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        verify(actual, expected);
        std::cout << "ch06_ex02_corner_matmul: GPU and CPU agree.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
