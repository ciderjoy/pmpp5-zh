#include "common/cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

__global__ void matmul_row(const float* left, const float* right,
                           float* output, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }
    for (int col = 0; col < n; ++col) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += left[row * n + k] * right[k * n + col];
        }
        output[row * n + col] = sum;
    }
}

__global__ void matmul_col(const float* left, const float* right,
                           float* output, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) {
        return;
    }
    for (int row = 0; row < n; ++row) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += left[row * n + k] * right[k * n + col];
        }
        output[row * n + col] = sum;
    }
}

std::vector<float> matmul_cpu(const std::vector<float>& left,
                              const std::vector<float>& right, int n) {
    const std::size_t elements = static_cast<std::size_t>(n) * n;
    if (n <= 0 || left.size() != elements || right.size() != elements) {
        throw std::invalid_argument("invalid matrix dimensions");
    }
    std::vector<float> output(elements, 0.0f);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += left[row * n + k] * right[k * n + col];
            }
            output[row * n + col] = sum;
        }
    }
    return output;
}

void verify(const std::vector<float>& actual,
            const std::vector<float>& expected, const char* label) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(std::string(label) + ": size mismatch");
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 1.0e-5f + 1.0e-5f * std::abs(expected[i]);
        if (std::abs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error(std::string(label) +
                                     ": mismatch at index " +
                                     std::to_string(i));
        }
    }
}

int main(int argc, char** argv) {
    try {
        constexpr int n = 19;
        const std::size_t elements = static_cast<std::size_t>(n) * n;
        std::vector<float> left(elements), right(elements), right_transposed(elements);
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                left[row * n + col] =
                    static_cast<float>((row * 7 + col * 3) % 23 - 11) /
                    8.0f;
                right[row * n + col] =
                    static_cast<float>((row * 5 + col * 13 + row * col * 3 + 7) %
                                           29 -
                                       14) /
                    9.0f;
            }
        }
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                right_transposed[col * n + row] = right[row * n + col];
            }
        }
        const std::vector<float> expected = matmul_cpu(left, right, n);
        const std::vector<float> transposed_result =
            matmul_cpu(left, right_transposed, n);
        const bool distinguishes_transpose =
            !std::equal(expected.begin(), expected.end(),
                        transposed_result.begin(), [](float a, float b) {
                            return std::abs(a - b) <= 1.0e-4f;
                        });
        if (!distinguishes_transpose) {
            throw std::runtime_error(
                "matrix fixture does not distinguish a transposed right operand");
        }

        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            report_cpu_only("ch03_ex01_matmul");
            return EXIT_SUCCESS;
        }

        device_buffer<float> d_left(elements), d_right(elements),
            d_output(elements);
        const std::size_t bytes = elements * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_left.get(), left.data(), bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_right.get(), right.data(), bytes,
                              cudaMemcpyHostToDevice));

        constexpr unsigned threads = 64;
        const unsigned blocks =
            static_cast<unsigned>((n + threads - 1) / threads);
        std::vector<float> actual(elements);

        matmul_row<<<blocks, threads>>>(d_left.get(), d_right.get(),
                                       d_output.get(), n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), d_output.get(), bytes,
                              cudaMemcpyDeviceToHost));
        verify(actual, expected, "row kernel");

        CUDA_CHECK(cudaMemset(d_output.get(), 0, bytes));
        matmul_col<<<blocks, threads>>>(d_left.get(), d_right.get(),
                                       d_output.get(), n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), d_output.get(), bytes,
                              cudaMemcpyDeviceToHost));
        verify(actual, expected, "column kernel");

        std::cout << "ch03_ex01_matmul: row and column kernels agree with CPU.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
