#include "common/cuda_check.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

__global__ void matvec(const float* matrix, const float* vector,
                       float* output, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }
    float sum = 0.0f;
    for (int col = 0; col < n; ++col) {
        sum += matrix[row * n + col] * vector[col];
    }
    output[row] = sum;
}

std::vector<float> matvec_cpu(const std::vector<float>& matrix,
                              const std::vector<float>& vector, int n) {
    if (n <= 0 || vector.size() != static_cast<std::size_t>(n) ||
        matrix.size() != static_cast<std::size_t>(n) * n) {
        throw std::invalid_argument("invalid matrix-vector dimensions");
    }
    std::vector<float> output(n, 0.0f);
    for (int row = 0; row < n; ++row) {
        double sum = 0.0;
        for (int col = 0; col < n; ++col) {
            sum += static_cast<double>(matrix[row * n + col]) * vector[col];
        }
        output[row] = static_cast<float>(sum);
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
            throw std::runtime_error("mismatch at row " +
                                     std::to_string(i));
        }
    }
}

int main(int argc, char** argv) {
    try {
        constexpr int n = 513;
        const std::size_t matrix_elements = static_cast<std::size_t>(n) * n;
        std::vector<float> matrix(matrix_elements), vector(n);
        for (int col = 0; col < n; ++col) {
            vector[col] = static_cast<float>((col % 17) - 8) / 16.0f;
        }
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                matrix[row * n + col] =
                    row == col ? 1.5f
                               : static_cast<float>((row + col) % 7 - 3) /
                                     1024.0f;
            }
        }
        const std::vector<float> expected = matvec_cpu(matrix, vector, n);
        if (!std::isfinite(expected.front()) ||
            !std::isfinite(expected.back())) {
            throw std::runtime_error("CPU reference produced non-finite data");
        }

        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            verify(matvec_cpu(matrix, vector, n), expected);
            report_cpu_only("ch03_ex02_matvec");
            return EXIT_SUCCESS;
        }

        device_buffer<float> d_matrix(matrix_elements), d_vector(n),
            d_output(n);
        CUDA_CHECK(cudaMemcpy(d_matrix.get(), matrix.data(),
                              matrix_elements * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vector.get(), vector.data(),
                              static_cast<std::size_t>(n) * sizeof(float),
                              cudaMemcpyHostToDevice));

        constexpr unsigned threads = 256;
        const unsigned blocks =
            static_cast<unsigned>((n + threads - 1) / threads);
        matvec<<<blocks, threads>>>(d_matrix.get(), d_vector.get(),
                                   d_output.get(), n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> actual(n);
        CUDA_CHECK(cudaMemcpy(actual.data(), d_output.get(),
                              static_cast<std::size_t>(n) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        verify(actual, expected);
        std::cout << "ch03_ex02_matvec: GPU and CPU agree.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
