#include "common/cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* c,
                           std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

std::vector<float> vector_add_cpu(const std::vector<float>& a,
                                  const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("vector sizes differ");
    }
    std::vector<float> result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
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
        constexpr std::size_t n = 1003;
        std::vector<float> a(n), b(n);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i) * 0.25f;
            b[i] = 2.0f - static_cast<float>(i) * 0.125f;
        }
        const std::vector<float> expected = vector_add_cpu(a, b);

        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            verify(vector_add_cpu(a, b), expected);
            report_cpu_only("ch02_ex02_vector_add");
            return EXIT_SUCCESS;
        }

        device_buffer<float> d_a(n), d_b(n), d_c(n);
        const std::size_t bytes = n * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), bytes,
                              cudaMemcpyHostToDevice));

        constexpr unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>((n + threads - 1) /
                                                       threads);
        vector_add<<<blocks, threads>>>(d_a.get(), d_b.get(), d_c.get(), n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> actual(n);
        CUDA_CHECK(cudaMemcpy(actual.data(), d_c.get(), bytes,
                              cudaMemcpyDeviceToHost));
        verify(actual, expected);
        std::cout << "ch02_ex02_vector_add: GPU and CPU results agree.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
