#include "common/cuda_check.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

__global__ void add_one_per_thread(const float* a, const float* b, float* c,
                                   std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_adjacent_pair(const float* a, const float* b, float* c,
                                  std::size_t n) {
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t first = 2 * thread;
    if (first < n) {
        c[first] = a[first] + b[first];
    }
    if (first + 1 < n) {
        c[first + 1] = a[first + 1] + b[first + 1];
    }
}

__global__ void add_segmented_pair(const float* a, const float* b, float* c,
                                   std::size_t n) {
    const std::size_t first =
        2 * static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (first < n) {
        c[first] = a[first] + b[first];
    }
    const std::size_t second = first + blockDim.x;
    if (second < n) {
        c[second] = a[second] + b[second];
    }
}

std::vector<float> add_cpu(const std::vector<float>& a,
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
            const std::vector<float>& expected, const char* label) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(std::string(label) + ": size mismatch");
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > 1.0e-6f) {
            throw std::runtime_error(std::string(label) +
                                     ": mismatch at index " +
                                     std::to_string(i));
        }
    }
}

int main(int argc, char** argv) {
    try {
        constexpr std::size_t n = 1003;
        std::vector<float> a(n), b(n);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i % 37) * 0.25f;
            b[i] = 3.0f - static_cast<float>(i % 29) * 0.125f;
        }
        const std::vector<float> expected = add_cpu(a, b);
        if (expected.front() != 3.0f ||
            std::abs(expected[1] - 3.125f) > 1.0e-6f) {
            throw std::runtime_error("CPU fixture validation failed");
        }

        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            verify(add_cpu(a, b), expected, "CPU");
            report_cpu_only("ch02_ex01-03_indexing");
            return EXIT_SUCCESS;
        }

        device_buffer<float> d_a(n), d_b(n), d_c(n);
        const std::size_t bytes = n * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), bytes,
                              cudaMemcpyHostToDevice));

        constexpr unsigned threads = 128;
        std::vector<float> actual(n);

        const unsigned one_blocks =
            static_cast<unsigned>((n + threads - 1) / threads);
        add_one_per_thread<<<one_blocks, threads>>>(d_a.get(), d_b.get(),
                                                    d_c.get(), n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), d_c.get(), bytes,
                              cudaMemcpyDeviceToHost));
        verify(actual, expected, "one-per-thread");

        CUDA_CHECK(cudaMemset(d_c.get(), 0, bytes));
        const std::size_t pair_tasks = (n + 1) / 2;
        const unsigned pair_blocks =
            static_cast<unsigned>((pair_tasks + threads - 1) / threads);
        add_adjacent_pair<<<pair_blocks, threads>>>(d_a.get(), d_b.get(),
                                                   d_c.get(), n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), d_c.get(), bytes,
                              cudaMemcpyDeviceToHost));
        verify(actual, expected, "adjacent-pair");

        CUDA_CHECK(cudaMemset(d_c.get(), 0, bytes));
        const unsigned segmented_blocks = static_cast<unsigned>(
            (n + 2 * threads - 1) / (2 * threads));
        add_segmented_pair<<<segmented_blocks, threads>>>(
            d_a.get(), d_b.get(), d_c.get(), n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), d_c.get(), bytes,
                              cudaMemcpyDeviceToHost));
        verify(actual, expected, "segmented-pair");

        std::cout << "ch02_ex01-03_indexing: all GPU mappings agree with CPU.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
