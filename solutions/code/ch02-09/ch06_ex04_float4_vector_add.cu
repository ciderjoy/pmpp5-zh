#include "common/cuda_check.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

__global__ void vector_add_float4(const float* x, const float* y, float* z,
                                  int n) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int vector_count = n / 4;
    if (thread < vector_count) {
        const float4 a = reinterpret_cast<const float4*>(x)[thread];
        const float4 b = reinterpret_cast<const float4*>(y)[thread];
        reinterpret_cast<float4*>(z)[thread] =
            make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    } else if (thread == vector_count) {
        for (int i = 4 * vector_count; i < n; ++i) {
            z[i] = x[i] + y[i];
        }
    }
}

std::vector<float> vector_add_cpu(const std::vector<float>& x,
                                  const std::vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("vector sizes differ");
    }
    std::vector<float> z(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        z[i] = x[i] + y[i];
    }
    return z;
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
        constexpr int n = 1027;
        std::vector<float> x(n), y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = static_cast<float>(i % 31) * 0.5f;
            y[i] = 4.0f - static_cast<float>(i % 23) * 0.25f;
        }
        const std::vector<float> expected = vector_add_cpu(x, y);
        if (expected.size() % 4 != 3 || expected.front() != 4.0f) {
            throw std::runtime_error("CPU tail fixture failed");
        }

        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            verify(vector_add_cpu(x, y), expected);
            report_cpu_only("ch06_ex04_float4_vector_add");
            return EXIT_SUCCESS;
        }

        device_buffer<float> d_x(n), d_y(n), d_z(n);
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_x.get(), x.data(), bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y.get(), y.data(), bytes,
                              cudaMemcpyHostToDevice));

        const int work = n / 4 + (n % 4 != 0);
        constexpr int threads = 256;
        const int blocks = (work + threads - 1) / threads;
        vector_add_float4<<<blocks, threads>>>(d_x.get(), d_y.get(), d_z.get(),
                                              n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> actual(n);
        CUDA_CHECK(cudaMemcpy(actual.data(), d_z.get(), bytes,
                              cudaMemcpyDeviceToHost));
        verify(actual, expected);
        std::cout << "ch06_ex04_float4_vector_add: GPU and CPU agree.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
