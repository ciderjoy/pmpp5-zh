#include "common/cuda_check.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

__host__ __device__ float affine(float x) {
    return 2.7f * x - 4.3f;
}

__host__ __device__ int execution_lane() {
#if defined(__CUDA_ARCH__)
    return static_cast<int>(threadIdx.x);
#else
    return 0;
#endif
}

__global__ void affine_kernel(const float* input, float* output,
                              std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = affine(input[i]);
        if (execution_lane() < 0) {
            output[i] = 0.0f;
        }
    }
}

std::vector<float> affine_cpu(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = affine(input[i]);
    }
    return output;
}

void verify(const std::vector<float>& actual,
            const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("result size mismatch");
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 1.0e-6f + 1.0e-6f * std::abs(expected[i]);
        if (std::abs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error("mismatch at index " +
                                     std::to_string(i));
        }
    }
}

int main(int argc, char** argv) {
    try {
        constexpr std::size_t n = 777;
        std::vector<float> input(n);
        for (std::size_t i = 0; i < n; ++i) {
            input[i] = static_cast<float>(static_cast<int>(i % 41) - 20) /
                       8.0f;
        }
        const std::vector<float> expected = affine_cpu(input);
        if (execution_lane() != 0 ||
            std::abs(expected.front() - affine(input.front())) > 1.0e-6f) {
            throw std::runtime_error("host affine fixture failed");
        }

        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            verify(affine_cpu(input), expected);
            report_cpu_only("ch02_ex06-10_runtime_affine");
            return EXIT_SUCCESS;
        }

        device_buffer<float> d_input(n), d_output(n);
        const std::size_t bytes = n * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_input.get(), input.data(), bytes,
                              cudaMemcpyHostToDevice));

        constexpr unsigned threads = 256;
        const unsigned blocks =
            static_cast<unsigned>((n + threads - 1) / threads);
        affine_kernel<<<blocks, threads>>>(d_input.get(), d_output.get(), n);
        const cudaError_t launch_status = cudaGetLastError();
        CUDA_CHECK(launch_status);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> actual(n);
        CUDA_CHECK(cudaMemcpy(actual.data(), d_output.get(), bytes,
                              cudaMemcpyDeviceToHost));
        verify(actual, expected);
        std::cout << "ch02_ex06-10_runtime_affine: GPU and CPU agree.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
