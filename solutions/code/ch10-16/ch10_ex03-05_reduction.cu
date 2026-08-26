#include "common/cuda_check.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

__global__ void right_owned_sum(float* input, float* output) {
    const unsigned int block_size = blockDim.x;
    const unsigned int thread = threadIdx.x;
    const unsigned int owner = block_size + thread;
    for (unsigned int stride = block_size; stride >= 1; stride >>= 1) {
        if (thread >= block_size - stride) {
            input[owner] += input[owner - stride];
        }
        __syncthreads();
    }
    if (thread == block_size - 1) {
        *output = input[2 * block_size - 1];
    }
}

template <int block_size, int coarse_factor>
__global__ void coarsened_max(const float* input, float* partial,
                              std::size_t count) {
    static_assert(block_size > 0 &&
                      (block_size & (block_size - 1)) == 0,
                  "block_size must be a power of two");
    static_assert(coarse_factor > 0, "coarse_factor must be positive");
    __shared__ float values[block_size];
    const std::size_t segment =
        static_cast<std::size_t>(blockIdx.x) *
        (2 * coarse_factor * block_size);
    const std::size_t owner = segment + threadIdx.x;
    float value = -CUDART_INF_F;
#pragma unroll
    for (int coarse = 0; coarse < 2 * coarse_factor; ++coarse) {
        const std::size_t index =
            owner + static_cast<std::size_t>(coarse) * block_size;
        if (index < count) {
            value = fmaxf(value, input[index]);
        }
    }
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            values[threadIdx.x] =
                fmaxf(values[threadIdx.x], values[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = values[0];
    }
}

template <int block_size, int coarse_factor>
__global__ void coarsened_sum_any(const float* input, float* partial,
                                  std::size_t count) {
    static_assert(block_size > 0 &&
                      (block_size & (block_size - 1)) == 0,
                  "block_size must be a power of two");
    static_assert(coarse_factor > 0, "coarse_factor must be positive");
    __shared__ float values[block_size];
    const std::size_t segment =
        static_cast<std::size_t>(blockIdx.x) *
        (2 * coarse_factor * block_size);
    const std::size_t owner = segment + threadIdx.x;
    float sum = 0.0f;
#pragma unroll
    for (int coarse = 0; coarse < 2 * coarse_factor; ++coarse) {
        const std::size_t index =
            owner + static_cast<std::size_t>(coarse) * block_size;
        if (index < count) {
            sum += input[index];
        }
    }
    values[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            values[threadIdx.x] += values[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = values[0];
    }
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

float sum_cpu(const std::vector<float>& input) {
    return std::accumulate(input.begin(), input.end(), 0.0f);
}

float max_cpu(const std::vector<float>& input) {
    if (input.empty()) {
        throw std::invalid_argument("maximum of an empty input is undefined");
    }
    return *std::max_element(input.begin(), input.end());
}

void verify_close(float actual, float expected, const char* label) {
    const float tolerance = 2.0e-5f + 2.0e-5f * std::abs(expected);
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(std::string(label) + ": expected " +
                                 std::to_string(expected) + ", got " +
                                 std::to_string(actual));
    }
}

std::size_t checked_grid_size(std::size_t count, std::size_t segment_size,
                              const cudaDeviceProp& properties) {
    if (count == 0 || segment_size == 0) {
        throw std::invalid_argument("a reduction launch requires data");
    }
    const std::size_t blocks = 1 + (count - 1) / segment_size;
    if (blocks > static_cast<std::size_t>(properties.maxGridSize[0]) ||
        blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::length_error("reduction grid.x exceeds device limits");
    }
    return blocks;
}

template <int block_size, int coarse_factor>
float reduce_sum_gpu(const std::vector<float>& input,
                     const cudaDeviceProp& properties) {
    if (input.empty()) {
        return 0.0f;
    }
    if (block_size > properties.maxThreadsPerBlock) {
        throw std::runtime_error("reduction block exceeds device limit");
    }
    device_buffer<float> first(input.size());
    device_buffer<float> second(input.size());
    CUDA_CHECK(cudaMemcpy(first.get(), input.data(),
                          input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    float* current = first.get();
    float* next = second.get();
    std::size_t remaining = input.size();
    constexpr std::size_t segment_size =
        2ULL * coarse_factor * block_size;
    while (remaining > 1) {
        const std::size_t blocks =
            checked_grid_size(remaining, segment_size, properties);
        coarsened_sum_any<block_size, coarse_factor>
            <<<static_cast<unsigned int>(blocks), block_size>>>(
                current, next, remaining);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        remaining = blocks;
        std::swap(current, next);
    }
    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, current, sizeof(result),
                          cudaMemcpyDeviceToHost));
    return result;
}

template <int block_size, int coarse_factor>
float reduce_max_gpu(const std::vector<float>& input,
                     const cudaDeviceProp& properties) {
    if (input.empty()) {
        throw std::invalid_argument("maximum of an empty input is undefined");
    }
    if (block_size > properties.maxThreadsPerBlock) {
        throw std::runtime_error("reduction block exceeds device limit");
    }
    device_buffer<float> first(input.size());
    device_buffer<float> second(input.size());
    CUDA_CHECK(cudaMemcpy(first.get(), input.data(),
                          input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    float* current = first.get();
    float* next = second.get();
    std::size_t remaining = input.size();
    constexpr std::size_t segment_size =
        2ULL * coarse_factor * block_size;
    while (remaining > 1) {
        const std::size_t blocks =
            checked_grid_size(remaining, segment_size, properties);
        coarsened_max<block_size, coarse_factor>
            <<<static_cast<unsigned int>(blocks), block_size>>>(
                current, next, remaining);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        remaining = blocks;
        std::swap(current, next);
    }
    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, current, sizeof(result),
                          cudaMemcpyDeviceToHost));
    return result;
}

int main(int argc, char** argv) {
    try {
        const std::vector<float> cpu_fixture{-7.0f, 2.0f, -3.0f, 5.0f};
        require(sum_cpu(cpu_fixture) == -3.0f,
                "CPU sum fixture did not execute correctly");
        require(max_cpu(cpu_fixture) == 5.0f,
                "CPU max fixture did not execute correctly");
        require(sum_cpu({}) == 0.0f,
                "empty CPU sum must equal the additive identity");

        constexpr std::size_t count = 1031;
        std::vector<float> input(count);
        for (std::size_t index = 0; index < count; ++index) {
            input[index] = static_cast<float>(
                static_cast<int>((index * 17 + 5) % 41) - 20);
        }
        input[777] = 123.0f;
        const float expected_sum = sum_cpu(input);
        const float expected_max = max_cpu(input);
        require(expected_max == 123.0f,
                "CPU reduction fixture lost its unique maximum");

        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            require(sum_cpu(input) == expected_sum,
                    "CPU sum repeat disagrees with fixture");
            require(max_cpu(input) == expected_max,
                    "CPU max repeat disagrees with fixture");
            report_cpu_only("ch10_ex03-05_reduction");
            return EXIT_SUCCESS;
        }

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

        constexpr int right_block = 64;
        if (right_block > properties.maxThreadsPerBlock) {
            throw std::runtime_error("right-owned block exceeds device limit");
        }
        std::vector<float> right_input(2 * right_block);
        for (std::size_t index = 0; index < right_input.size(); ++index) {
            right_input[index] = static_cast<float>(index % 9) - 4.0f;
        }
        const float expected_right = sum_cpu(right_input);
        device_buffer<float> d_right_input(right_input.size());
        device_buffer<float> d_right_output(1);
        CUDA_CHECK(cudaMemcpy(d_right_input.get(), right_input.data(),
                              right_input.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        right_owned_sum<<<1, right_block>>>(d_right_input.get(),
                                            d_right_output.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float actual_right = 0.0f;
        CUDA_CHECK(cudaMemcpy(&actual_right, d_right_output.get(),
                              sizeof(actual_right), cudaMemcpyDeviceToHost));
        verify_close(actual_right, expected_right, "right-owned reduction");

        constexpr int reduction_block = 64;
        constexpr int coarse_factor = 2;
        verify_close(reduce_sum_gpu<reduction_block, coarse_factor>(
                         input, properties),
                     expected_sum, "arbitrary-length sum reduction");
        verify_close(reduce_max_gpu<reduction_block, coarse_factor>(
                         input, properties),
                     expected_max, "coarsened max reduction");

        std::cout << "ch10_ex03-05_reduction: all kernels agree with CPU.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
