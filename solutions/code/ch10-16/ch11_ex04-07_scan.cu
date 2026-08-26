#include "common/cuda_check.hpp"

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

template <int block_size>
__device__ float block_inclusive(float value, float* scratch) {
    const unsigned int thread = threadIdx.x;
    scratch[thread] = value;
    __syncthreads();
    for (unsigned int stride = 1; stride < block_size; stride <<= 1) {
        const float addend = thread >= stride ? scratch[thread - stride]
                                               : 0.0f;
        __syncthreads();
        if (thread >= stride) {
            scratch[thread] += addend;
        }
        __syncthreads();
    }
    return scratch[thread];
}

template <int block_size, int coarse_factor>
__global__ void scan_vec_exchange(const float* input, float* output,
                                  std::size_t count) {
    static_assert(block_size > 0 &&
                      (block_size & (block_size - 1)) == 0,
                  "block_size must be a power of two");
    static_assert(coarse_factor > 0 && coarse_factor % 4 == 0,
                  "coarse_factor must be a positive multiple of four");
    constexpr int packed_items = coarse_factor / 4;
    using PackedLoad =
        cub::BlockLoad<float4, block_size, packed_items,
                       cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using PackedStore =
        cub::BlockStore<float4, block_size, packed_items,
                        cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using ScalarLoad =
        cub::BlockLoad<float, block_size, coarse_factor,
                       cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using ScalarStore =
        cub::BlockStore<float, block_size, coarse_factor,
                        cub::BLOCK_STORE_WARP_TRANSPOSE>;
    union ExchangeStorage {
        typename PackedLoad::TempStorage packed_load;
        typename PackedStore::TempStorage packed_store;
        typename ScalarLoad::TempStorage scalar_load;
        typename ScalarStore::TempStorage scalar_store;
    };
    __shared__ ExchangeStorage exchange;
    __shared__ float totals[block_size];
    const std::size_t segment =
        static_cast<std::size_t>(blockIdx.x) * block_size * coarse_factor;
    float local[coarse_factor];
    const std::size_t segment_capacity = block_size * coarse_factor;
    const int valid_items = segment < count
                                ? static_cast<int>(
                                      min(segment_capacity, count - segment))
                                : 0;
    const bool vector_path =
        valid_items == static_cast<int>(segment_capacity) &&
        (reinterpret_cast<std::uintptr_t>(input + segment) & 15U) == 0U &&
        (reinterpret_cast<std::uintptr_t>(output + segment) & 15U) == 0U;
    if (vector_path) {
        float4 packed[packed_items];
        PackedLoad(exchange.packed_load)
            .Load(reinterpret_cast<const float4*>(input + segment), packed);
#pragma unroll
        for (int item = 0; item < packed_items; ++item) {
            local[4 * item] = packed[item].x;
            local[4 * item + 1] = packed[item].y;
            local[4 * item + 2] = packed[item].z;
            local[4 * item + 3] = packed[item].w;
        }
    } else {
        ScalarLoad(exchange.scalar_load)
            .Load(input + segment, local, valid_items, 0.0f);
    }
    __syncthreads();

#pragma unroll
    for (int item = 1; item < coarse_factor; ++item) {
        local[item] += local[item - 1];
    }
    block_inclusive<block_size>(local[coarse_factor - 1], totals);
    const float before =
        threadIdx.x == 0 ? 0.0f : totals[threadIdx.x - 1];
#pragma unroll
    for (int item = 0; item < coarse_factor; ++item) {
        local[item] += before;
    }
    __syncthreads();

    if (vector_path) {
        float4 packed[packed_items];
#pragma unroll
        for (int item = 0; item < packed_items; ++item) {
            packed[item] = make_float4(
                local[4 * item], local[4 * item + 1], local[4 * item + 2],
                local[4 * item + 3]);
        }
        PackedStore(exchange.packed_store)
            .Store(reinterpret_cast<float4*>(output + segment), packed);
    } else {
        ScalarStore(exchange.scalar_store)
            .Store(output + segment, local, valid_items);
    }
}

enum : unsigned int { empty_state = 0, aggregate_state = 1, prefix_state = 2 };

__device__ float decoupled_lookback(float block_sum, unsigned int block_id,
                                    volatile float* aggregates,
                                    volatile float* prefixes,
                                    unsigned int* states) {
    __shared__ float exclusive;
    if (threadIdx.x == blockDim.x - 1) {
        aggregates[block_id] = block_sum;
        // Payload first, then publish its state. The fence plus atomic state
        // access is the legacy-CUDA equivalent of release/acquire publication.
        __threadfence();
        atomicExch(states + block_id, aggregate_state);

        float before = 0.0f;
        int predecessor = static_cast<int>(block_id) - 1;
        while (predecessor >= 0) {
            unsigned int kind = empty_state;
            do {
                kind = atomicAdd(states + predecessor, 0U);
            } while (kind == empty_state);
            if (kind == prefix_state) {
                before += prefixes[predecessor];
                break;
            }
            before += aggregates[predecessor];
            --predecessor;
        }
        exclusive = before;
        prefixes[block_id] = before + block_sum;
        __threadfence();
        atomicExch(states + block_id, prefix_state);
    }
    __syncthreads();
    return exclusive;
}

template <int block_size>
__global__ void decoupled_scan(const float* input, float* output,
                               std::size_t count, unsigned int* counter,
                               float* aggregates, float* prefixes,
                               unsigned int* states) {
    static_assert(block_size > 0 &&
                      (block_size & (block_size - 1)) == 0,
                  "block_size must be a power of two");
    __shared__ float scan_values[block_size];
    __shared__ unsigned int logical_block;
    if (threadIdx.x == 0) {
        logical_block = atomicAdd(counter, 1U);
    }
    __syncthreads();
    const std::size_t index =
        static_cast<std::size_t>(logical_block) * block_size + threadIdx.x;
    const float value = index < count ? input[index] : 0.0f;
    const float local_inclusive =
        block_inclusive<block_size>(value, scan_values);
    const float base = decoupled_lookback(
        scan_values[block_size - 1], logical_block, aggregates, prefixes,
        states);
    if (index < count) {
        output[index] = base + local_inclusive;
    }
}

template <int block_size>
__global__ void brent_kung_scan(const float* input, float* output,
                                std::size_t count) {
    static_assert(block_size > 0 &&
                      (block_size & (block_size - 1)) == 0,
                  "block_size must be a power of two");
    __shared__ float values[block_size];
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * block_size + threadIdx.x;
    values[threadIdx.x] = index < count ? input[index] : 0.0f;
    __syncthreads();

    for (unsigned int stride = 1; stride < block_size; stride <<= 1) {
        const unsigned int target =
            (threadIdx.x + 1) * (stride << 1) - 1;
        if (target < block_size) {
            values[target] += values[target - stride];
        }
        __syncthreads();
    }
    for (unsigned int stride = block_size >> 2; stride > 0; stride >>= 1) {
        const unsigned int target =
            (threadIdx.x + 1) * (stride << 1) + stride - 1;
        if (target < block_size) {
            values[target] += values[target - stride];
        }
        __syncthreads();
    }
    if (index < count) {
        output[index] = values[threadIdx.x];
    }
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<float> inclusive_scan_cpu(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    float prefix = 0.0f;
    for (std::size_t index = 0; index < input.size(); ++index) {
        prefix += input[index];
        output[index] = prefix;
    }
    return output;
}

std::vector<float> segmented_scan_cpu(const std::vector<float>& input,
                                      std::size_t segment_size) {
    if (segment_size == 0) {
        throw std::invalid_argument("segment size must be positive");
    }
    std::vector<float> output(input.size());
    for (std::size_t begin = 0; begin < input.size(); begin += segment_size) {
        float prefix = 0.0f;
        const std::size_t end =
            input.size() - begin < segment_size ? input.size()
                                                : begin + segment_size;
        for (std::size_t index = begin; index < end; ++index) {
            prefix += input[index];
            output[index] = prefix;
        }
    }
    return output;
}

void verify(const std::vector<float>& actual,
            const std::vector<float>& expected, const char* label) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(std::string(label) + ": size mismatch");
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float tolerance =
            2.0e-5f + 2.0e-5f * std::abs(expected[index]);
        if (!std::isfinite(actual[index]) ||
            std::abs(actual[index] - expected[index]) > tolerance) {
            throw std::runtime_error(std::string(label) +
                                     ": mismatch at index " +
                                     std::to_string(index));
        }
    }
}

unsigned int checked_grid(std::size_t count, std::size_t segment_size,
                          const cudaDeviceProp& properties) {
    if (count == 0 || segment_size == 0) {
        throw std::invalid_argument("scan launch requires nonempty input");
    }
    const std::size_t blocks = 1 + (count - 1) / segment_size;
    if (blocks > static_cast<std::size_t>(properties.maxGridSize[0]) ||
        blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::length_error("scan grid.x exceeds device limits");
    }
    return static_cast<unsigned int>(blocks);
}

int main(int argc, char** argv) {
    try {
        const std::vector<float> canonical{4.0f, 6.0f, 7.0f, 1.0f,
                                           2.0f, 8.0f, 5.0f, 2.0f};
        const std::vector<float> canonical_expected{
            4.0f, 10.0f, 17.0f, 18.0f, 20.0f, 28.0f, 33.0f, 35.0f};
        require(inclusive_scan_cpu(canonical) == canonical_expected,
                "CPU scan fixture did not produce the textbook result");
        require(inclusive_scan_cpu({}).empty(),
                "empty CPU scan must remain empty");

        std::vector<float> input(997);
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] = static_cast<float>((index * 7 + 3) % 11);
        }
        const std::vector<float> global_expected = inclusive_scan_cpu(input);

        constexpr int vector_block = 64;
        constexpr int coarse_factor = 12;
        const std::vector<float> vector_expected = segmented_scan_cpu(
            input, static_cast<std::size_t>(vector_block) * coarse_factor);
        constexpr int brent_block = 128;
        const std::vector<float> brent_expected =
            segmented_scan_cpu(input, brent_block);

        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            verify(inclusive_scan_cpu(input), global_expected,
                   "CPU global scan repeat");
            verify(segmented_scan_cpu(input, vector_block * coarse_factor),
                   vector_expected, "CPU vector scan repeat");
            verify(segmented_scan_cpu(input, brent_block), brent_expected,
                   "CPU Brent-Kung repeat");
            report_cpu_only("ch11_ex04-07_scan");
            return EXIT_SUCCESS;
        }

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        if (vector_block > properties.maxThreadsPerBlock ||
            brent_block > properties.maxThreadsPerBlock) {
            throw std::runtime_error("scan block exceeds device limit");
        }

        device_buffer<float> d_input(input.size());
        device_buffer<float> d_output(input.size());
        CUDA_CHECK(cudaMemcpy(d_input.get(), input.data(),
                              input.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        std::vector<float> actual(input.size());

        const unsigned int vector_grid = checked_grid(
            input.size(), vector_block * coarse_factor, properties);
        scan_vec_exchange<vector_block, coarse_factor>
            <<<vector_grid, vector_block>>>(d_input.get(), d_output.get(),
                                            input.size());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), d_output.get(),
                              actual.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        verify(actual, vector_expected, "vector exchange block scan");

        const unsigned int global_grid =
            checked_grid(input.size(), brent_block, properties);
        device_buffer<unsigned int> d_counter(1);
        device_buffer<float> d_aggregates(global_grid);
        device_buffer<float> d_prefixes(global_grid);
        device_buffer<unsigned int> d_states(global_grid);
        CUDA_CHECK(cudaMemset(d_counter.get(), 0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_states.get(), 0,
                              global_grid * sizeof(unsigned int)));
        decoupled_scan<brent_block><<<global_grid, brent_block>>>(
            d_input.get(), d_output.get(), input.size(), d_counter.get(),
            d_aggregates.get(), d_prefixes.get(), d_states.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), d_output.get(),
                              actual.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        verify(actual, global_expected, "decoupled-lookback scan");
        std::vector<unsigned int> states(global_grid);
        CUDA_CHECK(cudaMemcpy(states.data(), d_states.get(),
                              states.size() * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        for (const unsigned int state : states) {
            require(state == prefix_state,
                    "decoupled-lookback block did not publish a prefix");
        }

        const unsigned int brent_grid =
            checked_grid(input.size(), brent_block, properties);
        brent_kung_scan<brent_block><<<brent_grid, brent_block>>>(
            d_input.get(), d_output.get(), input.size());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), d_output.get(),
                              actual.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        verify(actual, brent_expected, "Brent-Kung block scan");

        std::cout << "ch11_ex04-07_scan: all kernels agree with CPU.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
