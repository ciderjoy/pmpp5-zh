#include "common/cuda_check.hpp"

#include <cuda_runtime.h>

#include <climits>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

__device__ __forceinline__ int keep(int value) { return value != 0; }

template <int block_size>
__device__ void block_exclusive_scan(int input, int& exclusive, int& total,
                                     int* scratch) {
    scratch[threadIdx.x] = input;
    __syncthreads();
    for (unsigned int stride = 1; stride < block_size; stride <<= 1) {
        const int addend =
            threadIdx.x >= stride ? scratch[threadIdx.x - stride] : 0;
        __syncthreads();
        if (threadIdx.x >= stride) {
            scratch[threadIdx.x] += addend;
        }
        __syncthreads();
    }
    exclusive = threadIdx.x == 0 ? 0 : scratch[threadIdx.x - 1];
    total = scratch[block_size - 1];
}

__global__ void evaluate_flags(const int* input, int* flags,
                               std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        flags[index] = keep(input[index]);
    }
}

__global__ void scatter_filtered(const int* input, const int* flags,
                                 const int* positions, int* output,
                                 std::size_t count, int* output_size) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        if (flags[index] != 0) {
            output[positions[index]] = input[index];
        }
        if (index == count - 1) {
            *output_size = positions[index] + flags[index];
        }
    }
}

template <int block_size>
__global__ void exclusive_scan_stage(const int* flags, int* positions,
                                     std::size_t count) {
    __shared__ int scratch[block_size];
    __shared__ int carry;
    if (threadIdx.x == 0) {
        carry = 0;
    }
    __syncthreads();
    for (std::size_t begin = 0; begin < count; begin += block_size) {
        const std::size_t index = begin + threadIdx.x;
        const int flag = index < count ? flags[index] : 0;
        int rank = 0;
        int total = 0;
        block_exclusive_scan<block_size>(flag, rank, total, scratch);
        const int chunk_base = carry;
        if (index < count) {
            positions[index] = chunk_base + rank;
        }
        __syncthreads();
        if (threadIdx.x == block_size - 1) {
            carry = chunk_base + total;
        }
        __syncthreads();
    }
}

template <int block_size>
void stable_filter_three_stage(const int* input, int* output,
                               std::size_t count, int* flags, int* positions,
                               int* output_size, unsigned int grid) {
    if (count == 0) {
        CUDA_CHECK(cudaMemset(output_size, 0, sizeof(*output_size)));
        return;
    }
    if (count > static_cast<std::size_t>(INT_MAX)) {
        throw std::length_error("filter positions exceed int range");
    }
    evaluate_flags<<<grid, block_size>>>(input, flags, count);
    CUDA_CHECK(cudaGetLastError());
    exclusive_scan_stage<block_size><<<1, block_size>>>(flags, positions,
                                                        count);
    CUDA_CHECK(cudaGetLastError());
    scatter_filtered<<<grid, block_size>>>(input, flags, positions, output,
                                           count, output_size);
    CUDA_CHECK(cudaGetLastError());
}

enum : unsigned int { empty_state = 0, aggregate_state = 1, prefix_state = 2 };

__device__ int publish_lookback(int total, unsigned int block_id,
                                volatile int* aggregates,
                                volatile int* prefixes,
                                unsigned int* states) {
    __shared__ int base;
    if (threadIdx.x == 0) {
        aggregates[block_id] = total;
        // Payload first, then publish its state. The fence plus atomic state
        // access is the legacy-CUDA equivalent of release/acquire publication.
        __threadfence();
        atomicExch(states + block_id, aggregate_state);
        int sum = 0;
        int predecessor = static_cast<int>(block_id) - 1;
        while (predecessor >= 0) {
            unsigned int kind = empty_state;
            do {
                kind = atomicAdd(states + predecessor, 0U);
            } while (kind == empty_state);
            if (kind == prefix_state) {
                sum += prefixes[predecessor];
                break;
            }
            sum += aggregates[predecessor];
            --predecessor;
        }
        base = sum;
        prefixes[block_id] = sum + total;
        __threadfence();
        atomicExch(states + block_id, prefix_state);
    }
    __syncthreads();
    return base;
}

template <int block_size>
__global__ void stable_filter_onepass(
    const int* input, int* output, std::size_t count, unsigned int* counter,
    int* aggregates, int* prefixes, unsigned int* states, int* output_size) {
    __shared__ int scan_storage[block_size];
    __shared__ unsigned int logical_block;
    if (threadIdx.x == 0) {
        logical_block = atomicAdd(counter, 1U);
    }
    __syncthreads();
    const unsigned int block_id = logical_block;
    const std::size_t index =
        static_cast<std::size_t>(block_id) * block_size + threadIdx.x;
    const int value = index < count ? input[index] : 0;
    const int flag = index < count && value != 0 ? 1 : 0;
    int rank = 0;
    int total = 0;
    block_exclusive_scan<block_size>(flag, rank, total, scan_storage);
    const int base =
        publish_lookback(total, block_id, aggregates, prefixes, states);
    if (flag != 0) {
        output[base + rank] = value;
    }
    const unsigned int segments = static_cast<unsigned int>(
        1 + (count - 1) / static_cast<std::size_t>(block_size));
    if (threadIdx.x == 0 && block_id == segments - 1) {
        *output_size = base + total;
    }
}

template <int block_size>
__global__ void stable_filter_private(
    const int* input, int* output, std::size_t count, unsigned int* counter,
    int* aggregates, int* prefixes, unsigned int* states, int* output_size) {
    __shared__ int scan_storage[block_size];
    __shared__ int packed[block_size];
    __shared__ int base_shared;
    __shared__ int total_shared;
    __shared__ unsigned int logical_block;
    if (threadIdx.x == 0) {
        logical_block = atomicAdd(counter, 1U);
    }
    __syncthreads();
    const unsigned int block_id = logical_block;
    const std::size_t index =
        static_cast<std::size_t>(block_id) * block_size + threadIdx.x;
    const int value = index < count ? input[index] : 0;
    const int flag = index < count && value != 0 ? 1 : 0;
    int rank = 0;
    int total = 0;
    block_exclusive_scan<block_size>(flag, rank, total, scan_storage);
    if (flag != 0) {
        packed[rank] = value;
    }
    __syncthreads();
    const int block_base =
        publish_lookback(total, block_id, aggregates, prefixes, states);
    if (threadIdx.x == 0) {
        base_shared = block_base;
        total_shared = total;
    }
    __syncthreads();
    for (int item = threadIdx.x; item < total_shared; item += block_size) {
        output[base_shared + item] = packed[item];
    }
    const unsigned int segments = static_cast<unsigned int>(
        1 + (count - 1) / static_cast<std::size_t>(block_size));
    if (threadIdx.x == 0 && block_id == segments - 1) {
        *output_size = base_shared + total_shared;
    }
}

template <int block_size, int coarse_factor>
__global__ void stable_filter_coarse(
    const int* input, int* output, std::size_t count, unsigned int* counter,
    int* aggregates, int* prefixes, unsigned int* states, int* output_size) {
    static_assert(coarse_factor > 0, "coarse_factor must be positive");
    __shared__ int scan_storage[block_size];
    __shared__ int packed[block_size * coarse_factor];
    __shared__ int base_shared;
    __shared__ int total_shared;
    __shared__ unsigned int logical_block;
    if (threadIdx.x == 0) {
        logical_block = atomicAdd(counter, 1U);
    }
    __syncthreads();
    const unsigned int block_id = logical_block;
    const std::size_t first =
        static_cast<std::size_t>(block_id) * block_size * coarse_factor +
        static_cast<std::size_t>(threadIdx.x) * coarse_factor;
    int values[coarse_factor];
    int local_count = 0;
#pragma unroll
    for (int item = 0; item < coarse_factor; ++item) {
        const std::size_t index = first + item;
        values[item] = index < count ? input[index] : 0;
        local_count += index < count && values[item] != 0 ? 1 : 0;
    }
    int thread_base = 0;
    int total = 0;
    block_exclusive_scan<block_size>(local_count, thread_base, total,
                                     scan_storage);
    int local_rank = 0;
#pragma unroll
    for (int item = 0; item < coarse_factor; ++item) {
        const std::size_t index = first + item;
        if (index < count && values[item] != 0) {
            packed[thread_base + local_rank] = values[item];
            ++local_rank;
        }
    }
    __syncthreads();
    const int block_base =
        publish_lookback(total, block_id, aggregates, prefixes, states);
    if (threadIdx.x == 0) {
        base_shared = block_base;
        total_shared = total;
    }
    __syncthreads();
    for (int item = threadIdx.x; item < total_shared; item += block_size) {
        output[base_shared + item] = packed[item];
    }
    const std::size_t segment_size =
        static_cast<std::size_t>(block_size) * coarse_factor;
    const unsigned int segments =
        static_cast<unsigned int>(1 + (count - 1) / segment_size);
    if (threadIdx.x == 0 && block_id == segments - 1) {
        *output_size = base_shared + total_shared;
    }
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<int> stable_filter_cpu(const std::vector<int>& input) {
    std::vector<int> output;
    output.reserve(input.size());
    for (const int value : input) {
        if (value != 0) {
            output.push_back(value);
        }
    }
    return output;
}

void verify(const std::vector<int>& actual, const std::vector<int>& expected,
            const char* label) {
    if (actual != expected) {
        const std::size_t common =
            actual.size() < expected.size() ? actual.size() : expected.size();
        std::size_t mismatch = 0;
        while (mismatch < common && actual[mismatch] == expected[mismatch]) {
            ++mismatch;
        }
        throw std::runtime_error(
            std::string(label) + ": mismatch at index " +
            std::to_string(mismatch) + " (actual size " +
            std::to_string(actual.size()) + ", expected size " +
            std::to_string(expected.size()) + ")");
    }
}

unsigned int checked_grid(std::size_t count, std::size_t segment_size,
                          const cudaDeviceProp& properties) {
    if (count == 0 || segment_size == 0) {
        throw std::invalid_argument("filter launch requires nonempty input");
    }
    const std::size_t blocks = 1 + (count - 1) / segment_size;
    if (blocks > static_cast<std::size_t>(properties.maxGridSize[0]) ||
        blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::length_error("filter grid.x exceeds device limits");
    }
    return static_cast<unsigned int>(blocks);
}

std::vector<int> copy_filter_result(const device_buffer<int>& output,
                                    const device_buffer<int>& output_size,
                                    std::size_t capacity,
                                    const char* label) {
    int size = -1;
    CUDA_CHECK(cudaMemcpy(&size, output_size.get(), sizeof(size),
                          cudaMemcpyDeviceToHost));
    if (size < 0 || static_cast<std::size_t>(size) > capacity) {
        throw std::runtime_error(std::string(label) +
                                 ": invalid output size");
    }
    std::vector<int> result(static_cast<std::size_t>(size));
    if (!result.empty()) {
        CUDA_CHECK(cudaMemcpy(result.data(), output.get(),
                              result.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }
    return result;
}

std::vector<int> run_three_stage(const std::vector<int>& input,
                                 const cudaDeviceProp& properties) {
    constexpr unsigned int block_size = 256;
    if (input.size() > static_cast<std::size_t>(INT_MAX)) {
        throw std::length_error("filter positions exceed int range");
    }
    if (block_size >
        static_cast<unsigned int>(properties.maxThreadsPerBlock)) {
        throw std::runtime_error("three-stage block exceeds device limit");
    }
    device_buffer<int> d_input(input.size());
    device_buffer<int> d_output(input.size());
    device_buffer<int> d_flags(input.size());
    device_buffer<int> d_positions(input.size());
    device_buffer<int> d_output_size(1);
    if (!input.empty()) {
        CUDA_CHECK(cudaMemcpy(d_input.get(), input.data(),
                              input.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    if (input.empty()) {
        stable_filter_three_stage<block_size>(
            nullptr, nullptr, 0, nullptr, nullptr, d_output_size.get(), 0);
    } else {
        const unsigned int grid =
            checked_grid(input.size(), block_size, properties);
        stable_filter_three_stage<block_size>(
            d_input.get(), d_output.get(), input.size(), d_flags.get(),
            d_positions.get(), d_output_size.get(), grid);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return copy_filter_result(d_output, d_output_size, input.size(),
                              "three-stage filter");
}

enum class one_pass_kind { direct, privatized, coarse };

std::vector<int> run_one_pass(const std::vector<int>& input,
                              const cudaDeviceProp& properties,
                              one_pass_kind kind) {
    if (input.empty()) {
        return {};
    }
    if (input.size() > static_cast<std::size_t>(INT_MAX)) {
        throw std::length_error("filter positions exceed int range");
    }
    constexpr int block_size = 128;
    constexpr int coarse_factor = 4;
    if (block_size > properties.maxThreadsPerBlock) {
        throw std::runtime_error("one-pass block exceeds device limit");
    }
    const std::size_t segment_size =
        kind == one_pass_kind::coarse
            ? static_cast<std::size_t>(block_size) * coarse_factor
            : static_cast<std::size_t>(block_size);
    const unsigned int grid =
        checked_grid(input.size(), segment_size, properties);

    device_buffer<int> d_input(input.size());
    device_buffer<int> d_output(input.size());
    device_buffer<int> d_output_size(1);
    device_buffer<unsigned int> d_counter(1);
    device_buffer<int> d_aggregates(grid);
    device_buffer<int> d_prefixes(grid);
    device_buffer<unsigned int> d_states(grid);
    CUDA_CHECK(cudaMemcpy(d_input.get(), input.data(),
                          input.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_counter.get(), 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_states.get(), 0,
                          static_cast<std::size_t>(grid) *
                              sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_output_size.get(), 0xFF, sizeof(int)));

    if (kind == one_pass_kind::direct) {
        stable_filter_onepass<block_size><<<grid, block_size>>>(
            d_input.get(), d_output.get(), input.size(), d_counter.get(),
            d_aggregates.get(), d_prefixes.get(), d_states.get(),
            d_output_size.get());
    } else if (kind == one_pass_kind::privatized) {
        stable_filter_private<block_size><<<grid, block_size>>>(
            d_input.get(), d_output.get(), input.size(), d_counter.get(),
            d_aggregates.get(), d_prefixes.get(), d_states.get(),
            d_output_size.get());
    } else {
        stable_filter_coarse<block_size, coarse_factor>
            <<<grid, block_size>>>(
                d_input.get(), d_output.get(), input.size(), d_counter.get(),
                d_aggregates.get(), d_prefixes.get(), d_states.get(),
                d_output_size.get());
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned int> states(grid);
    CUDA_CHECK(cudaMemcpy(states.data(), d_states.get(),
                          states.size() * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    for (const unsigned int state : states) {
        require(state == prefix_state,
                "one-pass block did not publish a prefix");
    }
    return copy_filter_result(d_output, d_output_size, input.size(),
                              "one-pass filter");
}

int main(int argc, char** argv) {
    try {
        const std::vector<int> canonical{0, 4, 0, -3, 8, 0, 0, 5};
        const std::vector<int> canonical_expected{4, -3, 8, 5};
        require(stable_filter_cpu(canonical) == canonical_expected,
                "CPU filter fixture did not preserve stable order");
        require(stable_filter_cpu({}).empty(),
                "empty CPU filter must remain empty");

        std::vector<int> input(2051);
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] = index % 5 == 0
                               ? 0
                               : static_cast<int>((index * 13) % 101) - 50;
        }
        input[1] = -17;
        input[input.size() - 1] = 29;
        const std::vector<int> expected = stable_filter_cpu(input);
        require(!expected.empty() && expected.front() == -17 &&
                    expected.back() == 29,
                "CPU filter boundary fixture failed");

        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            verify(stable_filter_cpu(input), expected, "CPU filter repeat");
            report_cpu_only("ch12_ex01-04_filter");
            return EXIT_SUCCESS;
        }

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

        verify(run_three_stage({}, properties), {},
               "empty three-stage filter");
        verify(run_three_stage(input, properties), expected,
               "three-stage filter");
        verify(run_one_pass(input, properties, one_pass_kind::direct),
               expected, "single-pass filter");
        verify(run_one_pass(input, properties, one_pass_kind::privatized),
               expected, "privatized single-pass filter");
        verify(run_one_pass(input, properties, one_pass_kind::coarse),
               expected, "coarsened privatized filter");

        std::cout << "ch12_ex01-04_filter: all variants agree with CPU.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
