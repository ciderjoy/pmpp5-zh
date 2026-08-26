#include "common/cuda_check.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

__host__ __device__ int co_rank(int rank, const int* first, int first_count,
                                const int* second, int second_count) {
    int first_rank = rank < first_count ? rank : first_count;
    int second_rank = rank - first_rank;
    int first_low = rank - second_count > 0 ? rank - second_count : 0;
    int second_low = rank - first_count > 0 ? rank - first_count : 0;
    while (true) {
        if (first_rank > 0 && second_rank < second_count &&
            first[first_rank - 1] > second[second_rank]) {
            const int delta = (first_rank - first_low + 1) / 2;
            second_low = second_rank;
            second_rank += delta;
            first_rank -= delta;
        } else if (second_rank > 0 && first_rank < first_count &&
                   second[second_rank - 1] >= first[first_rank]) {
            const int delta = (second_rank - second_low + 1) / 2;
            first_low = first_rank;
            first_rank += delta;
            second_rank -= delta;
        } else {
            return first_rank;
        }
    }
}

__device__ void merge_sequential(const int* first, int first_count,
                                 const int* second, int second_count,
                                 int* output) {
    int first_index = 0;
    int second_index = 0;
    int output_index = 0;
    while (first_index < first_count && second_index < second_count) {
        if (first[first_index] <= second[second_index]) {
            output[output_index++] = first[first_index++];
        } else {
            output[output_index++] = second[second_index++];
        }
    }
    while (first_index < first_count) {
        output[output_index++] = first[first_index++];
    }
    while (second_index < second_count) {
        output[output_index++] = second[second_index++];
    }
}

__global__ void co_rank_all(const int* first, int first_count,
                            const int* second, int second_count,
                            int* first_ranks) {
    const int rank =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int output_count = first_count + second_count;
    if (rank <= output_count) {
        first_ranks[rank] =
            co_rank(rank, first, first_count, second, second_count);
    }
}

template <int tile_size>
__global__ void tiled_merge_exact_loads(const int* first, int first_count,
                                        const int* second, int second_count,
                                        int* output) {
    static_assert(tile_size > 0, "tile_size must be positive");
    __shared__ int first_tile[tile_size];
    __shared__ int second_tile[tile_size];
    __shared__ int first_take_shared;
    __shared__ int second_take_shared;
    __shared__ int output_take_shared;

    const int output_count = first_count + second_count;
    int first_consumed = 0;
    int second_consumed = 0;
    int output_completed = 0;
    while (output_completed < output_count) {
        if (threadIdx.x == 0) {
            const int remaining_output = output_count - output_completed;
            output_take_shared =
                remaining_output < tile_size ? remaining_output : tile_size;
            const int remaining_first = first_count - first_consumed;
            const int remaining_second = second_count - second_consumed;
            first_take_shared =
                co_rank(output_take_shared, first + first_consumed,
                        remaining_first, second + second_consumed,
                        remaining_second);
            second_take_shared = output_take_shared - first_take_shared;
        }
        __syncthreads();

        for (int item = threadIdx.x; item < first_take_shared;
             item += blockDim.x) {
            first_tile[item] = first[first_consumed + item];
        }
        for (int item = threadIdx.x; item < second_take_shared;
             item += blockDim.x) {
            second_tile[item] = second[second_consumed + item];
        }
        __syncthreads();

        const int output_begin = static_cast<int>(
            (static_cast<unsigned long long>(threadIdx.x) *
             output_take_shared) /
            blockDim.x);
        const int output_end = static_cast<int>(
            (static_cast<unsigned long long>(threadIdx.x + 1) *
             output_take_shared) /
            blockDim.x);
        const int first_begin =
            co_rank(output_begin, first_tile, first_take_shared, second_tile,
                    second_take_shared);
        const int second_begin = output_begin - first_begin;
        const int first_end =
            co_rank(output_end, first_tile, first_take_shared, second_tile,
                    second_take_shared);
        const int second_end = output_end - first_end;
        merge_sequential(first_tile + first_begin, first_end - first_begin,
                         second_tile + second_begin,
                         second_end - second_begin,
                         output + output_completed + output_begin);
        __syncthreads();

        first_consumed += first_take_shared;
        second_consumed += second_take_shared;
        output_completed += output_take_shared;
        __syncthreads();
    }
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<int> merge_cpu(const std::vector<int>& first,
                           const std::vector<int>& second) {
    if (!std::is_sorted(first.begin(), first.end()) ||
        !std::is_sorted(second.begin(), second.end())) {
        throw std::invalid_argument("merge inputs must be sorted");
    }
    std::vector<int> output;
    output.reserve(first.size() + second.size());
    std::merge(first.begin(), first.end(), second.begin(), second.end(),
               std::back_inserter(output));
    return output;
}

std::vector<int> co_ranks_cpu(const std::vector<int>& first,
                              const std::vector<int>& second) {
    if (first.size() > static_cast<std::size_t>(INT_MAX) ||
        second.size() > static_cast<std::size_t>(INT_MAX) ||
        first.size() + second.size() > static_cast<std::size_t>(INT_MAX)) {
        throw std::length_error("co-rank fixture exceeds int range");
    }
    std::vector<int> ranks(first.size() + second.size() + 1);
    for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
        ranks[rank] = co_rank(static_cast<int>(rank), first.data(),
                              static_cast<int>(first.size()), second.data(),
                              static_cast<int>(second.size()));
    }
    return ranks;
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
            std::to_string(mismatch));
    }
}

int main(int argc, char** argv) {
    try {
        const std::vector<int> first_example{1, 7, 8, 9, 10};
        const std::vector<int> second_example{7, 10, 10, 12};
        require(co_rank(8, first_example.data(),
                        static_cast<int>(first_example.size()),
                        second_example.data(),
                        static_cast<int>(second_example.size())) == 5,
                "exercise 1 CPU co-rank must be (5,3)");
        const std::vector<int> first_correction{1, 7, 8, 9, 11};
        require(co_rank(6, first_correction.data(),
                        static_cast<int>(first_correction.size()),
                        second_example.data(),
                        static_cast<int>(second_example.size())) == 4,
                "exercise 2 CPU co-rank must be (4,2)");
        require(merge_cpu({}, second_example) == second_example,
                "CPU merge must handle an empty first input");

        std::vector<int> first(149);
        std::vector<int> second(113);
        for (std::size_t index = 0; index < first.size(); ++index) {
            first[index] = static_cast<int>((index / 3) * 2);
        }
        for (std::size_t index = 0; index < second.size(); ++index) {
            second[index] = static_cast<int>((index / 2) * 3);
        }
        const std::vector<int> expected = merge_cpu(first, second);
        const std::vector<int> expected_ranks = co_ranks_cpu(first, second);
        require(expected.size() == first.size() + second.size() &&
                    std::is_sorted(expected.begin(), expected.end()),
                "CPU merge fixture is invalid");

        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            verify(merge_cpu(first, second), expected, "CPU merge repeat");
            verify(co_ranks_cpu(first, second), expected_ranks,
                   "CPU co-rank repeat");
            report_cpu_only("ch13_ex01-03_merge");
            return EXIT_SUCCESS;
        }

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        constexpr int threads = 48;
        constexpr int tile_size = 64;
        if (threads > properties.maxThreadsPerBlock) {
            throw std::runtime_error("merge block exceeds device limit");
        }
        const std::size_t shared_bytes =
            (2 * tile_size + 3) * sizeof(int);
        if (shared_bytes > properties.sharedMemPerBlock) {
            throw std::runtime_error("merge tile exceeds shared-memory limit");
        }

        device_buffer<int> d_first(first.size());
        device_buffer<int> d_second(second.size());
        device_buffer<int> d_output(expected.size());
        device_buffer<int> d_ranks(expected_ranks.size());
        CUDA_CHECK(cudaMemcpy(d_first.get(), first.data(),
                              first.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_second.get(), second.data(),
                              second.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

        const std::size_t rank_blocks =
            1 + (expected_ranks.size() - 1) / 128;
        if (rank_blocks >
            static_cast<std::size_t>(properties.maxGridSize[0])) {
            throw std::length_error("co-rank grid.x exceeds device limit");
        }
        co_rank_all<<<static_cast<unsigned int>(rank_blocks), 128>>>(
            d_first.get(), static_cast<int>(first.size()), d_second.get(),
            static_cast<int>(second.size()), d_ranks.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int> actual_ranks(expected_ranks.size());
        CUDA_CHECK(cudaMemcpy(actual_ranks.data(), d_ranks.get(),
                              actual_ranks.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
        verify(actual_ranks, expected_ranks, "GPU co-ranks");

        tiled_merge_exact_loads<tile_size><<<1, threads>>>(
            d_first.get(), static_cast<int>(first.size()), d_second.get(),
            static_cast<int>(second.size()), d_output.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int> actual(expected.size());
        CUDA_CHECK(cudaMemcpy(actual.data(), d_output.get(),
                              actual.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
        verify(actual, expected, "exact-load tiled merge");

        std::cout << "ch13_ex01-03_merge: co-ranks and tiled merge agree with CPU.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
