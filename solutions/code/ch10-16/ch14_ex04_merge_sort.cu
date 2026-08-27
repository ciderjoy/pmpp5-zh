#include "common/cuda_check.hpp"
#include "ch10-16/reference_algorithms.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

__device__ std::size_t co_rank_device(std::size_t rank, const int* left,
                                      std::size_t left_size,
                                      const int* right,
                                      std::size_t right_size) {
    std::size_t low = rank > right_size ? rank - right_size : 0;
    std::size_t high = rank < left_size ? rank : left_size;
    while (low < high) {
        const std::size_t i = low + (high - low) / 2;
        const std::size_t j = rank - i;
        if (j > 0 && i < left_size && right[j - 1] >= left[i]) {
            low = i + 1;
        } else {
            high = i;
        }
    }
    return low;
}

__device__ void serial_merge(const int* left, std::size_t left_size,
                             const int* right, std::size_t right_size,
                             int* output) {
    std::size_t i = 0;
    std::size_t j = 0;
    std::size_t k = 0;
    while (i < left_size && j < right_size) {
        output[k++] = left[i] <= right[j] ? left[i++] : right[j++];
    }
    while (i < left_size) {
        output[k++] = left[i++];
    }
    while (j < right_size) {
        output[k++] = right[j++];
    }
}

template <int items_per_thread>
__global__ void merge_pass(const int* source, int* destination,
                           std::size_t size, std::size_t width,
                           std::size_t pair_span,
                           std::size_t tiles_per_pair) {
    const std::size_t task = blockIdx.x;
    const std::size_t pair = task / tiles_per_pair;
    const std::size_t tile = task % tiles_per_pair;
    const std::size_t base = pair * pair_span;
    if (base >= size) {
        return;
    }
    const std::size_t left_size =
        width < size - base ? width : size - base;
    const std::size_t right_base = base + left_size;
    const std::size_t right_size =
        width < size - right_base ? width : size - right_base;
    const std::size_t total = left_size + right_size;
    const std::size_t tile_items = blockDim.x * items_per_thread;
    const std::size_t tile_begin = tile * tile_items;
    if (tile_begin >= total) {
        return;
    }
    const std::size_t thread_begin =
        static_cast<std::size_t>(threadIdx.x) * items_per_thread;
    if (thread_begin >= total - tile_begin) {
        return;
    }
    const std::size_t first = tile_begin + thread_begin;
    const std::size_t remaining = total - first;
    const std::size_t last =
        first + (remaining < items_per_thread ? remaining : items_per_thread);
    const int* left = source + base;
    const int* right = left + left_size;
    const std::size_t i0 =
        co_rank_device(first, left, left_size, right, right_size);
    const std::size_t j0 = first - i0;
    const std::size_t i1 =
        co_rank_device(last, left, left_size, right, right_size);
    const std::size_t j1 = last - i1;
    serial_merge(left + i0, i1 - i0, right + j0, j1 - j0,
                 destination + base + first);
}

std::vector<int> merge_sort_cuda(const std::vector<int>& input) {
    if (input.empty()) {
        return {};
    }
    constexpr unsigned block_size = 128;
    constexpr int items_per_thread = 8;
    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (block_size > static_cast<unsigned>(properties.maxThreadsPerBlock)) {
        throw std::runtime_error("merge-sort block exceeds device limit");
    }
    device_buffer<int> first(input.size()), second(input.size());
    CUDA_CHECK(cudaMemcpy(first.get(), input.data(),
                          input.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    int* source = first.get();
    int* destination = second.get();

    for (std::size_t width = 1; width < input.size();) {
        const std::size_t pair_span =
            width > input.size() / 2 ? input.size() : 2 * width;
        const std::size_t pairs =
            1 + (input.size() - 1) / pair_span;
        const std::size_t tile_items = block_size * items_per_thread;
        const std::size_t tiles_per_pair =
            1 + (pair_span - 1) / tile_items;
        if (pairs > std::numeric_limits<std::size_t>::max() /
                        tiles_per_pair) {
            throw std::length_error("merge-sort task count overflow");
        }
        const std::size_t blocks_64 = pairs * tiles_per_pair;
        if (blocks_64 == 0 ||
            blocks_64 > std::numeric_limits<unsigned>::max() ||
            blocks_64 >
                static_cast<std::size_t>(properties.maxGridSize[0])) {
            throw std::length_error("merge-sort grid exceeds grid.x range");
        }
        merge_pass<items_per_thread>
            <<<static_cast<unsigned>(blocks_64), block_size>>>(
                source, destination, input.size(), width, pair_span,
                tiles_per_pair);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(source, destination);
        if (width > input.size() / 2) {
            width = input.size();
        } else {
            width *= 2;
        }
    }
    std::vector<int> output(input.size());
    CUDA_CHECK(cudaMemcpy(output.data(), source, output.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    return output;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        pmpp::reference::test_ch13();
        pmpp::reference::test_ch14();
        std::vector<int> input(1031);
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<int>((i * 37 + i / 7) % 211) - 105;
            if (i % 23 == 0) {
                input[i] = 17;
            }
        }
        const std::vector<int> expected =
            pmpp::reference::merge_sort_reference(input);
        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            pmpp::reference::verify_exact(
                pmpp::reference::merge_sort_reference(input), expected,
                "chapter 14 CPU merge-sort repeat");
            report_cpu_only("ch14_ex04_merge_sort");
            return EXIT_SUCCESS;
        }
        pmpp::reference::verify_exact(merge_sort_cuda(input), expected,
                                      "CUDA merge sort");
        std::cout << "ch14_ex04_merge_sort: CUDA merge sort passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
