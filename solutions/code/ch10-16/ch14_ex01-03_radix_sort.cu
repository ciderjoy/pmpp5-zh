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

template <int block_size>
__device__ int block_exclusive_sum(int value, int* scratch, int& total) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int offset = 1; offset < block_size; offset <<= 1) {
        const int add = threadIdx.x >= static_cast<unsigned>(offset)
                            ? scratch[threadIdx.x - offset]
                            : 0;
        __syncthreads();
        scratch[threadIdx.x] += add;
        __syncthreads();
    }
    total = scratch[block_size - 1];
    return scratch[threadIdx.x] - value;
}

template <int block_size>
__global__ void pack_one_bit(const unsigned* input, unsigned* packed,
                             unsigned* counts, std::size_t size,
                             unsigned shift) {
    __shared__ int scan_storage[block_size];
    __shared__ unsigned local[block_size];

    const std::size_t base =
        static_cast<std::size_t>(blockIdx.x) * block_size;
    const std::size_t index = base + threadIdx.x;
    const int valid = index < size;
    const int bit = valid ? static_cast<int>((input[index] >> shift) & 1U)
                          : 0;
    int ones_before = 0;
    int total_ones = 0;
    ones_before = block_exclusive_sum<block_size>(
        valid ? bit : 0, scan_storage, total_ones);
    const std::size_t remaining = size > base ? size - base : 0;
    const int valid_count = static_cast<int>(
        remaining < block_size ? remaining : block_size);
    const int zeros = valid_count - total_ones;
    if (valid) {
        const int position = bit ? zeros + ones_before
                                 : static_cast<int>(threadIdx.x) - ones_before;
        local[position] = input[index];
    }
    __syncthreads();
    if (static_cast<int>(threadIdx.x) < valid_count) {
        packed[base + threadIdx.x] = local[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        counts[blockIdx.x] = static_cast<unsigned>(zeros);
        counts[gridDim.x + blockIdx.x] =
            static_cast<unsigned>(total_ones);
    }
}

__global__ void scatter_one_bit(const unsigned* packed, unsigned* output,
                                const unsigned* counts,
                                const unsigned* offsets, std::size_t size,
                                unsigned groups) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }
    const unsigned local = threadIdx.x;
    const unsigned zeros = counts[blockIdx.x];
    const unsigned bucket = local < zeros ? 0U : 1U;
    const unsigned rank = local < zeros ? local : local - zeros;
    output[offsets[bucket * groups + blockIdx.x] + rank] = packed[index];
}

template <int block_size, int radix_bits>
__global__ void pack_multibit(const unsigned* input, unsigned* packed,
                              unsigned char* packed_digit,
                              unsigned* counts, std::size_t size,
                              unsigned shift) {
    static_assert(radix_bits >= 1 && radix_bits <= 8,
                  "unsigned char digit requires 1..8 bits");
    constexpr int radix = 1 << radix_bits;
    __shared__ int scan_storage[block_size];
    __shared__ unsigned keys[block_size];
    __shared__ unsigned char digits[block_size];

    const std::size_t base =
        static_cast<std::size_t>(blockIdx.x) * block_size;
    const std::size_t index = base + threadIdx.x;
    const bool valid = index < size;
    const unsigned key = valid ? input[index] : 0U;
    const unsigned digit = (key >> shift) & (radix - 1U);
    int running = 0;
    for (int bucket = 0; bucket < radix; ++bucket) {
        int rank = 0;
        int total = 0;
        const int flag = valid && digit == static_cast<unsigned>(bucket);
        rank = block_exclusive_sum<block_size>(flag, scan_storage, total);
        if (flag != 0) {
            keys[running + rank] = key;
            digits[running + rank] = static_cast<unsigned char>(digit);
        }
        if (threadIdx.x == 0) {
            counts[bucket * gridDim.x + blockIdx.x] =
                static_cast<unsigned>(total);
        }
        running += total;
        __syncthreads();
    }
    const std::size_t remaining = size > base ? size - base : 0;
    const int valid_count = static_cast<int>(
        remaining < block_size ? remaining : block_size);
    if (static_cast<int>(threadIdx.x) < valid_count) {
        packed[base + threadIdx.x] = keys[threadIdx.x];
        packed_digit[base + threadIdx.x] = digits[threadIdx.x];
    }
}

template <int radix_bits>
__global__ void scatter_multibit(const unsigned* packed,
                                 const unsigned char* packed_digit,
                                 unsigned* output, const unsigned* counts,
                                 const unsigned* offsets, std::size_t size,
                                 unsigned groups) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }
    const unsigned digit = packed_digit[index];
    unsigned local_start = 0;
    for (unsigned bucket = 0; bucket < digit; ++bucket) {
        local_start += counts[bucket * groups + blockIdx.x];
    }
    const unsigned rank = threadIdx.x - local_start;
    output[offsets[digit * groups + blockIdx.x] + rank] = packed[index];
}

template <int block_size>
__global__ void pack_one_bit_x4(const unsigned* input, unsigned* packed,
                                unsigned* counts, std::size_t size,
                                unsigned shift) {
    constexpr int coarse = 4;
    __shared__ int scan_storage[block_size];
    __shared__ unsigned local[block_size * coarse];

    const std::size_t block_base =
        static_cast<std::size_t>(blockIdx.x) * block_size * coarse;
    const std::size_t first =
        block_base + static_cast<std::size_t>(threadIdx.x) * coarse;
    unsigned values[coarse]{0, 0, 0, 0};
    int valid = 0;
    if (first + 3 < size &&
        (reinterpret_cast<std::size_t>(input + first) & 15U) == 0) {
        const uint4 vector = *reinterpret_cast<const uint4*>(input + first);
        values[0] = vector.x;
        values[1] = vector.y;
        values[2] = vector.z;
        values[3] = vector.w;
        valid = coarse;
    } else {
        for (int item = 0; item < coarse; ++item) {
            if (first + item < size) {
                values[item] = input[first + item];
                ++valid;
            }
        }
    }
    int local_zeros = 0;
    for (int item = 0; item < valid; ++item) {
        local_zeros += ((values[item] >> shift) & 1U) == 0;
    }
    int zero_base = 0;
    int total_zeros = 0;
    zero_base = block_exclusive_sum<block_size>(
        local_zeros, scan_storage, total_zeros);
    const std::size_t remaining = size > block_base ? size - block_base : 0;
    const int valid_total = static_cast<int>(
        remaining < block_size * coarse ? remaining : block_size * coarse);
    const int valid_before =
        min(static_cast<int>(threadIdx.x) * coarse, valid_total);
    const int one_base = valid_before - zero_base;
    int zero_rank = 0;
    int one_rank = 0;
    for (int item = 0; item < valid; ++item) {
        if (((values[item] >> shift) & 1U) == 0) {
            local[zero_base + zero_rank++] = values[item];
        } else {
            local[total_zeros + one_base + one_rank++] = values[item];
        }
    }
    __syncthreads();
    for (int position = static_cast<int>(threadIdx.x);
         position < valid_total; position += block_size) {
        packed[block_base + position] = local[position];
    }
    if (threadIdx.x == 0) {
        counts[blockIdx.x] = static_cast<unsigned>(total_zeros);
        counts[gridDim.x + blockIdx.x] =
            static_cast<unsigned>(valid_total - total_zeros);
    }
}

template <int block_size>
__global__ void scatter_one_bit_x4(const unsigned* packed,
                                   unsigned* output,
                                   const unsigned* counts,
                                   const unsigned* offsets,
                                   std::size_t size, unsigned groups) {
    constexpr unsigned coarse = 4;
    const std::size_t base =
        static_cast<std::size_t>(blockIdx.x) * block_size * coarse;
    const std::size_t remaining = size > base ? size - base : 0;
    const unsigned valid = static_cast<unsigned>(
        remaining < block_size * coarse ? remaining : block_size * coarse);
    const unsigned zeros = counts[blockIdx.x];
    for (unsigned position = threadIdx.x; position < valid;
         position += block_size) {
        const unsigned bucket = position < zeros ? 0U : 1U;
        const unsigned rank =
            position < zeros ? position : position - zeros;
        output[offsets[bucket * groups + blockIdx.x] + rank] =
            packed[base + position];
    }
}

__global__ void scan_counts_kernel(const unsigned* counts, unsigned* offsets,
                                   int count) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned prefix = 0;
        for (int index = 0; index < count; ++index) {
            offsets[index] = prefix;
            prefix += counts[index];
        }
    }
}

void scan_counts(const device_buffer<unsigned>& counts,
                 const device_buffer<unsigned>& offsets, int count) {
    if (count <= 0) {
        throw std::invalid_argument("count scan must be nonempty");
    }
    scan_counts_kernel<<<1, 1>>>(counts.get(), offsets.get(), count);
    CUDA_CHECK(cudaGetLastError());
}

std::vector<unsigned> sort_one_bit(const std::vector<unsigned>& input,
                                   bool coarse) {
    if (input.empty()) {
        return {};
    }
    if (input.size() > std::numeric_limits<unsigned>::max()) {
        throw std::length_error("radix offsets exceed unsigned range");
    }
    constexpr unsigned block_size = 128;
    const std::size_t items_per_block =
        coarse ? block_size * 4U : block_size;
    const std::size_t group_count =
        1 + (input.size() - 1) / items_per_block;
    if (group_count > static_cast<std::size_t>(
                          std::numeric_limits<int>::max() / 2)) {
        throw std::length_error("radix grid is outside supported range");
    }
    const unsigned groups = static_cast<unsigned>(group_count);
    device_buffer<unsigned> first(input.size()), second(input.size()),
        packed(input.size()), counts(2U * groups), offsets(2U * groups);
    CUDA_CHECK(cudaMemcpy(first.get(), input.data(),
                          input.size() * sizeof(unsigned),
                          cudaMemcpyHostToDevice));
    unsigned* source = first.get();
    unsigned* destination = second.get();
    for (unsigned shift = 0; shift < 32; ++shift) {
        if (coarse) {
            pack_one_bit_x4<block_size><<<groups, block_size>>>(
                source, packed.get(), counts.get(), input.size(), shift);
        } else {
            pack_one_bit<block_size><<<groups, block_size>>>(
                source, packed.get(), counts.get(), input.size(), shift);
        }
        CUDA_CHECK(cudaGetLastError());
        scan_counts(counts, offsets, static_cast<int>(2U * groups));
        if (coarse) {
            scatter_one_bit_x4<block_size><<<groups, block_size>>>(
                packed.get(), destination, counts.get(), offsets.get(),
                input.size(), groups);
        } else {
            scatter_one_bit<<<groups, block_size>>>(
                packed.get(), destination, counts.get(), offsets.get(),
                input.size(), groups);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(source, destination);
    }
    std::vector<unsigned> output(input.size());
    CUDA_CHECK(cudaMemcpy(output.data(), source,
                          output.size() * sizeof(unsigned),
                          cudaMemcpyDeviceToHost));
    return output;
}

std::vector<unsigned> sort_multibit(const std::vector<unsigned>& input) {
    if (input.empty()) {
        return {};
    }
    if (input.size() > std::numeric_limits<unsigned>::max()) {
        throw std::length_error("radix offsets exceed unsigned range");
    }
    constexpr int radix_bits = 4;
    constexpr int radix = 1 << radix_bits;
    constexpr unsigned block_size = 128;
    const std::size_t group_count = 1 + (input.size() - 1) / block_size;
    if (group_count > static_cast<std::size_t>(
                          std::numeric_limits<int>::max() / radix)) {
        throw std::length_error("multibit radix grid is outside range");
    }
    const unsigned groups = static_cast<unsigned>(group_count);
    device_buffer<unsigned> first(input.size()), second(input.size()),
        packed(input.size()), counts(radix * groups), offsets(radix * groups);
    device_buffer<unsigned char> digits(input.size());
    CUDA_CHECK(cudaMemcpy(first.get(), input.data(),
                          input.size() * sizeof(unsigned),
                          cudaMemcpyHostToDevice));
    unsigned* source = first.get();
    unsigned* destination = second.get();
    for (unsigned shift = 0; shift < 32; shift += radix_bits) {
        pack_multibit<block_size, radix_bits><<<groups, block_size>>>(
            source, packed.get(), digits.get(), counts.get(), input.size(),
            shift);
        CUDA_CHECK(cudaGetLastError());
        scan_counts(counts, offsets, static_cast<int>(radix * groups));
        scatter_multibit<radix_bits><<<groups, block_size>>>(
            packed.get(), digits.get(), destination, counts.get(),
            offsets.get(), input.size(), groups);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(source, destination);
    }
    std::vector<unsigned> output(input.size());
    CUDA_CHECK(cudaMemcpy(output.data(), source,
                          output.size() * sizeof(unsigned),
                          cudaMemcpyDeviceToHost));
    return output;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        pmpp::reference::test_ch14();
        std::vector<unsigned> input(1031);
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<unsigned>(
                (i * 2'654'435'761ULL + (i % 17) * 97) & 0xffffffffULL);
            if (i % 19 == 0) {
                input[i] = 7;
            }
        }
        const std::vector<unsigned> expected =
            pmpp::reference::radix_sort_reference(input);
        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            pmpp::reference::verify_exact(
                pmpp::reference::radix_sort_reference(input), expected,
                "chapter 14 CPU radix repeat");
            report_cpu_only("ch14_ex01-03_radix_sort");
            return EXIT_SUCCESS;
        }

        pmpp::reference::verify_exact(sort_one_bit(input, false), expected,
                                      "one-bit radix sort");
        pmpp::reference::verify_exact(sort_multibit(input), expected,
                                      "four-bit radix sort");
        pmpp::reference::verify_exact(sort_one_bit(input, true), expected,
                                      "x4 one-bit radix sort");
        std::cout << "ch14_ex01-03_radix_sort: three CUDA paths passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
