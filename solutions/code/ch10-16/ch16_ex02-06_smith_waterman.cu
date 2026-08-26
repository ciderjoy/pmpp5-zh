#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING

#include "common/cuda_check.hpp"
#include "ch10-16/reference_algorithms.hpp"

#include <cooperative_groups.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace cg = cooperative_groups;

__device__ unsigned acquire_flag(unsigned* flag) {
    const unsigned value = atomicAdd(flag, 0U);
    if (value != 0) {
        __threadfence();
    }
    return value;
}

__device__ void release_flag(unsigned* flag) {
    // Every producer thread orders its table writes before thread 0 publishes
    // completion. A fence executed only by thread 0 would not cover stores
    // issued by the other threads in the block.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicExch(flag, 1U);
    }
    __syncthreads();
}

__device__ __forceinline__ int sw_max(int diagonal, int west, int north) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    return __vimax3_s32_relu(diagonal, west, north);
#else
    return max(0, max(diagonal, max(west, north)));
#endif
}

template <int tile_rows, int tile_columns>
__device__ void smith_waterman_tile(
    int* table, const char* read, const char* reference, int rows,
    int columns, int tile_row, int tile_column, int match, int mismatch,
    int gap, int* shared) {
    constexpr int leading_dimension = tile_columns + 1;
    const int row_base = 1 + tile_row * tile_rows;
    const int column_base = 1 + tile_column * tile_columns;
    for (int column = threadIdx.x; column < tile_columns;
         column += blockDim.x) {
        shared[column + 1] =
            column_base + column < columns
                ? table[static_cast<std::size_t>(row_base - 1) * columns +
                        column_base + column]
                : 0;
    }
    for (int row = threadIdx.x; row < tile_rows; row += blockDim.x) {
        shared[(row + 1) * leading_dimension] =
            row_base + row < rows
                ? table[static_cast<std::size_t>(row_base + row) * columns +
                        column_base - 1]
                : 0;
    }
    if (threadIdx.x == 0) {
        shared[0] =
            table[static_cast<std::size_t>(row_base - 1) * columns +
                  column_base - 1];
    }
    __syncthreads();

    for (int wave = 0; wave < tile_rows + tile_columns - 1; ++wave) {
        const int local_column = threadIdx.x;
        const int local_row = wave - local_column;
        const int global_row = row_base + local_row;
        const int global_column = column_base + local_column;
        if (local_column < tile_columns && local_row >= 0 &&
            local_row < tile_rows && global_row < rows &&
            global_column < columns) {
            const int northwest =
                shared[local_row * leading_dimension + local_column];
            const int north =
                shared[local_row * leading_dimension + local_column + 1];
            const int west = shared[(local_row + 1) * leading_dimension +
                                    local_column];
            const int substitution =
                read[global_row - 1] == reference[global_column - 1]
                    ? match
                    : mismatch;
            shared[(local_row + 1) * leading_dimension +
                   local_column + 1] =
                sw_max(northwest + substitution, west + gap, north + gap);
        }
        __syncthreads();
    }
    for (int position = threadIdx.x; position < tile_rows * tile_columns;
         position += blockDim.x) {
        const int local_row = position / tile_columns;
        const int local_column = position % tile_columns;
        const int global_row = row_base + local_row;
        const int global_column = column_base + local_column;
        if (global_row < rows && global_column < columns) {
            table[static_cast<std::size_t>(global_row) * columns +
                  global_column] =
                shared[(local_row + 1) * leading_dimension +
                       local_column + 1];
        }
    }
    __syncthreads();
}

template <int tile_rows, int tile_columns>
__global__ void smith_waterman_rect_wave(
    int* table, const char* read, const char* reference, int rows,
    int columns, int tile_wave, int match, int mismatch, int gap) {
    static_assert(tile_columns > 0 && tile_columns <= 1024,
                  "one thread is required per tile column");
    __shared__ int shared[(tile_rows + 1) * (tile_columns + 1)];
    const int tile_row = blockIdx.x;
    const int tile_column = tile_wave - tile_row;
    const int tile_row_count = (rows - 1 + tile_rows - 1) / tile_rows;
    const int tile_column_count =
        (columns - 1 + tile_columns - 1) / tile_columns;
    if (tile_row < tile_row_count && tile_column >= 0 &&
        tile_column < tile_column_count) {
        smith_waterman_tile<tile_rows, tile_columns>(
            table, read, reference, rows, columns, tile_row, tile_column,
            match, mismatch, gap, shared);
    }
}

template <int tile_size>
__global__ void smith_waterman_cooperative(
    int* table, const char* read, const char* reference, int rows,
    int columns, int match, int mismatch, int gap) {
    static_assert(tile_size > 0 && tile_size <= 1024,
                  "one thread is required per tile column");
    cg::grid_group grid = cg::this_grid();
    __shared__ int shared[(tile_size + 1) * (tile_size + 1)];
    const int tile_row_count = (rows - 1 + tile_size - 1) / tile_size;
    const int tile_column_count =
        (columns - 1 + tile_size - 1) / tile_size;
    const int tile_row = blockIdx.x;
    for (int wave = 0; wave < tile_row_count + tile_column_count - 1;
         ++wave) {
        const int tile_column = wave - tile_row;
        if (tile_row < tile_row_count && tile_column >= 0 &&
            tile_column < tile_column_count) {
            smith_waterman_tile<tile_size, tile_size>(
                table, read, reference, rows, columns, tile_row,
                tile_column, match, mismatch, gap, shared);
        }
        grid.sync();
    }
}

template <int tile_size>
__global__ void smith_waterman_unidirectional(
    int* table, const char* read, const char* reference, int rows,
    int columns, int match, int mismatch, int gap, unsigned* row_counter,
    unsigned* done) {
    static_assert(tile_size > 0 && tile_size <= 1024,
                  "launch exactly tile_size threads");
    __shared__ int shared[(tile_size + 1) * (tile_size + 1)];
    __shared__ unsigned tile_row_shared;
    if (threadIdx.x == 0) {
        tile_row_shared = atomicAdd(row_counter, 1U);
    }
    __syncthreads();
    const unsigned tile_row = tile_row_shared;
    const unsigned tile_row_count =
        static_cast<unsigned>((rows - 1 + tile_size - 1) / tile_size);
    const unsigned tile_column_count =
        static_cast<unsigned>((columns - 1 + tile_size - 1) / tile_size);
    if (tile_row >= tile_row_count) {
        return;
    }
    for (unsigned tile_column = 0; tile_column < tile_column_count;
        ++tile_column) {
        if (tile_row > 0 && threadIdx.x == 0) {
            unsigned* above =
                done + (tile_row - 1) * tile_column_count + tile_column;
            while (acquire_flag(above) == 0) {
            }
        }
        __syncthreads();
        smith_waterman_tile<tile_size, tile_size>(
            table, read, reference, rows, columns,
            static_cast<int>(tile_row), static_cast<int>(tile_column), match,
            mismatch, gap, shared);
        release_flag(done + tile_row * tile_column_count + tile_column);
    }
}

template <int tile_size>
__global__ void smith_waterman_hypertile(
    int* table, const char* read, const char* reference, int rows,
    int columns, int match, int mismatch, int gap, unsigned* row_counter,
    unsigned* done) {
    static_assert(tile_size > 0 && tile_size <= 1024,
                  "launch exactly tile_size threads");
    __shared__ int tile[tile_size * tile_size];
    __shared__ unsigned tile_row_shared;
    if (threadIdx.x == 0) {
        tile_row_shared = atomicAdd(row_counter, 1U);
    }
    __syncthreads();
    const unsigned tile_row = tile_row_shared;
    const unsigned tile_row_count =
        static_cast<unsigned>((rows - 1 + tile_size - 1) / tile_size);
    const unsigned tile_column_count =
        static_cast<unsigned>((columns - 1 + tile_size - 1) / tile_size);
    if (tile_row >= tile_row_count) {
        return;
    }
    const unsigned pitch = tile_column_count + 1;
    for (unsigned tile_column = 0; tile_column <= tile_column_count;
         ++tile_column) {
        if (tile_row > 0 && threadIdx.x == 0) {
            const unsigned needed = tile_column < tile_column_count
                                        ? tile_column + 1
                                        : tile_column;
            unsigned* above = done + (tile_row - 1) * pitch + needed;
            while (acquire_flag(above) == 0) {
            }
        }
        __syncthreads();
        for (int position = threadIdx.x;
             position < tile_size * tile_size; position += tile_size) {
            tile[position] = 0;
        }
        __syncthreads();

        const int local_row = threadIdx.x;
        const int global_row =
            1 + static_cast<int>(tile_row) * tile_size + local_row;
        for (int local_column = 0; local_column < tile_size;
             ++local_column) {
            const int global_column =
                1 + static_cast<int>(tile_column) * tile_size +
                local_column - local_row;
            if (global_row < rows && global_column >= 1 &&
                global_column < columns) {
                const int north =
                    local_row == 0 || local_column == 0
                        ? table[static_cast<std::size_t>(global_row - 1) *
                                    columns +
                                global_column]
                        : tile[(local_row - 1) * tile_size +
                               local_column - 1];
                const int west =
                    local_column == 0
                        ? table[static_cast<std::size_t>(global_row) *
                                    columns +
                                global_column - 1]
                        : tile[local_row * tile_size + local_column - 1];
                const int northwest =
                    local_row == 0 || local_column <= 1
                        ? table[static_cast<std::size_t>(global_row - 1) *
                                    columns +
                                global_column - 1]
                        : tile[(local_row - 1) * tile_size +
                               local_column - 2];
                const int substitution =
                    read[global_row - 1] == reference[global_column - 1]
                        ? match
                        : mismatch;
                tile[local_row * tile_size + local_column] =
                    sw_max(northwest + substitution, west + gap,
                           north + gap);
            }
            __syncthreads();
        }
        for (int row = 0; row < tile_size; ++row) {
            const int global_output_row =
                1 + static_cast<int>(tile_row) * tile_size + row;
            const int global_column =
                1 + static_cast<int>(tile_column) * tile_size +
                static_cast<int>(threadIdx.x) - row;
            if (global_output_row < rows && global_column >= 1 &&
                global_column < columns) {
                table[static_cast<std::size_t>(global_output_row) * columns +
                      global_column] = tile[row * tile_size + threadIdx.x];
            }
        }
        __syncthreads();
        release_flag(done + tile_row * pitch + tile_column);
    }
}

void reset_table(const device_buffer<int>& table, std::size_t elements) {
    CUDA_CHECK(cudaMemset(table.get(), 0, elements * sizeof(int)));
}

void verify_device_table(const device_buffer<int>& table,
                         const std::vector<int>& expected,
                         const char* label) {
    std::vector<int> actual(expected.size());
    CUDA_CHECK(cudaMemcpy(actual.data(), table.get(),
                          actual.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    pmpp::reference::verify_exact(actual, expected, label);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        pmpp::reference::test_ch16();
        constexpr int tile_size = 8;
        constexpr int match = 2;
        constexpr int mismatch = -1;
        constexpr int gap = -2;
        const std::string read{"GATTACAGATT"};
        const std::string reference{"GCATGCUACATGA"};
        const int rows = static_cast<int>(read.size()) + 1;
        const int columns = static_cast<int>(reference.size()) + 1;
        const std::size_t table_elements =
            static_cast<std::size_t>(rows) * columns;
        const std::vector<int> expected = pmpp::reference::smith_waterman(
            read, reference, {match, mismatch, gap});
        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            pmpp::reference::verify_exact(
                pmpp::reference::smith_waterman(
                    read, reference, {match, mismatch, gap}),
                expected, "chapter 16 CPU Smith-Waterman repeat");
            report_cpu_only("ch16_ex02-06_smith_waterman");
            return EXIT_SUCCESS;
        }

        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        if (tile_size > properties.maxThreadsPerBlock ||
            (tile_size + 1) * (tile_size + 1) * sizeof(int) >
                properties.sharedMemPerBlock) {
            throw std::runtime_error("Smith-Waterman tile exceeds resources");
        }
        const int tile_row_count =
            (rows - 1 + tile_size - 1) / tile_size;
        const int tile_column_count =
            (columns - 1 + tile_size - 1) / tile_size;
        if (tile_row_count == 0 || tile_column_count == 0) {
            throw std::runtime_error("fixture must contain two sequences");
        }

        device_buffer<int> device_table(table_elements);
        device_buffer<char> device_read(read.size()),
            device_reference(reference.size());
        device_buffer<unsigned> row_counter(1),
            done(static_cast<std::size_t>(tile_row_count) *
                 (tile_column_count + 1));
        CUDA_CHECK(cudaMemcpy(device_read.get(), read.data(), read.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_reference.get(), reference.data(),
                              reference.size(), cudaMemcpyHostToDevice));

        reset_table(device_table, table_elements);
        for (int wave = 0;
             wave < tile_row_count + tile_column_count - 1; ++wave) {
            smith_waterman_rect_wave<tile_size, tile_size>
                <<<tile_row_count, tile_size>>>(
                    device_table.get(), device_read.get(),
                    device_reference.get(), rows, columns, wave, match,
                    mismatch, gap);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        verify_device_table(device_table, expected, "rectangular wave");

        int cooperative_supported = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(
            &cooperative_supported, cudaDevAttrCooperativeLaunch, device));
        int active_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_per_sm, smith_waterman_cooperative<tile_size>, tile_size,
            0));
        if (cooperative_supported != 0 &&
            tile_row_count <= active_per_sm * properties.multiProcessorCount) {
            reset_table(device_table, table_elements);
            int* table_pointer = device_table.get();
            char* read_pointer = device_read.get();
            char* reference_pointer = device_reference.get();
            int rows_argument = rows;
            int columns_argument = columns;
            int match_argument = match;
            int mismatch_argument = mismatch;
            int gap_argument = gap;
            void* arguments[]{&table_pointer, &read_pointer,
                              &reference_pointer, &rows_argument,
                              &columns_argument, &match_argument,
                              &mismatch_argument, &gap_argument};
            CUDA_CHECK(cudaLaunchCooperativeKernel(
                reinterpret_cast<void*>(
                    smith_waterman_cooperative<tile_size>),
                dim3(tile_row_count), dim3(tile_size), arguments, 0,
                nullptr));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            verify_device_table(device_table, expected, "cooperative wave");
        } else {
            std::cout << "cooperative path skipped: device capability or "
                         "residency limit.\n";
        }

        reset_table(device_table, table_elements);
        CUDA_CHECK(cudaMemset(row_counter.get(), 0, sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(
            done.get(), 0,
            static_cast<std::size_t>(tile_row_count) * tile_column_count *
                sizeof(unsigned)));
        smith_waterman_unidirectional<tile_size>
            <<<tile_row_count, tile_size>>>(
                device_table.get(), device_read.get(),
                device_reference.get(), rows, columns, match, mismatch, gap,
                row_counter.get(), done.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        verify_device_table(device_table, expected, "unidirectional wave");

        reset_table(device_table, table_elements);
        CUDA_CHECK(cudaMemset(row_counter.get(), 0, sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(
            done.get(), 0,
            static_cast<std::size_t>(tile_row_count) *
                (tile_column_count + 1) * sizeof(unsigned)));
        smith_waterman_hypertile<tile_size>
            <<<tile_row_count, tile_size>>>(
                device_table.get(), device_read.get(),
                device_reference.get(), rows, columns, match, mismatch, gap,
                row_counter.get(), done.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        verify_device_table(device_table, expected, "hypertile wave");

        std::cout << "ch16_ex02-06_smith_waterman: CUDA paths passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
