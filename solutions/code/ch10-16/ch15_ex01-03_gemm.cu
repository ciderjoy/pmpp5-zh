#include "common/cuda_check.hpp"
#include "ch10-16/reference_algorithms.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

__device__ __forceinline__ void load_tile_vec4(
    const float* input, unsigned leading_dimension, unsigned max_row,
    unsigned max_column, float* shared, unsigned shared_leading_dimension,
    unsigned height, unsigned width) {
    const unsigned vector_columns = (width + 3) / 4;
    const unsigned tasks = height * vector_columns;
    for (unsigned task = threadIdx.x; task < tasks;
         task += blockDim.x) {
        const unsigned row = task / vector_columns;
        const unsigned column = 4 * (task % vector_columns);
        float values[4]{0.0f, 0.0f, 0.0f, 0.0f};
        const bool full = row < max_row && column + 3 < max_column &&
                          column + 3 < width;
        const float* source =
            full ? input + static_cast<std::size_t>(row) *
                               leading_dimension + column
                 : nullptr;
        if (full &&
            (reinterpret_cast<std::size_t>(source) & 15U) == 0) {
            const float4 vector =
                *reinterpret_cast<const float4*>(source);
            values[0] = vector.x;
            values[1] = vector.y;
            values[2] = vector.z;
            values[3] = vector.w;
        } else {
#pragma unroll
            for (unsigned item = 0; item < 4; ++item) {
                if (row < max_row && column + item < max_column &&
                    column + item < width) {
                    values[item] =
                        input[static_cast<std::size_t>(row) *
                                  leading_dimension +
                              column + item];
                }
            }
        }
#pragma unroll
        for (unsigned item = 0; item < 4; ++item) {
            if (column + item < width) {
                shared[static_cast<std::size_t>(row) *
                           shared_leading_dimension +
                       column + item] = values[item];
            }
        }
    }
}

template <int tile_columns>
__device__ __forceinline__ void write_tile_vec4(
    float* output, std::size_t leading_dimension, std::size_t max_row,
    std::size_t max_column, std::size_t base_row, std::size_t base_column,
    const float registers[][tile_columns], unsigned rows, unsigned columns) {
    const unsigned write_columns =
        columns < static_cast<unsigned>(tile_columns)
            ? columns
            : static_cast<unsigned>(tile_columns);
    for (unsigned local_row = 0; local_row < rows; ++local_row) {
        if (base_row >= max_row ||
            static_cast<std::size_t>(local_row) >= max_row - base_row) {
            continue;
        }
        const std::size_t global_row = base_row + local_row;
        for (unsigned local_column = 0; local_column < write_columns;
             local_column += 4) {
            if (base_column >= max_column ||
                static_cast<std::size_t>(local_column) >=
                    max_column - base_column) {
                continue;
            }
            const std::size_t global_column = base_column + local_column;
            const bool full = write_columns - local_column >= 4 &&
                              max_column - global_column >= 4;
            if (full) {
                float* destination =
                    output + global_row * leading_dimension + global_column;
                if ((reinterpret_cast<std::size_t>(destination) & 15U) ==
                    0) {
                    const float4 vector = make_float4(
                        registers[local_row][local_column],
                        registers[local_row][local_column + 1],
                        registers[local_row][local_column + 2],
                        registers[local_row][local_column + 3]);
                    *reinterpret_cast<float4*>(destination) = vector;
                } else {
#pragma unroll
                    for (unsigned item = 0; item < 4; ++item) {
                        destination[item] =
                            registers[local_row][local_column + item];
                    }
                }
            } else {
                float* destination =
                    output + global_row * leading_dimension + global_column;
#pragma unroll
                for (unsigned item = 0; item < 4; ++item) {
                    if (item < write_columns - local_column &&
                        static_cast<std::size_t>(item) <
                            max_column - global_column) {
                        destination[item] =
                            registers[local_row][local_column + item];
                    }
                }
            }
        }
    }
}

template <int tile_height, int tile_width>
__global__ void tile_roundtrip(const float* input, float* output,
                               unsigned leading_dimension,
                               unsigned valid_rows,
                               unsigned valid_columns) {
    __shared__ float tile[tile_height][tile_width];
    load_tile_vec4(input, leading_dimension, valid_rows, valid_columns,
                   &tile[0][0], tile_width, tile_height, tile_width);
    __syncthreads();
    float registers[1][tile_width]{};
    if (threadIdx.x < tile_height) {
#pragma unroll
        for (int column = 0; column < tile_width; ++column) {
            registers[0][column] = tile[threadIdx.x][column];
        }
    }
    write_tile_vec4<tile_width>(output, leading_dimension, valid_rows,
                                valid_columns, threadIdx.x, 0, registers, 1,
                                tile_width);
}

template <int block_rows = 128, int block_columns = 128,
          int block_inner = 8>
__global__ void gemm_rearranged(const float* left, const float* right,
                                float* output, int rows, int columns,
                                int inner) {
    static_assert(block_rows == 128 && block_columns == 128 &&
                      block_inner == 8,
                  "mapping constants");
    __shared__ float left_tile[block_rows][block_inner + 1];
    __shared__ float right_tile[block_inner][block_columns];
    float accumulators[4][4][4]{};

    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int lane = thread & 31;
    const int warp_row = warp / 4;
    const int warp_column = warp % 4;
    const int lane_row = lane / 4;
    const int lane_column = lane % 4;
    const int local_row = warp_row * 64 + lane_row * 4;
    const int local_column = warp_column * 32 + lane_column * 4;
    const int block_row = blockIdx.y * block_rows;
    const int block_column = blockIdx.x * block_columns;

    for (int inner_base = 0; inner_base < inner;
         inner_base += block_inner) {
        for (int position = thread; position < block_rows * block_inner;
             position += blockDim.x) {
            const int row = position / block_inner;
            const int k = position % block_inner;
            const int global_row = block_row + row;
            const int global_inner = inner_base + k;
            left_tile[row][k] =
                global_row < rows && global_inner < inner
                    ? left[static_cast<std::size_t>(global_row) * inner +
                           global_inner]
                    : 0.0f;
        }
        for (int position = thread;
             position < block_inner * block_columns;
             position += blockDim.x) {
            const int k = position / block_columns;
            const int column = position % block_columns;
            const int global_inner = inner_base + k;
            const int global_column = block_column + column;
            right_tile[k][column] =
                global_inner < inner && global_column < columns
                    ? right[static_cast<std::size_t>(global_inner) *
                                columns +
                            global_column]
                    : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < block_inner; ++k) {
#pragma unroll
            for (int quadrant = 0; quadrant < 4; ++quadrant) {
                const int row_offset = (quadrant / 2) * 32;
                const int column_offset = (quadrant % 2) * 16;
#pragma unroll
                for (int row = 0; row < 4; ++row) {
                    const float left_value =
                        left_tile[local_row + row_offset + row][k];
#pragma unroll
                    for (int column = 0; column < 4; ++column) {
                        accumulators[quadrant][row][column] +=
                            left_value *
                            right_tile[k][local_column + column_offset +
                                          column];
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int quadrant = 0; quadrant < 4; ++quadrant) {
        const int row_offset = (quadrant / 2) * 32;
        const int column_offset = (quadrant % 2) * 16;
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            const int global_row =
                block_row + local_row + row_offset + row;
            const int global_column =
                block_column + local_column + column_offset;
            if (global_row < rows && global_column < columns) {
                float* destination =
                    output + static_cast<std::size_t>(global_row) * columns +
                    global_column;
                if (global_column + 3 < columns &&
                    (reinterpret_cast<std::size_t>(destination) & 15U) ==
                        0) {
                    *reinterpret_cast<float4*>(destination) = make_float4(
                        accumulators[quadrant][row][0],
                        accumulators[quadrant][row][1],
                        accumulators[quadrant][row][2],
                        accumulators[quadrant][row][3]);
                } else {
                    for (int column = 0;
                         column < 4 && global_column + column < columns;
                         ++column) {
                        destination[column] =
                            accumulators[quadrant][row][column];
                    }
                }
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        pmpp::reference::test_ch15();
        constexpr unsigned copy_rows = 13;
        constexpr unsigned copy_columns = 19;
        std::vector<float> copy_input(copy_rows * copy_columns);
        for (std::size_t i = 0; i < copy_input.size(); ++i) {
            copy_input[i] =
                static_cast<float>(static_cast<int>(i % 29) - 14) / 8.0f;
        }

        constexpr int rows = 131;
        constexpr int inner = 17;
        constexpr int columns = 130;
        std::vector<float> left(static_cast<std::size_t>(rows) * inner);
        std::vector<float> right(static_cast<std::size_t>(inner) * columns);
        for (std::size_t i = 0; i < left.size(); ++i) {
            left[i] =
                static_cast<float>(static_cast<int>(i % 23) - 11) / 16.0f;
        }
        for (std::size_t i = 0; i < right.size(); ++i) {
            right[i] =
                static_cast<float>(static_cast<int>(i % 19) - 9) / 16.0f;
        }
        const std::vector<float> expected =
            pmpp::reference::matmul(left, right, rows, inner, columns);
        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            pmpp::reference::verify_close(
                pmpp::reference::matmul(left, right, rows, inner, columns),
                expected, 0.0f, 0.0f, "chapter 15 CPU GEMM repeat");
            report_cpu_only("ch15_ex01-03_gemm");
            return EXIT_SUCCESS;
        }

        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        if (properties.maxThreadsPerBlock < 256) {
            throw std::runtime_error("device cannot launch 256-thread GEMM");
        }

        device_buffer<float> device_copy_input(copy_input.size()),
            device_copy_output(copy_input.size());
        CUDA_CHECK(cudaMemcpy(device_copy_input.get(), copy_input.data(),
                              copy_input.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        tile_roundtrip<16, 20><<<1, 64>>>(
            device_copy_input.get(), device_copy_output.get(), copy_columns,
            copy_rows, copy_columns);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> copy_output(copy_input.size());
        CUDA_CHECK(cudaMemcpy(copy_output.data(), device_copy_output.get(),
                              copy_output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        pmpp::reference::verify_exact(copy_output, copy_input,
                                      "vector tile roundtrip");

        device_buffer<float> device_left(left.size()),
            device_right(right.size()), device_output(expected.size());
        CUDA_CHECK(cudaMemcpy(device_left.get(), left.data(),
                              left.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_right.get(), right.data(),
                              right.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        const dim3 block(256);
        const dim3 grid((columns + 127) / 128, (rows + 127) / 128);
        gemm_rearranged<<<grid, block>>>(
            device_left.get(), device_right.get(), device_output.get(), rows,
            columns, inner);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> actual(expected.size());
        CUDA_CHECK(cudaMemcpy(actual.data(), device_output.get(),
                              actual.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        pmpp::reference::verify_close(actual, expected, 2.0e-4f, 2.0e-4f,
                                      "rearranged GEMM");
        std::cout << "ch15_ex01-03_gemm: vector copy and GEMM passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
