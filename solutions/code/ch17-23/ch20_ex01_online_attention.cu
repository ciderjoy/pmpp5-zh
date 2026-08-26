#include "common/cuda_check.hpp"
#include "ch17_23_cpu_algorithms.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cuda/std/functional>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int logical_warp_threads = 32;
constexpr int block_threads = 512;
constexpr int warps_per_block = block_threads / logical_warp_threads;
constexpr int query_tile_rows = 32;
constexpr int rows_per_warp = query_tile_rows / warps_per_block;
constexpr int key_tile_rows = 32;
constexpr int feature_count = 128;
constexpr int features_per_lane = feature_count / logical_warp_threads;
constexpr int key_tile_elements = key_tile_rows * feature_count;
constexpr int padded_key_elements =
    key_tile_elements + (key_tile_elements >> 5);

static_assert(rows_per_warp == 2);
static_assert(features_per_lane == 4);

__device__ __forceinline__ int padded_index(int logical_index) {
    return logical_index + (logical_index >> 5);
}

__device__ inline void initialize(
    float output_state[rows_per_warp][features_per_lane],
    float denominator[rows_per_warp], float maximum[rows_per_warp]) {
#pragma unroll
    for (int row = 0; row < rows_per_warp; ++row) {
        denominator[row] = 0.0f;
        maximum[row] = -CUDART_INF_F;
#pragma unroll
        for (int feature = 0; feature < features_per_lane; ++feature) {
            output_state[row][feature] = 0.0f;
        }
    }
}

__device__ inline void load_key_and_value(const float* key,
                                          const float* value,
                                          float* key_transposed,
                                          float* value_tile, int tile) {
    for (int index = static_cast<int>(threadIdx.x);
         index < key_tile_elements; index += block_threads) {
        const int tile_row = index / feature_count;
        const int feature = index % feature_count;
        const std::size_t global =
            static_cast<std::size_t>(tile * key_tile_rows + tile_row) *
                feature_count +
            feature;
        const int transposed = feature * key_tile_rows + tile_row;
        key_transposed[padded_index(transposed)] = key[global];
        value_tile[index] = value[global];
    }
}

__global__ void flashattention_forward_kernel(
    const float* query, const float* key, const float* value, int rows,
    float scaling, float* normalizer, float* output, unsigned* bad) {
    using WarpReduction = cub::WarpReduce<float, logical_warp_threads>;
    __shared__ float key_transposed[padded_key_elements];
    __shared__ float score_or_probability[query_tile_rows][key_tile_rows];
    __shared__ float value_tile[key_tile_rows][feature_count];
    __shared__ typename WarpReduction::TempStorage
        temp_store[warps_per_block];

    const int lane = static_cast<int>(threadIdx.x) &
                     (logical_warp_threads - 1);
    const int warp = static_cast<int>(threadIdx.x) / logical_warp_threads;
    const int query_tiles = rows / query_tile_rows;
    const int key_tiles = rows / key_tile_rows;

    for (int query_tile = static_cast<int>(blockIdx.x);
         query_tile < query_tiles; query_tile += static_cast<int>(gridDim.x)) {
        float output_state[rows_per_warp][features_per_lane];
        float denominator[rows_per_warp];
        float maximum[rows_per_warp];
        float query_state[rows_per_warp][features_per_lane];
        initialize(output_state, denominator, maximum);

#pragma unroll
        for (int local_row = 0; local_row < rows_per_warp; ++local_row) {
            const int row = query_tile * query_tile_rows +
                            warp * rows_per_warp + local_row;
#pragma unroll
            for (int part = 0; part < features_per_lane; ++part) {
                const int feature = lane + part * logical_warp_threads;
                query_state[local_row][part] =
                    query[static_cast<std::size_t>(row) * feature_count +
                          feature];
            }
        }

        for (int key_tile = 0; key_tile < key_tiles; ++key_tile) {
            load_key_and_value(key, value, key_transposed,
                               &value_tile[0][0], key_tile);
            __syncthreads();

#pragma unroll
            for (int local_row = 0; local_row < rows_per_warp;
                 ++local_row) {
                const int shared_row = warp * rows_per_warp + local_row;
                const int row = query_tile * query_tile_rows + shared_row;
                const int column = key_tile * key_tile_rows + lane;
                float score = 0.0f;
#pragma unroll
                for (int feature = 0; feature < feature_count; ++feature) {
                    const float q = __shfl_sync(
                        0xffffffffU,
                        query_state[local_row]
                                   [feature / logical_warp_threads],
                        feature % logical_warp_threads);
                    score += q * key_transposed[padded_index(
                                     feature * key_tile_rows + lane)];
                }
                score = row < column ? -CUDART_INF_F : score * scaling;

                const float maximum_lane_zero =
                    WarpReduction(temp_store[warp])
                        .Reduce(score, ::cuda::maximum<>{});
                const float tile_maximum = __shfl_sync(
                    0xffffffffU, maximum_lane_zero, 0);
                __syncwarp();

                const float new_maximum =
                    fmaxf(maximum[local_row], tile_maximum);
                const float previous_scale =
                    expf(maximum[local_row] - new_maximum);
                const float probability = expf(score - new_maximum);
                score_or_probability[shared_row][lane] = probability;
                __syncwarp();

                const float sum_lane_zero =
                    WarpReduction(temp_store[warp])
                        .Reduce(probability, ::cuda::std::plus<>{});
                const float tile_sum =
                    __shfl_sync(0xffffffffU, sum_lane_zero, 0);
                __syncwarp();

#pragma unroll
                for (int part = 0; part < features_per_lane; ++part) {
                    const int feature = lane + part * logical_warp_threads;
                    float contribution = 0.0f;
#pragma unroll
                    for (int tile_column = 0;
                         tile_column < key_tile_rows; ++tile_column) {
                        contribution +=
                            score_or_probability[shared_row][tile_column] *
                            value_tile[tile_column][feature];
                    }
                    output_state[local_row][part] =
                        output_state[local_row][part] * previous_scale +
                        contribution;
                }
                denominator[local_row] =
                    denominator[local_row] * previous_scale + tile_sum;
                maximum[local_row] = new_maximum;
                __syncwarp();
            }

            // Every warp must finish reading the shared K/V/S tiles before
            // any thread overwrites them for the next key tile.
            __syncthreads();
        }

#pragma unroll
        for (int local_row = 0; local_row < rows_per_warp; ++local_row) {
            const int row = query_tile * query_tile_rows +
                            warp * rows_per_warp + local_row;
            if (!(denominator[local_row] > 0.0f) ||
                !isfinite(denominator[local_row])) {
                atomicExch(bad, 1U);
                continue;
            }
            if (lane == 0) {
                normalizer[row] = denominator[local_row];
            }
#pragma unroll
            for (int part = 0; part < features_per_lane; ++part) {
                const int feature = lane + part * logical_warp_threads;
                output[static_cast<std::size_t>(row) * feature_count +
                       feature] =
                    output_state[local_row][part] /
                    denominator[local_row];
            }
        }
    }
}

__global__ void warp_reduce_demo(const float* input, float* output,
                                 int valid_items) {
    constexpr int demo_warps = 2;
    using WarpReduction = cub::WarpReduce<float, logical_warp_threads>;
    __shared__ typename WarpReduction::TempStorage temp_store[demo_warps];
    const int lane = static_cast<int>(threadIdx.x) % logical_warp_threads;
    const int warp = static_cast<int>(threadIdx.x) / logical_warp_threads;
    if (warp >= demo_warps) {
        return;
    }
    const int input_index = warp * logical_warp_threads + lane;
    const float thread_value =
        lane < valid_items ? input[input_index] : 0.0f;
    const float aggregate = WarpReduction(temp_store[warp])
                                .Reduce(thread_value,
                                        ::cuda::std::plus<>{});
    if (lane == 0) {
        output[warp] = aggregate;
    }
}

struct GpuAttentionResult {
    pmpp_examples::AttentionResult attention;
    std::vector<float> warp_sums;
};

GpuAttentionResult run_attention_gpu(const std::vector<float>& query,
                                     const std::vector<float>& key,
                                     const std::vector<float>& value, int rows,
                                     int features, float scaling) {
    pmpp_examples::validate_attention(query, key, value, rows, features,
                                      scaling);
    if (features != feature_count || rows % query_tile_rows != 0) {
        throw std::invalid_argument(
            "teaching kernel requires d == 128 and N % 32 == 0");
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes,
                                     flashattention_forward_kernel));
    if (properties.maxThreadsPerBlock < block_threads ||
        properties.sharedMemPerBlock < attributes.sharedSizeBytes) {
        throw std::runtime_error(
            "device cannot launch the 512-thread teaching kernel");
    }
    int active_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, flashattention_forward_kernel, block_threads,
        0));
    if (active_blocks_per_sm <= 0) {
        throw std::runtime_error("teaching kernel has zero occupancy");
    }
    const int row_tiles = rows / query_tile_rows;
    const int resident_blocks =
        active_blocks_per_sm * properties.multiProcessorCount;
    const int blocks = std::min(row_tiles, resident_blocks);

    device_buffer<float> device_query(query.size());
    device_buffer<float> device_key(key.size());
    device_buffer<float> device_value(value.size());
    device_buffer<float> device_output(query.size());
    device_buffer<float> device_normalizer(static_cast<std::size_t>(rows));
    device_buffer<unsigned> device_bad(1);
    device_buffer<float> device_warp_input(64);
    device_buffer<float> device_warp_output(2);
    CUDA_CHECK(cudaMemcpy(device_query.get(), query.data(),
                          query.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_key.get(), key.data(),
                          key.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_value.get(), value.data(),
                          value.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_bad.get(), 0, sizeof(unsigned)));
    std::vector<float> warp_input(64);
    for (std::size_t index = 0; index < warp_input.size(); ++index) {
        warp_input[index] = static_cast<float>(index + 1);
    }
    CUDA_CHECK(cudaMemcpy(device_warp_input.get(), warp_input.data(),
                          warp_input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    flashattention_forward_kernel<<<blocks, block_threads>>>(
        device_query.get(), device_key.get(), device_value.get(), rows, scaling,
        device_normalizer.get(), device_output.get(), device_bad.get());
    CUDA_CHECK(cudaGetLastError());
    warp_reduce_demo<<<1, 64>>>(device_warp_input.get(),
                               device_warp_output.get(), 32);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned bad = 0;
    CUDA_CHECK(cudaMemcpy(&bad, device_bad.get(), sizeof(unsigned),
                          cudaMemcpyDeviceToHost));
    if (bad != 0) {
        throw std::runtime_error("GPU causal attention produced bad state");
    }
    GpuAttentionResult result;
    result.attention.output.resize(query.size());
    result.attention.normalizer.resize(static_cast<std::size_t>(rows));
    CUDA_CHECK(cudaMemcpy(result.attention.output.data(), device_output.get(),
                          query.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.attention.normalizer.data(),
                          device_normalizer.get(),
                          static_cast<std::size_t>(rows) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    result.warp_sums.resize(2);
    CUDA_CHECK(cudaMemcpy(result.warp_sums.data(), device_warp_output.get(),
                          result.warp_sums.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return result;
}

void make_inputs(std::vector<float>& query, std::vector<float>& key,
                 std::vector<float>& value, int rows) {
    const std::size_t count =
        static_cast<std::size_t>(rows) * feature_count;
    query.resize(count);
    key.resize(count);
    value.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        query[index] =
            static_cast<float>(static_cast<int>(index % 17) - 8) * 0.015f;
        key[index] =
            static_cast<float>(static_cast<int>(index % 19) - 9) * 0.012f;
        value[index] =
            static_cast<float>(static_cast<int>(index % 23) - 11) * 0.02f;
    }
}

void run_cpu_assertions(const std::vector<float>& query,
                        const std::vector<float>& key,
                        const std::vector<float>& value, int rows,
                        float scaling) {
    const pmpp_examples::AttentionResult reference =
        pmpp_examples::attention_causal_reference_cpu(
            query, key, value, rows, feature_count, scaling);
    const pmpp_examples::AttentionResult online =
        pmpp_examples::attention_causal_online_cpu(
            query, key, value, rows, feature_count, scaling, key_tile_rows);
    pmpp_examples::expect_near(online.output, reference.output, 2.0e-6f,
                               2.0e-5f, "CPU causal online attention");
    pmpp_examples::expect_near(online.normalizer, reference.normalizer,
                               2.0e-6f, 2.0e-5f,
                               "CPU causal online normalizer");
    for (int feature = 0; feature < feature_count; ++feature) {
        pmpp_examples::expect(
            std::fabs(reference.output[static_cast<std::size_t>(feature)] -
                      value[static_cast<std::size_t>(feature)]) < 1.0e-6f,
            "causal row zero contains only key zero");
    }
    pmpp_examples::expect(
        std::fabs(reference.normalizer[0] - 1.0f) < 1.0e-6f,
        "causal row-zero normalizer");
    bool rejected = false;
    try {
        (void)pmpp_examples::attention_causal_online_cpu(
            query, key, value, rows, feature_count, scaling, 0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    pmpp_examples::expect(rejected, "CPU attention tile validation");

    std::vector<float> extreme_query(query.size(), 0.0f);
    std::vector<float> extreme_key(key.size(), 0.0f);
    const int last_row = rows - 1;
    for (int feature = 0; feature < feature_count; ++feature) {
        extreme_query[static_cast<std::size_t>(last_row) * feature_count +
                      feature] = 1.0f;
        extreme_key[static_cast<std::size_t>(key_tile_rows) * feature_count +
                    feature] = 10.0f;
    }
    const pmpp_examples::AttentionResult extreme_reference =
        pmpp_examples::attention_causal_reference_cpu(
            extreme_query, extreme_key, value, rows, feature_count, scaling);
    const pmpp_examples::AttentionResult extreme_online =
        pmpp_examples::attention_causal_online_cpu(
            extreme_query, extreme_key, value, rows, feature_count, scaling,
            key_tile_rows);
    pmpp_examples::expect_near(
        extreme_online.output, extreme_reference.output, 2.0e-6f, 2.0e-5f,
        "CPU causal attention after a large second-tile maximum jump");
    pmpp_examples::expect_near(
        extreme_online.normalizer, extreme_reference.normalizer, 2.0e-6f,
        2.0e-5f, "CPU causal normalizer after maximum rescaling");

    const std::vector<float> zero(query.size(), 0.0f);
    const pmpp_examples::AttentionResult zero_reference =
        pmpp_examples::attention_causal_reference_cpu(
            zero, zero, value, rows, feature_count, scaling);
    const pmpp_examples::AttentionResult zero_online =
        pmpp_examples::attention_causal_online_cpu(
            zero, zero, value, rows, feature_count, scaling, key_tile_rows);
    pmpp_examples::expect_near(zero_online.output, zero_reference.output,
                               2.0e-6f, 2.0e-5f,
                               "CPU causal attention with zero logits");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool cpu_only = pmpp_examples::validate_cli(argc, argv);
        constexpr int rows = 96;
        const float scaling =
            1.0f / std::sqrt(static_cast<float>(feature_count));
        std::vector<float> query;
        std::vector<float> key;
        std::vector<float> value;
        make_inputs(query, key, value, rows);
        run_cpu_assertions(query, key, value, rows, scaling);
        const pmpp_examples::AttentionResult reference =
            pmpp_examples::attention_causal_reference_cpu(
                query, key, value, rows, feature_count, scaling);
        if (cpu_only || !has_cuda_device()) {
            report_cpu_only("ch20_ex01_online_attention");
            return EXIT_SUCCESS;
        }
        const GpuAttentionResult actual = run_attention_gpu(
            query, key, value, rows, feature_count, scaling);
        pmpp_examples::expect_near(actual.attention.output, reference.output,
                                   3.0e-5f, 7.0e-5f,
                                   "GPU causal tiled attention");
        pmpp_examples::expect_near(actual.attention.normalizer,
                                   reference.normalizer, 3.0e-5f, 7.0e-5f,
                                   "GPU causal tiled normalizer");
        pmpp_examples::expect(
            actual.warp_sums.size() == 2 &&
                std::fabs(actual.warp_sums[0] - 528.0f) < 1.0e-5f &&
                std::fabs(actual.warp_sums[1] - 1552.0f) < 1.0e-5f,
            "GPU WarpReduce sums");
        std::cout << "ch20_ex01_online_attention: GPU comparisons passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ch20_ex01_online_attention: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
