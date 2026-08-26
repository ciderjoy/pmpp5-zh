#include "common/cuda_check.hpp"
#include "ch17_23_cpu_algorithms.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using pmpp_examples::JdsMatrix;
using pmpp_examples::Triplet;

__global__ void spmv_jds(const float* values, const int* col_idx,
                         const int* row_order, const int* iter_ptr, int rows,
                         int iterations, int cols, const float* x, float* y,
                         unsigned* bad) {
    const int sorted_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_row >= rows) {
        return;
    }
    float sum = 0.0f;
    for (int item = 0; item < iterations; ++item) {
        const int first = iter_ptr[item];
        const int last = iter_ptr[item + 1];
        const int active = last - first;
        if (active < 0 || active > rows) {
            atomicExch(bad, 1U);
            return;
        }
        if (sorted_row >= active) {
            break;
        }
        const int position = first + sorted_row;
        const int col = col_idx[position];
        if (col < 0 || col >= cols) {
            atomicExch(bad, 1U);
            return;
        }
        sum += values[position] * x[col];
    }
    const int output_row = row_order[sorted_row];
    if (output_row < 0 || output_row >= rows) {
        atomicExch(bad, 1U);
        return;
    }
    y[output_row] = sum;
}

std::vector<float> jds_spmv_gpu(const JdsMatrix& matrix,
                                const std::vector<float>& x) {
    pmpp_examples::validate_jds(matrix);
    if (x.size() != static_cast<std::size_t>(matrix.cols)) {
        throw std::invalid_argument("SpMV vector length mismatch");
    }
    if (matrix.rows == 0) {
        return {};
    }
    device_buffer<float> device_values(matrix.values.size());
    device_buffer<int> device_columns(matrix.col_idx.size());
    device_buffer<int> device_rows(matrix.row_order.size());
    device_buffer<int> device_ptr(matrix.iter_ptr.size());
    device_buffer<float> device_x(x.size());
    device_buffer<float> device_y(static_cast<std::size_t>(matrix.rows));
    device_buffer<unsigned> device_bad(1);

    if (!matrix.values.empty()) {
        CUDA_CHECK(cudaMemcpy(device_values.get(), matrix.values.data(),
                              matrix.values.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_columns.get(), matrix.col_idx.data(),
                              matrix.col_idx.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(device_rows.get(), matrix.row_order.data(),
                          matrix.row_order.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_ptr.get(), matrix.iter_ptr.data(),
                          matrix.iter_ptr.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_x.get(), x.data(), x.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_bad.get(), 0, sizeof(unsigned)));

    constexpr int threads = 256;
    const int iterations = static_cast<int>(matrix.iter_ptr.size()) - 1;
    spmv_jds<<<(matrix.rows + threads - 1) / threads, threads>>>(
        device_values.get(), device_columns.get(), device_rows.get(),
        device_ptr.get(), matrix.rows, iterations, matrix.cols, device_x.get(),
        device_y.get(), device_bad.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned bad = 0;
    CUDA_CHECK(cudaMemcpy(&bad, device_bad.get(), sizeof(unsigned),
                          cudaMemcpyDeviceToHost));
    if (bad != 0) {
        throw std::runtime_error("GPU rejected JDS metadata");
    }
    std::vector<float> result(static_cast<std::size_t>(matrix.rows));
    CUDA_CHECK(cudaMemcpy(result.data(), device_y.get(),
                          result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return result;
}

void run_cpu_assertions(const JdsMatrix& matrix,
                        const std::vector<float>& x) {
    pmpp_examples::validate_jds(matrix);
    pmpp_examples::expect(matrix.row_order ==
                              std::vector<int>({0, 2, 3, 1}),
                          "CPU JDS stable ordering");
    pmpp_examples::expect(matrix.iter_ptr == std::vector<int>({0, 4, 7}),
                          "CPU JDS diagonal pointers");
    pmpp_examples::expect_near(
        pmpp_examples::jds_spmv_cpu(matrix, x),
        std::vector<float>({37.0f, 40.0f, 27.0f, 11.0f}), 0.0f, 0.0f,
        "CPU JDS SpMV");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool cpu_only = pmpp_examples::validate_cli(argc, argv);
        const std::vector<Triplet> entries{
            {0, 0, 1.0f}, {0, 2, 7.0f}, {1, 2, 8.0f}, {2, 1, 4.0f},
            {2, 2, 3.0f}, {3, 0, 2.0f}, {3, 3, 1.0f}};
        const std::vector<float> x{2.0f, 3.0f, 5.0f, 7.0f};
        const JdsMatrix matrix = pmpp_examples::make_jds(entries, 4, 4);
        run_cpu_assertions(matrix, x);
        const std::vector<float> reference =
            pmpp_examples::jds_spmv_cpu(matrix, x);
        if (cpu_only || !has_cuda_device()) {
            report_cpu_only("ch17_ex05_jds_spmv");
            return EXIT_SUCCESS;
        }
        const std::vector<float> actual = jds_spmv_gpu(matrix, x);
        pmpp_examples::expect_near(actual, reference, 1.0e-6f, 1.0e-6f,
                                   "GPU JDS SpMV");
        std::cout << "ch17_ex05_jds_spmv: GPU comparison passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ch17_ex05_jds_spmv: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
