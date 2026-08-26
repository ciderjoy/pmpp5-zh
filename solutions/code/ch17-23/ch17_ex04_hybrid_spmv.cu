#include "common/cuda_check.hpp"
#include "ch17_23_cpu_algorithms.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using pmpp_examples::HybridMatrix;
using pmpp_examples::Triplet;

__global__ void ell_spmv(const float* values, const int* columns,
                         const int* row_lengths, int rows, int width,
                         int cols, const float* x, float* y,
                         unsigned* bad) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const int length = row_lengths[row];
    if (length < 0 || length > width) {
        atomicExch(bad, 1U);
        return;
    }
    float sum = 0.0f;
    for (int item = 0; item < length; ++item) {
        const int position = item * rows + row;
        const int col = columns[position];
        if (col < 0 || col >= cols) {
            atomicExch(bad, 1U);
            return;
        }
        sum += values[position] * x[col];
    }
    y[row] = sum;
}

std::vector<float> hybrid_spmv_gpu(const HybridMatrix& matrix,
                                   const std::vector<float>& x) {
    const std::vector<float> validated =
        pmpp_examples::hybrid_spmv_cpu(matrix, x);
    (void)validated;
    if (matrix.rows == 0 || matrix.width == 0) {
        return pmpp_examples::hybrid_spmv_cpu(matrix, x);
    }
    device_buffer<float> device_values(matrix.ell_values.size());
    device_buffer<int> device_columns(matrix.ell_cols.size());
    device_buffer<int> device_lengths(matrix.row_lengths.size());
    device_buffer<float> device_x(x.size());
    device_buffer<float> device_y(static_cast<std::size_t>(matrix.rows));
    device_buffer<unsigned> device_bad(1);

    CUDA_CHECK(cudaMemcpy(device_values.get(), matrix.ell_values.data(),
                          matrix.ell_values.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_columns.get(), matrix.ell_cols.data(),
                          matrix.ell_cols.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_lengths.get(), matrix.row_lengths.data(),
                          matrix.row_lengths.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_x.get(), x.data(), x.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_bad.get(), 0, sizeof(unsigned)));

    constexpr int threads = 256;
    ell_spmv<<<(matrix.rows + threads - 1) / threads, threads>>>(
        device_values.get(), device_columns.get(), device_lengths.get(),
        matrix.rows, matrix.width, matrix.cols, device_x.get(), device_y.get(),
        device_bad.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned bad = 0;
    CUDA_CHECK(cudaMemcpy(&bad, device_bad.get(), sizeof(unsigned),
                          cudaMemcpyDeviceToHost));
    if (bad != 0) {
        throw std::runtime_error("GPU rejected HYB metadata");
    }
    std::vector<float> y(static_cast<std::size_t>(matrix.rows));
    CUDA_CHECK(cudaMemcpy(y.data(), device_y.get(), y.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (const Triplet& entry : matrix.overflow) {
        y[static_cast<std::size_t>(entry.row)] +=
            entry.value * x[static_cast<std::size_t>(entry.col)];
    }
    return y;
}

void run_cpu_assertions(const std::vector<Triplet>& entries,
                        const std::vector<float>& x) {
    const HybridMatrix hybrid = pmpp_examples::make_hybrid(entries, 4, 4, 1);
    const std::vector<float> expected =
        pmpp_examples::coo_spmv_cpu(entries, 4, 4, x);
    pmpp_examples::expect_near(
        pmpp_examples::hybrid_spmv_cpu(hybrid, x), expected, 0.0f, 0.0f,
        "CPU HYB SpMV");
    pmpp_examples::expect(hybrid.overflow.size() == 3,
                          "CPU HYB overflow partition");
    const HybridMatrix pure_coo = pmpp_examples::make_hybrid(entries, 4, 4, 0);
    pmpp_examples::expect_near(
        pmpp_examples::hybrid_spmv_cpu(pure_coo, x), expected, 0.0f, 0.0f,
        "CPU pure COO endpoint");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool cpu_only = pmpp_examples::validate_cli(argc, argv);
        const std::vector<Triplet> entries{
            {0, 0, 1.0f}, {0, 2, 7.0f}, {1, 2, 8.0f}, {2, 1, 4.0f},
            {2, 2, 3.0f}, {3, 0, 2.0f}, {3, 3, 1.0f}};
        const std::vector<float> x{2.0f, 3.0f, 5.0f, 7.0f};
        run_cpu_assertions(entries, x);
        const HybridMatrix hybrid = pmpp_examples::make_hybrid(entries, 4, 4, 1);
        const std::vector<float> reference =
            pmpp_examples::hybrid_spmv_cpu(hybrid, x);
        if (cpu_only || !has_cuda_device()) {
            report_cpu_only("ch17_ex04_hybrid_spmv");
            return EXIT_SUCCESS;
        }
        const std::vector<float> actual = hybrid_spmv_gpu(hybrid, x);
        pmpp_examples::expect_near(actual, reference, 1.0e-6f, 1.0e-6f,
                                   "GPU HYB SpMV");
        std::cout << "ch17_ex04_hybrid_spmv: GPU comparison passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ch17_ex04_hybrid_spmv: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
