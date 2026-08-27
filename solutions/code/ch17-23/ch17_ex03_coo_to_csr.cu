#include "common/cuda_check.hpp"
#include "ch17_23_cpu_algorithms.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using pmpp_examples::CsrMatrix;
using pmpp_examples::Triplet;

__global__ void count_rows(const int* coo_row, const int* coo_col, int count,
                           int rows, int cols, unsigned* row_counts,
                           unsigned* bad) {
    const int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= count) {
        return;
    }
    const int row = coo_row[entry];
    const int col = coo_col[entry];
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        atomicExch(bad, 1U);
        return;
    }
    atomicAdd(&row_counts[row], 1U);
}

__global__ void scatter_csr(const int* coo_row, const int* coo_col,
                            const float* coo_value, int count, int rows,
                            int cols, unsigned* next, int* csr_col,
                            float* csr_value, unsigned* bad) {
    const int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= count) {
        return;
    }
    const int row = coo_row[entry];
    const int col = coo_col[entry];
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        atomicExch(bad, 1U);
        return;
    }
    const unsigned position = atomicAdd(&next[row], 1U);
    csr_col[position] = col;
    csr_value[position] = coo_value[entry];
}

CsrMatrix coo_to_csr_gpu(const std::vector<Triplet>& entries, int rows,
                         int cols) {
    pmpp_examples::validate_triplets(entries, rows, cols);
    if (entries.size() > static_cast<std::size_t>(
                             std::numeric_limits<int>::max())) {
        throw std::overflow_error("too many COO entries");
    }
    const int count = static_cast<int>(entries.size());
    std::vector<int> host_row(entries.size());
    std::vector<int> host_col(entries.size());
    std::vector<float> host_value(entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        host_row[i] = entries[i].row;
        host_col[i] = entries[i].col;
        host_value[i] = entries[i].value;
    }

    device_buffer<int> device_row(entries.size());
    device_buffer<int> device_col(entries.size());
    device_buffer<float> device_value(entries.size());
    device_buffer<unsigned> device_counts(static_cast<std::size_t>(rows));
    device_buffer<unsigned> device_row_ptr(static_cast<std::size_t>(rows) + 1);
    device_buffer<unsigned> device_next(static_cast<std::size_t>(rows));
    device_buffer<int> device_csr_col(entries.size());
    device_buffer<float> device_csr_value(entries.size());
    device_buffer<unsigned> device_bad(1);

    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(device_row.get(), host_row.data(),
                              entries.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_col.get(), host_col.data(),
                              entries.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_value.get(), host_value.data(),
                              entries.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    if (rows > 0) {
        CUDA_CHECK(cudaMemset(device_counts.get(), 0,
                              static_cast<std::size_t>(rows) *
                                  sizeof(unsigned)));
    }
    CUDA_CHECK(cudaMemset(device_bad.get(), 0, sizeof(unsigned)));

    constexpr int threads = 256;
    if (count > 0) {
        count_rows<<<(count + threads - 1) / threads, threads>>>(
            device_row.get(), device_col.get(), count, rows, cols,
            device_counts.get(), device_bad.get());
        CUDA_CHECK(cudaGetLastError());
    }
    if (rows > 0) {
        std::size_t temporary_bytes = 0;
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr, temporary_bytes, device_counts.get(),
            device_row_ptr.get(), rows));
        device_buffer<unsigned char> temporary_storage(temporary_bytes);
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            temporary_storage.get(), temporary_bytes, device_counts.get(),
            device_row_ptr.get(), rows));
    }
    const unsigned nonzeros = static_cast<unsigned>(count);
    CUDA_CHECK(cudaMemcpy(device_row_ptr.get() + rows, &nonzeros,
                          sizeof(nonzeros), cudaMemcpyHostToDevice));
    if (rows > 0) {
        CUDA_CHECK(cudaMemcpy(device_next.get(), device_row_ptr.get(),
                              static_cast<std::size_t>(rows) *
                                  sizeof(unsigned),
                              cudaMemcpyDeviceToDevice));
    }
    if (count > 0) {
        scatter_csr<<<(count + threads - 1) / threads, threads>>>(
            device_row.get(), device_col.get(), device_value.get(), count,
            rows, cols, device_next.get(), device_csr_col.get(),
            device_csr_value.get(), device_bad.get());
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned bad = 0;
    CUDA_CHECK(cudaMemcpy(&bad, device_bad.get(), sizeof(unsigned),
                          cudaMemcpyDeviceToHost));
    if (bad != 0) {
        throw std::out_of_range("GPU rejected a COO coordinate");
    }
    CsrMatrix result;
    result.rows = rows;
    result.cols = cols;
    result.row_ptr.resize(static_cast<std::size_t>(rows) + 1);
    result.col_idx.resize(entries.size());
    result.values.resize(entries.size());
    CUDA_CHECK(cudaMemcpy(result.row_ptr.data(), device_row_ptr.get(),
                          result.row_ptr.size() * sizeof(unsigned),
                          cudaMemcpyDeviceToHost));
    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(result.col_idx.data(), device_csr_col.get(),
                              entries.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(result.values.data(), device_csr_value.get(),
                              entries.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    return result;
}

void run_cpu_assertions(const std::vector<Triplet>& entries) {
    const CsrMatrix reference = pmpp_examples::coo_to_csr_cpu(entries, 4, 4);
    pmpp_examples::expect(
        reference.row_ptr == std::vector<unsigned>({0, 2, 3, 5, 7}),
        "CPU COO-to-CSR row pointers");
    bool rejected = false;
    try {
        pmpp_examples::coo_to_csr_cpu({{4, 0, 1.0f}}, 4, 4);
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    pmpp_examples::expect(rejected, "CPU COO validation");
    const CsrMatrix zero_rows = pmpp_examples::coo_to_csr_cpu({}, 0, 4);
    pmpp_examples::expect(zero_rows.row_ptr == std::vector<unsigned>({0}),
                          "CPU zero-row CSR sentinel");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool cpu_only = pmpp_examples::validate_cli(argc, argv);
        const std::vector<Triplet> entries{
            {2, 2, 3.0f}, {0, 2, 7.0f}, {3, 3, 1.0f}, {1, 2, 8.0f},
            {0, 0, 1.0f}, {3, 0, 2.0f}, {2, 1, 4.0f}};
        run_cpu_assertions(entries);
        const CsrMatrix reference =
            pmpp_examples::coo_to_csr_cpu(entries, 4, 4);
        if (cpu_only || !has_cuda_device()) {
            report_cpu_only("ch17_ex03_coo_to_csr");
            return EXIT_SUCCESS;
        }
        const CsrMatrix actual = coo_to_csr_gpu(entries, 4, 4);
        pmpp_examples::expect(actual.row_ptr == reference.row_ptr,
                              "GPU CSR row pointers");
        pmpp_examples::expect(
            pmpp_examples::canonical_entries(actual) ==
                pmpp_examples::canonical_entries(reference),
            "GPU COO-to-CSR entries");
        const CsrMatrix zero_rows = coo_to_csr_gpu({}, 0, 4);
        pmpp_examples::expect(zero_rows.row_ptr ==
                                  std::vector<unsigned>({0}),
                              "GPU zero-row CSR sentinel");
        std::cout << "ch17_ex03_coo_to_csr: GPU comparison passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ch17_ex03_coo_to_csr: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
