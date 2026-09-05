#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/cuda_check.cuh"

namespace {

constexpr unsigned int THREADS_PER_BLOCK = 256;
constexpr float TOLERANCE = 1.0e-6f;

std::size_t parse_element_count(int argc, char** argv) {
    // 默认值 2^20 足以启动许多线程，也不会占用太多显存。
    if (argc == 1) {
        return 1U << 20;
    }
    if (argc != 2) {
        throw std::invalid_argument("expected zero or one argument");
    }

    std::size_t parsed_characters = 0;
    const std::string text = argv[1];
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument("N must be a positive integer");
    }
    const unsigned long long value = std::stoull(text, &parsed_characters);
    if (parsed_characters != text.size() || value == 0) {
        throw std::invalid_argument("N must be a positive integer");
    }
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::out_of_range("N is too large for std::size_t");
    }
    return static_cast<std::size_t>(value);
}

// CPU 基准版本：一个循环依次完成 N 次加法。
// 后面的 CUDA 内核会把每次循环迭代映射给一个 GPU 线程。
void vec_add_cpu(const std::vector<float>& a,
                 const std::vector<float>& b,
                 std::vector<float>& c) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] + b[i];
    }
}

// C[i] = A[i] + B[i]
//
// 一个 block 中有 blockDim.x 个线程。
// 每个线程通过 blockIdx.x、blockDim.x 和 threadIdx.x 算出自己的全局下标。
__global__ void vecAddKernel(const float* a,
                             const float* b,
                             float* c,
                             std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // 最后一个 block 往往不能刚好填满，因此必须做边界检查。
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

bool verify_result(const std::vector<float>& gpu_result,
                   const std::vector<float>& cpu_result) {
    std::size_t error_count = 0;
    float max_error = 0.0f;

    for (std::size_t i = 0; i < gpu_result.size(); ++i) {
        const bool finite = std::isfinite(gpu_result[i]) &&
                            std::isfinite(cpu_result[i]);
        const float error = finite
                                ? std::fabs(gpu_result[i] - cpu_result[i])
                                : std::numeric_limits<float>::infinity();
        max_error = std::max(max_error, error);

        if (!finite || error > TOLERANCE) {
            ++error_count;
            if (error_count <= 5) {
                std::cerr << "Mismatch at i=" << i
                          << ": GPU=" << gpu_result[i]
                          << ", CPU=" << cpu_result[i]
                          << ", error=" << error << '\n';
            }
        }
    }

    if (error_count == 0) {
        std::cout << "Verification PASSED, max error = " << max_error << '\n';
        return true;
    }

    std::cout << "Verification FAILED, errors = " << error_count
              << ", max error = " << max_error << '\n';
    return false;
}

void print_samples(const std::vector<float>& a,
                   const std::vector<float>& b,
                   const std::vector<float>& c) {
    const std::size_t shown = std::min<std::size_t>(5, c.size());
    std::cout << "\nFirst " << shown << " results:\n";
    for (std::size_t i = 0; i < shown; ++i) {
        std::cout << "  C[" << i << "] = " << a[i] << " + " << b[i]
                  << " = " << c[i] << '\n';
    }
    if (c.size() > shown) {
        const std::size_t i = c.size() - 1;
        std::cout << "  ...\n"
                  << "  C[" << i << "] = " << a[i] << " + " << b[i]
                  << " = " << c[i] << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t n = 0;
    try {
        n = parse_element_count(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Usage: " << argv[0] << " [positive_element_count]\n"
                  << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    if (n > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        std::cerr << "N is too large: byte count would overflow\n";
        return EXIT_FAILURE;
    }
    const std::size_t bytes = n * sizeof(float);

    // 避免写成 (n + block_size - 1) / block_size 时的加法溢出。
    const std::size_t block_count =
        (n - 1) / THREADS_PER_BLOCK + 1;
    // CUDA 文本中给出的 gridDim.x 上限为 2^31-1。
    if (block_count >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        std::cerr << "N is too large for this one-dimensional launch\n";
        return EXIT_FAILURE;
    }
    const unsigned int blocks = static_cast<unsigned int>(block_count);

    // -------------------------
    // 1. 准备 Host 数据和 CPU 基准答案
    // -------------------------
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n, 0.0f);
    std::vector<float> h_expected(n);

    for (std::size_t i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i % 1000) * 0.001f;
        h_b[i] = static_cast<float>((i * 3) % 1000) * 0.001f;
    }
    vec_add_cpu(h_a, h_b, h_expected);

    // -------------------------
    // 2. 在 Device（GPU）上分配内存
    // -------------------------
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), bytes));

    // Host -> Device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // -------------------------
    // 3. 配置并启动 kernel
    // -------------------------
    const std::size_t launched_threads =
        static_cast<std::size_t>(blocks) * THREADS_PER_BLOCK;

    std::cout << "Vector addition: C = A + B\n"
              << "N                    = " << n << '\n'
              << "threads per block    = " << THREADS_PER_BLOCK << '\n'
              << "blocks               = " << blocks << '\n'
              << "launched threads     = " << launched_threads << '\n'
              << "extra guarded threads= " << launched_threads - n << "\n\n"
              << "Thread-to-data mapping:\n"
              << "  i = blockIdx.x * blockDim.x + threadIdx.x\n"
              << "  for block 1, thread 0: i = 1 * "
              << THREADS_PER_BLOCK << " + 0 = " << THREADS_PER_BLOCK
              << "\n\n";

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    vecAddKernel<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n);

    // 内核启动是异步的：
    // 1) cudaGetLastError 检查启动配置等同步错误；
    // 2) cudaEventSynchronize 等待 GPU，并暴露执行期间的异步错误。
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Device -> Host
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // -------------------------
    // 4. 和 CPU 基准答案逐元素对比
    // -------------------------
    const bool passed = verify_result(h_c, h_expected);
    print_samples(h_a, h_b, h_c);

    // 这是一次内核执行的教学性估算，不是严谨 benchmark：
    // 它不包含 PCIe 数据传输时间，也没有预热和重复采样。
    std::cout << "\nKernel time = " << elapsed_ms << " ms\n";
    if (elapsed_ms > 0.0f) {
        // 每个元素读取 A、B 并写入 C，共移动约 3*sizeof(float) 字节。
        const double gigabytes = 3.0 * static_cast<double>(bytes) / 1.0e9;
        const double effective_bandwidth =
            gigabytes / (static_cast<double>(elapsed_ms) / 1.0e3);
        std::cout << "Approx. kernel effective bandwidth = "
                  << effective_bandwidth << " GB/s\n";
    }

    // -------------------------
    // 5. 释放资源
    // -------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
