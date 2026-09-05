#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/cuda_check.cuh"

namespace {

constexpr int DEFAULT_WIDTH = 4;
constexpr int DEFAULT_BLOCK_WIDTH = 2;
constexpr float ABSOLUTE_TOLERANCE = 1.0e-4f;
constexpr float RELATIVE_TOLERANCE = 1.0e-5f;

int parse_positive_int(const char* text, const char* name) {
    const std::string value_text = text;
    if (value_text.empty() || value_text.front() == '-') {
        throw std::invalid_argument(std::string(name) +
                                    " must be a positive integer");
    }

    std::size_t parsed_characters = 0;
    const long long value = std::stoll(value_text, &parsed_characters);
    if (parsed_characters != value_text.size() || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(name) +
                                    " must be a positive int");
    }
    return static_cast<int>(value);
}

// 与书中图 3.11 相同的朴素方阵乘法：
// 一个二维线程计算输出矩阵 P 的一个元素 P[row, col]。
__global__ void MatrixMulKernel(const float* M,
                                const float* N,
                                float* P,
                                int Width) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Width && col < Width) {
        float Pvalue = 0.0f;
        for (int k = 0; k < Width; ++k) {
            // M 的第 row 行与 N 的第 col 列做点积。
            // 两个矩阵都按行主序存储：
            // M[row, k] -> M[row*Width+k]
            // N[k, col] -> N[k*Width+col]
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}

void matrix_multiply_cpu(const std::vector<float>& m,
                         const std::vector<float>& n,
                         std::vector<float>& p,
                         int width) {
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            float value = 0.0f;
            for (int k = 0; k < width; ++k) {
                value += m[row * width + k] * n[k * width + col];
            }
            p[row * width + col] = value;
        }
    }
}

void initialize_matrices(std::vector<float>& m,
                         std::vector<float>& n,
                         int width) {
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            const int index = row * width + col;
            m[index] = static_cast<float>((index % 9) + 1);
            n[index] = static_cast<float>(
                ((3 * row + 2 * col + 1) % 7) - 3);
        }
    }
}

void print_matrix(const std::vector<float>& matrix,
                  int width,
                  const char* name) {
    std::cout << name << ":\n";
    for (int row = 0; row < width; ++row) {
        std::cout << "  ";
        for (int col = 0; col < width; ++col) {
            std::cout << std::setw(9) << std::fixed << std::setprecision(1)
                      << matrix[row * width + col];
        }
        std::cout << '\n';
    }
}

void trace_first_dot_product(const std::vector<float>& m,
                             const std::vector<float>& n,
                             int width) {
    float sum = 0.0f;
    std::cout << "How thread (row=0, col=0) computes P[0,0]:\n";
    for (int k = 0; k < width; ++k) {
        const float product = m[k] * n[k * width];
        sum += product;
        std::cout << "  k=" << k
                  << ": M[0*" << width << '+' << k << "] * N["
                  << k << '*' << width << "+0] = "
                  << m[k] << " * " << n[k * width]
                  << " = " << product
                  << ", partial sum = " << sum << '\n';
    }
}

bool verify_result(const std::vector<float>& gpu_result,
                   const std::vector<float>& cpu_result,
                   int width) {
    std::size_t mismatch_count = 0;
    float max_error = 0.0f;

    for (std::size_t i = 0; i < gpu_result.size(); ++i) {
        const float expected = cpu_result[i];
        const float actual = gpu_result[i];
        const bool finite = std::isfinite(expected) && std::isfinite(actual);
        const float error = finite
                                ? std::fabs(actual - expected)
                                : std::numeric_limits<float>::infinity();
        const float allowed_error =
            ABSOLUTE_TOLERANCE +
            RELATIVE_TOLERANCE * std::fabs(expected);
        max_error = std::max(max_error, error);

        if (!finite || error > allowed_error) {
            ++mismatch_count;
            if (mismatch_count <= 5) {
                std::cerr << "Mismatch at (row=" << i / width
                          << ", col=" << i % width
                          << "): GPU=" << actual
                          << ", CPU=" << expected
                          << ", error=" << error << '\n';
            }
        }
    }

    if (mismatch_count == 0) {
        std::cout << "Verification PASSED, max error = "
                  << max_error << '\n';
        return true;
    }

    std::cout << "Verification FAILED: " << mismatch_count
              << " elements differ, max error = " << max_error << '\n';
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    // 默认值刻意复现正文图 3.12：4×4 方阵、2×2 线程块。
    int width = DEFAULT_WIDTH;
    int block_width = DEFAULT_BLOCK_WIDTH;
    try {
        if (argc >= 2) {
            width = parse_positive_int(argv[1], "width");
        }
        if (argc == 3) {
            block_width = parse_positive_int(argv[2], "block_width");
        } else if (argc > 3) {
            throw std::invalid_argument("too many arguments");
        }
    } catch (const std::exception& error) {
        std::cerr << "Usage: " << argv[0] << " [width [block_width]]\n"
                  << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    if (block_width > 32 ||
        block_width * block_width > 1024) {
        std::cerr << "block_width must create at most 1024 threads; "
                     "for a square block use block_width <= 32.\n";
        return EXIT_FAILURE;
    }
    if (width > 2048) {
        std::cerr << "This teaching implementation is intentionally limited "
                     "to width <= 2048 because it performs O(width^3) work.\n";
        return EXIT_FAILURE;
    }

    const std::size_t element_count =
        static_cast<std::size_t>(width) * width;
    const std::size_t bytes = element_count * sizeof(float);

    std::vector<float> h_m(element_count);
    std::vector<float> h_n(element_count);
    std::vector<float> h_p(element_count, 0.0f);
    std::vector<float> h_expected(element_count, 0.0f);
    initialize_matrices(h_m, h_n, width);
    matrix_multiply_cpu(h_m, h_n, h_expected, width);

    float* d_m = nullptr;
    float* d_n = nullptr;
    float* d_p = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_m), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_n), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_p), bytes));
    CUDA_CHECK(
        cudaMemcpy(d_m, h_m.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_n, h_n.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(block_width, block_width, 1);
    const dim3 grid(
        (static_cast<unsigned int>(width) - 1) / block.x + 1,
        (static_cast<unsigned int>(width) - 1) / block.y + 1,
        1);
    const std::size_t launched_threads =
        static_cast<std::size_t>(grid.x) * block.x *
        static_cast<std::size_t>(grid.y) * block.y;

    std::cout << "Naive square matrix multiplication: P = M * N\n"
              << "matrix width          = " << width << '\n'
              << "block (x, y)          = (" << block.x << ", " << block.y
              << ")\n"
              << "grid (x, y)           = (" << grid.x << ", " << grid.y
              << ")\n"
              << "launched threads      = " << launched_threads << '\n'
              << "output elements       = " << element_count << '\n'
              << "extra guarded threads = "
              << launched_threads - element_count << "\n\n"
              << "Each valid thread computes one output element:\n"
              << "  row = blockIdx.y * blockDim.y + threadIdx.y\n"
              << "  col = blockIdx.x * blockDim.x + threadIdx.x\n"
              << "  P[row,col] = dot(M[row,:], N[:,col])\n\n";

    MatrixMulKernel<<<grid, block>>>(d_m, d_n, d_p, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(h_p.data(), d_p, bytes, cudaMemcpyDeviceToHost));
    const bool passed = verify_result(h_p, h_expected, width);

    if (width <= 8) {
        std::cout << '\n';
        print_matrix(h_m, width, "M");
        std::cout << '\n';
        print_matrix(h_n, width, "N");
        std::cout << '\n';
        trace_first_dot_product(h_m, h_n, width);
        std::cout << '\n';
        print_matrix(h_p, width, "P = M * N");
    }

    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_n));
    CUDA_CHECK(cudaFree(d_p));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

