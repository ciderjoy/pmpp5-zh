#include <cuda_runtime.h>

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

// 书中的 BLUR_SIZE 表示“半径”，不是窗口边长。
// BLUR_SIZE=1 对应 (2*1+1) × (2*1+1) = 3×3 窗口。
constexpr int BLUR_SIZE = 1;
constexpr unsigned int BLOCK_WIDTH = 16;

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

// 与书中图 3.8 相同：一个线程负责一个输出像素。
// 和灰度化不同，该线程会读取目标像素周围的多个输入像素。
__global__ void blurKernel(const unsigned char* in,
                           unsigned char* out,
                           int width,
                           int height) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pixel_sum = 0;
        int valid_pixel_count = 0;

        for (int blur_row = -BLUR_SIZE;
             blur_row <= BLUR_SIZE;
             ++blur_row) {
            for (int blur_col = -BLUR_SIZE;
                 blur_col <= BLUR_SIZE;
                 ++blur_col) {
                const int current_row = row + blur_row;
                const int current_col = col + blur_col;

                // 边缘像素的邻域有一部分落在图像外，不能访问。
                if (current_row >= 0 && current_row < height &&
                    current_col >= 0 && current_col < width) {
                    pixel_sum +=
                        in[current_row * width + current_col];
                    ++valid_pixel_count;
                }
            }
        }

        // 角点、边缘和内部像素使用的有效邻居数量不同。
        out[row * width + col] = static_cast<unsigned char>(
            static_cast<float>(pixel_sum) / valid_pixel_count);
    }
}

void blur_cpu(const std::vector<unsigned char>& input,
              std::vector<unsigned char>& output,
              int width,
              int height) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int pixel_sum = 0;
            int valid_pixel_count = 0;

            for (int blur_row = -BLUR_SIZE;
                 blur_row <= BLUR_SIZE;
                 ++blur_row) {
                for (int blur_col = -BLUR_SIZE;
                     blur_col <= BLUR_SIZE;
                     ++blur_col) {
                    const int current_row = row + blur_row;
                    const int current_col = col + blur_col;
                    if (current_row >= 0 && current_row < height &&
                        current_col >= 0 && current_col < width) {
                        pixel_sum +=
                            input[current_row * width + current_col];
                        ++valid_pixel_count;
                    }
                }
            }

            output[row * width + col] = static_cast<unsigned char>(
                static_cast<float>(pixel_sum) / valid_pixel_count);
        }
    }
}

void initialize_image(std::vector<unsigned char>& image,
                      int width,
                      int height) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            // 平滑背景上放一个亮点，运行后可以直观看到亮度向邻域扩散。
            const bool is_center =
                row == height / 2 && col == width / 2;
            image[row * width + col] =
                is_center
                    ? 255
                    : static_cast<unsigned char>(
                          20 + (31LL * row + 17LL * col) % 160);
        }
    }
}

int valid_neighbor_count(int row, int col, int width, int height) {
    int count = 0;
    for (int blur_row = -BLUR_SIZE; blur_row <= BLUR_SIZE; ++blur_row) {
        for (int blur_col = -BLUR_SIZE; blur_col <= BLUR_SIZE; ++blur_col) {
            const int current_row = row + blur_row;
            const int current_col = col + blur_col;
            if (current_row >= 0 && current_row < height &&
                current_col >= 0 && current_col < width) {
                ++count;
            }
        }
    }
    return count;
}

void print_image(const std::vector<unsigned char>& image,
                 int width,
                 int height,
                 const char* title) {
    std::cout << title << '\n';
    for (int row = 0; row < height; ++row) {
        std::cout << "  ";
        for (int col = 0; col < width; ++col) {
            std::cout << std::setw(4)
                      << static_cast<int>(image[row * width + col]);
        }
        std::cout << '\n';
    }
}

bool verify_exact(const std::vector<unsigned char>& gpu_result,
                  const std::vector<unsigned char>& cpu_result,
                  int width) {
    std::size_t mismatch_count = 0;
    for (std::size_t i = 0; i < gpu_result.size(); ++i) {
        if (gpu_result[i] != cpu_result[i]) {
            ++mismatch_count;
            if (mismatch_count <= 5) {
                std::cerr << "Mismatch at (row=" << i / width
                          << ", col=" << i % width
                          << "): GPU=" << static_cast<int>(gpu_result[i])
                          << ", CPU=" << static_cast<int>(cpu_result[i])
                          << '\n';
            }
        }
    }

    if (mismatch_count == 0) {
        std::cout << "Verification PASSED: every blurred byte matches.\n";
        return true;
    }

    std::cout << "Verification FAILED: " << mismatch_count
              << " pixels differ.\n";
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    // 小尺寸默认值便于直接在终端观察输入和输出矩阵。
    int width = 8;
    int height = 6;
    try {
        if (argc == 3) {
            width = parse_positive_int(argv[1], "width");
            height = parse_positive_int(argv[2], "height");
        } else if (argc != 1) {
            throw std::invalid_argument("expected either zero or two arguments");
        }
    } catch (const std::exception& error) {
        std::cerr << "Usage: " << argv[0] << " [width height]\n"
                  << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    if (width > std::numeric_limits<int>::max() / height) {
        std::cerr << "Image is too large for int-based indexing.\n";
        return EXIT_FAILURE;
    }
    const std::size_t pixel_count =
        static_cast<std::size_t>(width) * height;

    std::vector<unsigned char> h_input(pixel_count);
    std::vector<unsigned char> h_output(pixel_count, 0);
    std::vector<unsigned char> h_expected(pixel_count, 0);
    initialize_image(h_input, width, height);
    blur_cpu(h_input, h_expected, width, height);

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), pixel_count));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), pixel_count));
    CUDA_CHECK(cudaMemcpy(d_input,
                          h_input.data(),
                          pixel_count,
                          cudaMemcpyHostToDevice));

    const dim3 block(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    const dim3 grid((static_cast<unsigned int>(width) - 1) / block.x + 1,
                    (static_cast<unsigned int>(height) - 1) / block.y + 1,
                    1);

    std::cout << "Box blur\n"
              << "image (width x height) = " << width << " x " << height
              << '\n'
              << "BLUR_SIZE (radius)     = " << BLUR_SIZE << '\n'
              << "window width          = " << 2 * BLUR_SIZE + 1 << '\n'
              << "block (x, y)          = (" << block.x << ", " << block.y
              << ")\n"
              << "grid (x, y)           = (" << grid.x << ", " << grid.y
              << ")\n\n"
              << "Valid samples in a 3x3 neighborhood:\n"
              << "  top-left corner = "
              << valid_neighbor_count(0, 0, width, height) << '\n'
              << "  top edge        = "
              << valid_neighbor_count(0, width / 2, width, height) << '\n'
              << "  interior        = "
              << valid_neighbor_count(
                     height / 2, width / 2, width, height)
              << "\n\n";

    blurKernel<<<grid, block>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(),
                          d_output,
                          pixel_count,
                          cudaMemcpyDeviceToHost));

    const bool passed = verify_exact(h_output, h_expected, width);

    // 大图不全部打印，避免终端被输出淹没。
    if (pixel_count <= 100) {
        std::cout << '\n';
        print_image(h_input, width, height, "Input image:");
        std::cout << '\n';
        print_image(h_output, width, height, "Blurred image:");
    } else {
        const std::size_t center =
            static_cast<std::size_t>(height / 2) * width + width / 2;
        std::cout << "Center pixel: input="
                  << static_cast<int>(h_input[center])
                  << ", output=" << static_cast<int>(h_output[center])
                  << '\n';
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
