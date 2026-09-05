#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/cuda_check.cuh"

namespace {

// 书中假设 RGB 图像采用交错布局：
// [R0, G0, B0, R1, G1, B1, ...]。
constexpr int CHANNELS = 3;
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

// 与书中图 3.4 相同：一个二维线程负责一个输出像素。
// 参数顺序特意保留为 Pout、Pin、width、height。
__global__ void colorToGrayscaleConversion(unsigned char* Pout,
                                            const unsigned char* Pin,
                                            int width,
                                            int height) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 网格尺寸向上取整，因此最右侧和最下方会有一部分“多余线程”。
    if (col < width && row < height) {
        // 动态分配的二维数组按行主序展平：
        // (row, col) -> row * width + col。
        const int gray_offset = row * width + col;

        // 每个 RGB 像素占用三个连续字节。
        const int rgb_offset = gray_offset * CHANNELS;
        const unsigned char r = Pin[rgb_offset];
        const unsigned char g = Pin[rgb_offset + 1];
        const unsigned char b = Pin[rgb_offset + 2];

        // 使用第二章引入、第三章内核复用的 BT.601 luma 系数。
        // static_cast<unsigned char> 与书中的强制转换一样会截断小数部分。
        Pout[gray_offset] = static_cast<unsigned char>(
            0.299f * r + 0.587f * g + 0.114f * b);
    }
}

void grayscale_cpu(const std::vector<unsigned char>& rgb,
                   std::vector<unsigned char>& grayscale) {
    for (std::size_t gray_offset = 0;
         gray_offset < grayscale.size();
         ++gray_offset) {
        const std::size_t rgb_offset = gray_offset * CHANNELS;
        const unsigned char r = rgb[rgb_offset];
        const unsigned char g = rgb[rgb_offset + 1];
        const unsigned char b = rgb[rgb_offset + 2];
        grayscale[gray_offset] = static_cast<unsigned char>(
            0.299f * r + 0.587f * g + 0.114f * b);
    }
}

void initialize_rgb(std::vector<unsigned char>& rgb,
                    int width,
                    int height) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const std::size_t gray_offset =
                static_cast<std::size_t>(row) * width + col;
            const std::size_t rgb_offset = gray_offset * CHANNELS;

            // 确定性图案便于重复运行和定位错误；三个通道故意使用不同公式。
            // LL 后缀让中间计算使用 long long，避免极端尺寸下 int 溢出。
            rgb[rgb_offset] = static_cast<unsigned char>(
                (17LL * row + 3LL * col) % 256);
            rgb[rgb_offset + 1] = static_cast<unsigned char>(
                (5LL * row + 11LL * col + 40LL) % 256);
            rgb[rgb_offset + 2] = static_cast<unsigned char>(
                (13LL * row + 7LL * col + 80LL) % 256);
        }
    }
}

bool verify_with_one_level_tolerance(
    const std::vector<unsigned char>& gpu_result,
    const std::vector<unsigned char>& cpu_result,
    int width) {
    std::size_t mismatch_count = 0;
    std::size_t one_level_difference_count = 0;
    for (std::size_t i = 0; i < gpu_result.size(); ++i) {
        const int difference = std::abs(
            static_cast<int>(gpu_result[i]) -
            static_cast<int>(cpu_result[i]));
        if (difference == 1) {
            ++one_level_difference_count;
        } else if (difference > 1) {
            ++mismatch_count;
            if (mismatch_count <= 5) {
                const std::size_t row = i / static_cast<std::size_t>(width);
                const std::size_t col = i % static_cast<std::size_t>(width);
                std::cerr << "Mismatch at (row=" << row << ", col=" << col
                          << "): GPU=" << static_cast<int>(gpu_result[i])
                          << ", CPU=" << static_cast<int>(cpu_result[i])
                          << '\n';
            }
        }
    }

    if (mismatch_count == 0) {
        std::cout << "Verification PASSED: every pixel is within one "
                     "grayscale level of the CPU result.\n";
        if (one_level_difference_count > 0) {
            std::cout << "  " << one_level_difference_count
                      << " pixels differ by exactly 1 because CPU and GPU "
                         "may contract floating-point operations differently.\n";
        }
        return true;
    }

    std::cout << "Verification FAILED: " << mismatch_count
              << " pixels differ by more than one grayscale level.\n";
    return false;
}

void print_pixel(const std::vector<unsigned char>& rgb,
                 const std::vector<unsigned char>& grayscale,
                 int width,
                 int row,
                 int col) {
    const std::size_t gray_offset =
        static_cast<std::size_t>(row) * width + col;
    const std::size_t rgb_offset = gray_offset * CHANNELS;
    std::cout << "  (row=" << row << ", col=" << col << ")"
              << " gray_offset=" << gray_offset
              << ", rgb_offset=" << rgb_offset
              << ", RGB=(" << static_cast<int>(rgb[rgb_offset])
              << ", " << static_cast<int>(rgb[rgb_offset + 1])
              << ", " << static_cast<int>(rgb[rgb_offset + 2])
              << ") -> L=" << static_cast<int>(grayscale[gray_offset])
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    // 默认使用正文中的 76 列 × 62 行图像。
    int width = 76;
    int height = 62;
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

    // 内核使用 int 计算 row*width+col，所以先保证线性索引不会溢出。
    if (width > std::numeric_limits<int>::max() / height ||
        static_cast<long long>(width) * height >
            std::numeric_limits<int>::max() / CHANNELS) {
        std::cerr << "Image is too large for the kernel's int-based "
                     "RGB indexing.\n";
        return EXIT_FAILURE;
    }

    const std::size_t pixel_count =
        static_cast<std::size_t>(width) * height;
    const std::size_t rgb_byte_count = pixel_count * CHANNELS;

    std::vector<unsigned char> h_rgb(rgb_byte_count);
    std::vector<unsigned char> h_grayscale(pixel_count, 0);
    std::vector<unsigned char> h_expected(pixel_count, 0);
    initialize_rgb(h_rgb, width, height);
    grayscale_cpu(h_rgb, h_expected);

    unsigned char* d_rgb = nullptr;
    unsigned char* d_grayscale = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rgb), rgb_byte_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_grayscale), pixel_count));
    CUDA_CHECK(cudaMemcpy(d_rgb,
                          h_rgb.data(),
                          rgb_byte_count,
                          cudaMemcpyHostToDevice));

    const dim3 block(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    const dim3 grid((static_cast<unsigned int>(width) - 1) / block.x + 1,
                    (static_cast<unsigned int>(height) - 1) / block.y + 1,
                    1);

    const std::size_t launched_columns =
        static_cast<std::size_t>(grid.x) * block.x;
    const std::size_t launched_rows =
        static_cast<std::size_t>(grid.y) * block.y;
    const std::size_t launched_threads = launched_columns * launched_rows;

    std::cout << "RGB to grayscale\n"
              << "image (width x height) = " << width << " x " << height
              << '\n'
              << "block (x, y)          = (" << block.x << ", " << block.y
              << ")\n"
              << "grid (x, y)           = (" << grid.x << ", " << grid.y
              << ")\n"
              << "launched pixel threads= " << launched_threads << '\n'
              << "valid pixels          = " << pixel_count << '\n'
              << "extra guarded threads = "
              << launched_threads - pixel_count << "\n\n"
              << "Mapping used by every thread:\n"
              << "  col = blockIdx.x * blockDim.x + threadIdx.x\n"
              << "  row = blockIdx.y * blockDim.y + threadIdx.y\n"
              << "  gray_offset = row * width + col\n"
              << "  rgb_offset  = gray_offset * 3\n\n";

    colorToGrayscaleConversion<<<grid, block>>>(
        d_grayscale, d_rgb, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_grayscale.data(),
                          d_grayscale,
                          pixel_count,
                          cudaMemcpyDeviceToHost));

    const bool passed = verify_with_one_level_tolerance(
        h_grayscale, h_expected, width);

    std::cout << "\nSample pixels:\n";
    print_pixel(h_rgb, h_grayscale, width, 0, 0);
    if (height > 16) {
        // 正文中的追踪例：blockIdx=(0,1)、threadIdx=(0,0)
        // 映射到 (row,col)=(16,0)。对默认宽度 76：
        // gray_offset=1216，rgb_offset=3648。
        print_pixel(h_rgb, h_grayscale, width, 16, 0);
    }
    print_pixel(h_rgb, h_grayscale, width, height - 1, width - 1);

    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_grayscale));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
