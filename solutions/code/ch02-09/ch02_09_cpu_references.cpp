#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void verify(const std::vector<float>& actual,
            const std::vector<float>& expected, float absolute_tolerance,
            float relative_tolerance, const std::string& label) {
    require(actual.size() == expected.size(), label + ": size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float tolerance =
            absolute_tolerance + relative_tolerance * std::abs(expected[i]);
        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error(label + ": mismatch at index " +
                                     std::to_string(i));
        }
    }
}

std::vector<float> vector_add_reference(const std::vector<float>& left,
                                        const std::vector<float>& right) {
    require(left.size() == right.size(), "vector sizes differ");
    std::vector<float> output(left.size());
    for (std::size_t i = 0; i < left.size(); ++i) {
        output[i] = left[i] + right[i];
    }
    return output;
}

void test_ch02_indexing() {
    constexpr std::size_t size = 1003;
    constexpr std::size_t block_size = 128;
    std::vector<float> left(size), right(size);
    for (std::size_t i = 0; i < size; ++i) {
        left[i] = static_cast<float>(i % 37) * 0.25f;
        right[i] = 3.0f - static_cast<float>(i % 29) * 0.125f;
    }
    const std::vector<float> expected =
        vector_add_reference(left, right);

    std::vector<float> one_per_thread(size, 0.0f);
    const std::size_t one_blocks = (size + block_size - 1) / block_size;
    for (std::size_t block = 0; block < one_blocks; ++block) {
        for (std::size_t thread = 0; thread < block_size; ++thread) {
            const std::size_t i = block * block_size + thread;
            if (i < size) {
                one_per_thread[i] = left[i] + right[i];
            }
        }
    }
    verify(one_per_thread, expected, 0.0f, 0.0f,
           "chapter 2 one-per-thread indexing");

    std::vector<float> adjacent(size, 0.0f);
    const std::size_t adjacent_tasks = (size + 1) / 2;
    for (std::size_t task = 0; task < adjacent_tasks; ++task) {
        const std::size_t first = 2 * task;
        adjacent[first] = left[first] + right[first];
        if (first + 1 < size) {
            adjacent[first + 1] = left[first + 1] + right[first + 1];
        }
    }
    verify(adjacent, expected, 0.0f, 0.0f,
           "chapter 2 adjacent-pair indexing");

    std::vector<float> segmented(size, 0.0f);
    const std::size_t segmented_blocks =
        (size + 2 * block_size - 1) / (2 * block_size);
    for (std::size_t block = 0; block < segmented_blocks; ++block) {
        for (std::size_t thread = 0; thread < block_size; ++thread) {
            const std::size_t first =
                2 * block * block_size + thread;
            if (first < size) {
                segmented[first] = left[first] + right[first];
            }
            const std::size_t second = first + block_size;
            if (second < size) {
                segmented[second] = left[second] + right[second];
            }
        }
    }
    verify(segmented, expected, 0.0f, 0.0f,
           "chapter 2 segmented-pair indexing");
}

float affine_reference(float value) {
    return 2.7f * value - 4.3f;
}

void test_ch02_runtime_affine() {
    std::vector<float> input(777), output(777);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 41) - 20) /
                   8.0f;
        output[i] = affine_reference(input[i]);
    }
    require(std::abs(output.front() + 11.05f) < 1.0e-5f,
            "chapter 2 affine known value failed");
    require(std::all_of(output.begin(), output.end(),
                        [](float value) { return std::isfinite(value); }),
            "chapter 2 affine produced non-finite output");
}

std::vector<float> matmul_reference(const std::vector<float>& left,
                                    const std::vector<float>& right, int n) {
    require(n > 0 &&
                left.size() == static_cast<std::size_t>(n) * n &&
                right.size() == static_cast<std::size_t>(n) * n,
            "invalid square matrices");
    std::vector<float> output(static_cast<std::size_t>(n) * n, 0.0f);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            double sum = 0.0;
            for (int inner = 0; inner < n; ++inner) {
                sum += static_cast<double>(left[row * n + inner]) *
                       right[inner * n + col];
            }
            output[row * n + col] = static_cast<float>(sum);
        }
    }
    return output;
}

void test_ch03_matmul() {
    constexpr int n = 19;
    const std::size_t elements = static_cast<std::size_t>(n) * n;
    std::vector<float> left(elements), identity(elements, 0.0f);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            left[row * n + col] =
                static_cast<float>((row * 7 + col * 3) % 23 - 11) / 8.0f;
        }
        identity[row * n + row] = 1.0f;
    }
    verify(matmul_reference(left, identity, n), left, 1.0e-6f, 1.0e-6f,
           "chapter 3 matrix multiply identity");
}

std::vector<float> matvec_reference(const std::vector<float>& matrix,
                                    const std::vector<float>& vector, int n) {
    require(n > 0 && vector.size() == static_cast<std::size_t>(n) &&
                matrix.size() == static_cast<std::size_t>(n) * n,
            "invalid matrix-vector dimensions");
    std::vector<float> output(n, 0.0f);
    for (int row = 0; row < n; ++row) {
        double sum = 0.0;
        for (int col = 0; col < n; ++col) {
            sum += static_cast<double>(matrix[row * n + col]) * vector[col];
        }
        output[row] = static_cast<float>(sum);
    }
    return output;
}

void test_ch03_matvec() {
    constexpr int n = 513;
    std::vector<float> matrix(static_cast<std::size_t>(n) * n, 0.0f);
    std::vector<float> vector(n);
    for (int i = 0; i < n; ++i) {
        matrix[i * n + i] = 1.5f;
        vector[i] = static_cast<float>((i % 17) - 8) / 16.0f;
    }
    std::vector<float> expected(n);
    std::transform(vector.begin(), vector.end(), expected.begin(),
                   [](float value) { return 1.5f * value; });
    verify(matvec_reference(matrix, vector, n), expected, 1.0e-6f, 1.0e-6f,
           "chapter 3 matrix-vector multiply");
}

std::vector<float> block_transpose_reference(const std::vector<float>& input,
                                             int width, int height,
                                             int block_width) {
    require(width > 0 && height > 0 && block_width > 0 &&
                input.size() == static_cast<std::size_t>(width) * height,
            "invalid block-transpose dimensions");
    std::vector<float> output = input;
    for (int base_y = 0; base_y < height; base_y += block_width) {
        for (int base_x = 0; base_x < width; base_x += block_width) {
            const int side =
                std::min({block_width, height - base_y, width - base_x});
            for (int y = 0; y < side; ++y) {
                for (int x = 0; x < side; ++x) {
                    output[(base_y + y) * width + base_x + x] =
                        input[(base_y + x) * width + base_x + y];
                }
            }
        }
    }
    return output;
}

void test_ch05_block_transpose() {
    constexpr int width = 22;
    constexpr int height = 14;
    std::vector<float> input(static_cast<std::size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            input[y * width + x] = static_cast<float>(100 * y + x);
        }
    }
    const std::vector<float> output =
        block_transpose_reference(input, width, height, 8);
    require(output[1] == 100.0f && output[width] == 1.0f,
            "chapter 5 first tile transpose failed");
    require(output[13 * width + 21] == input[13 * width + 21],
            "chapter 5 partial-tile boundary failed");
}

std::vector<float> corner_matmul_reference(
    const std::vector<float>& left,
    const std::vector<float>& right_column_major, int rows, int inner,
    int columns) {
    require(rows > 0 && inner > 0 && columns > 0 &&
                left.size() == static_cast<std::size_t>(rows) * inner &&
                right_column_major.size() ==
                    static_cast<std::size_t>(inner) * columns,
            "invalid corner-matmul dimensions");
    std::vector<float> output(static_cast<std::size_t>(rows) * columns,
                              0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < columns; ++col) {
            double sum = 0.0;
            for (int k = 0; k < inner; ++k) {
                sum += static_cast<double>(left[row * inner + k]) *
                       right_column_major[k + col * inner];
            }
            output[row * columns + col] = static_cast<float>(sum);
        }
    }
    return output;
}

void test_ch06_corner_matmul() {
    constexpr int rows = 37;
    constexpr int inner = 45;
    constexpr int columns = 35;
    std::vector<float> left(static_cast<std::size_t>(rows) * inner, 0.0f);
    std::vector<float> right_column(
        static_cast<std::size_t>(inner) * columns, 0.0f);
    for (int row = 0; row < rows; ++row) {
        left[row * inner + row % inner] = 2.0f;
    }
    for (int col = 0; col < columns; ++col) {
        for (int k = 0; k < inner; ++k) {
            right_column[k + col * inner] =
                static_cast<float>(100 * col + k);
        }
    }
    const std::vector<float> output = corner_matmul_reference(
        left, right_column, rows, inner, columns);
    require(output[0] == 0.0f &&
                output[(rows - 1) * columns + (columns - 1)] ==
                    2.0f * (100.0f * (columns - 1) + (rows - 1)),
            "chapter 6 column-major matrix multiply failed");
}

void test_ch06_float4_tail() {
    constexpr std::size_t size = 1027;
    std::vector<float> left(size), right(size);
    for (std::size_t i = 0; i < size; ++i) {
        left[i] = static_cast<float>(i % 31) * 0.5f;
        right[i] = 4.0f - static_cast<float>(i % 23) * 0.25f;
    }
    const std::vector<float> expected =
        vector_add_reference(left, right);
    std::vector<float> vectorized(size, 0.0f);
    const std::size_t vector_count = size / 4;
    for (std::size_t vector = 0; vector < vector_count; ++vector) {
        for (std::size_t lane = 0; lane < 4; ++lane) {
            const std::size_t i = 4 * vector + lane;
            vectorized[i] = left[i] + right[i];
        }
    }
    for (std::size_t i = 4 * vector_count; i < size; ++i) {
        vectorized[i] = left[i] + right[i];
    }
    require(size - 4 * vector_count == 3,
            "chapter 6 float4 fixture did not exercise a 3-element tail");
    verify(vectorized, expected, 0.0f, 0.0f,
           "chapter 6 float4 tail");
}

std::vector<float> convolution_3d_reference(
    const std::vector<float>& input, const std::vector<float>& filter,
    int radius, int width, int height, int depth) {
    const int size = 2 * radius + 1;
    require(radius >= 0 && width > 0 && height > 0 && depth > 0 &&
                input.size() ==
                    static_cast<std::size_t>(width) * height * depth &&
                filter.size() ==
                    static_cast<std::size_t>(size) * size * size,
            "invalid 3D convolution dimensions");
    std::vector<float> output(input.size(), 0.0f);
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double sum = 0.0;
                for (int fz = 0; fz < size; ++fz) {
                    for (int fy = 0; fy < size; ++fy) {
                        for (int fx = 0; fx < size; ++fx) {
                            const int ix = x - radius + fx;
                            const int iy = y - radius + fy;
                            const int iz = z - radius + fz;
                            if (ix >= 0 && ix < width && iy >= 0 &&
                                iy < height && iz >= 0 && iz < depth) {
                                sum += static_cast<double>(
                                           filter[(fz * size + fy) * size +
                                                  fx]) *
                                       input[(iz * height + iy) * width + ix];
                            }
                        }
                    }
                }
                output[(z * height + y) * width + x] =
                    static_cast<float>(sum);
            }
        }
    }
    return output;
}

std::vector<float> convolution_2d_reference(
    const std::vector<float>& input, const std::vector<float>& filter,
    int radius, int width, int height) {
    const int size = 2 * radius + 1;
    require(radius >= 0 && width > 0 && height > 0 &&
                input.size() == static_cast<std::size_t>(width) * height &&
                filter.size() == static_cast<std::size_t>(size) * size,
            "invalid 2D convolution dimensions");
    std::vector<float> output(input.size(), 0.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int fy = 0; fy < size; ++fy) {
                for (int fx = 0; fx < size; ++fx) {
                    const int ix = x - radius + fx;
                    const int iy = y - radius + fy;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        sum += static_cast<double>(filter[fy * size + fx]) *
                               input[iy * width + ix];
                    }
                }
            }
            output[y * width + x] = static_cast<float>(sum);
        }
    }
    return output;
}

void test_ch07_convolution() {
    constexpr int width3 = 13;
    constexpr int height3 = 11;
    constexpr int depth3 = 9;
    std::vector<float> input3(
        static_cast<std::size_t>(width3) * height3 * depth3, 1.0f);
    std::vector<float> filter3(27, 1.0f);
    const std::vector<float> output3 = convolution_3d_reference(
        input3, filter3, 1, width3, height3, depth3);
    require(output3.front() == 8.0f,
            "chapter 7 3D corner zero-padding failed");
    const int center3 =
        ((depth3 / 2) * height3 + height3 / 2) * width3 + width3 / 2;
    require(output3[center3] == 27.0f,
            "chapter 7 3D interior convolution failed");

    constexpr int width2 = 19;
    constexpr int height2 = 18;
    std::vector<float> input2(
        static_cast<std::size_t>(width2) * height2, 1.0f);
    std::vector<float> filter2(25, 1.0f);
    const std::vector<float> output2 =
        convolution_2d_reference(input2, filter2, 2, width2, height2);
    require(output2.front() == 9.0f,
            "chapter 7 2D corner zero-padding failed");
    require(output2[(height2 / 2) * width2 + width2 / 2] == 25.0f,
            "chapter 7 2D interior convolution failed");
}

}  // namespace

int main() {
    try {
        const std::vector<std::pair<const char*, std::function<void()>>> tests = {
            {"ch02 indexing", test_ch02_indexing},
            {"ch02 runtime/affine", test_ch02_runtime_affine},
            {"ch03 matrix multiply", test_ch03_matmul},
            {"ch03 matrix-vector multiply", test_ch03_matvec},
            {"ch05 block transpose", test_ch05_block_transpose},
            {"ch06 corner matrix multiply", test_ch06_corner_matmul},
            {"ch06 float4 tail", test_ch06_float4_tail},
            {"ch07 convolution", test_ch07_convolution},
        };
        for (const auto& test : tests) {
            test.second();
            std::cout << "PASS: " << test.first << '\n';
        }
        std::cout << "ch02_09_cpu_references: " << tests.size()
                  << " reference groups passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
