#include "common/cuda_check.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int radius3 = 1;
constexpr int kernel3 = 2 * radius3 + 1;
constexpr int input_tile3 = 8;
constexpr int output_tile3 = input_tile3 - 2 * radius3;
__constant__ float filter3_constant_memory[kernel3 * kernel3 * kernel3];
__constant__ float filter3_tiled_memory[kernel3 * kernel3 * kernel3];

constexpr int radius2 = 2;
constexpr int kernel2 = 2 * radius2 + 1;
constexpr int output_tile2 = 16;
constexpr int input_tile2 = output_tile2 + 2 * radius2;
__constant__ float filter2_output_memory[kernel2 * kernel2];

__global__ void convolution_3d_basic(const float* input, const float* filter,
                                     float* output, int radius, int width,
                                     int height, int depth) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) {
        return;
    }
    const int size = 2 * radius + 1;
    float sum = 0.0f;
    for (int fz = 0; fz < size; ++fz) {
        for (int fy = 0; fy < size; ++fy) {
            for (int fx = 0; fx < size; ++fx) {
                const int ix = x - radius + fx;
                const int iy = y - radius + fy;
                const int iz = z - radius + fz;
                if (ix >= 0 && ix < width && iy >= 0 && iy < height &&
                    iz >= 0 && iz < depth) {
                    const int input_index =
                        (iz * height + iy) * width + ix;
                    const int filter_index =
                        (fz * size + fy) * size + fx;
                    sum += filter[filter_index] * input[input_index];
                }
            }
        }
    }
    output[(z * height + y) * width + x] = sum;
}

__global__ void convolution_3d_constant(const float* input, float* output,
                                        int width, int height, int depth) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) {
        return;
    }
    float sum = 0.0f;
    for (int fz = 0; fz < kernel3; ++fz) {
        for (int fy = 0; fy < kernel3; ++fy) {
            for (int fx = 0; fx < kernel3; ++fx) {
                const int ix = x - radius3 + fx;
                const int iy = y - radius3 + fy;
                const int iz = z - radius3 + fz;
                if (ix >= 0 && ix < width && iy >= 0 && iy < height &&
                    iz >= 0 && iz < depth) {
                    sum += filter3_constant_memory
                               [(fz * kernel3 + fy) * kernel3 + fx] *
                           input[(iz * height + iy) * width + ix];
                }
            }
        }
    }
    output[(z * height + y) * width + x] = sum;
}

__global__ void convolution_3d_tiled(const float* input, float* output,
                                     int width, int height, int depth) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int x = blockIdx.x * output_tile3 + tx - radius3;
    const int y = blockIdx.y * output_tile3 + ty - radius3;
    const int z = blockIdx.z * output_tile3 + tz - radius3;
    __shared__ float tile[input_tile3][input_tile3][input_tile3];
    tile[tz][ty][tx] = x >= 0 && x < width && y >= 0 && y < height &&
                               z >= 0 && z < depth
                           ? input[(z * height + y) * width + x]
                           : 0.0f;
    __syncthreads();

    const int local_x = tx - radius3;
    const int local_y = ty - radius3;
    const int local_z = tz - radius3;
    if (local_x >= 0 && local_x < output_tile3 && local_y >= 0 &&
        local_y < output_tile3 && local_z >= 0 &&
        local_z < output_tile3 && x >= 0 && x < width && y >= 0 &&
        y < height && z >= 0 && z < depth) {
        float sum = 0.0f;
        for (int fz = 0; fz < kernel3; ++fz) {
            for (int fy = 0; fy < kernel3; ++fy) {
                for (int fx = 0; fx < kernel3; ++fx) {
                    sum += filter3_tiled_memory
                               [(fz * kernel3 + fy) * kernel3 + fx] *
                           tile[local_z + fz][local_y + fy][local_x + fx];
                }
            }
        }
        output[(z * height + y) * width + x] = sum;
    }
}

__global__ void convolution_2d_output_threads(const float* input,
                                               float* output, int width,
                                               int height) {
    __shared__ float tile[input_tile2][input_tile2];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int lane = ty * output_tile2 + tx;
    constexpr int workers = output_tile2 * output_tile2;
    for (int linear = lane; linear < input_tile2 * input_tile2;
         linear += workers) {
        const int shared_y = linear / input_tile2;
        const int shared_x = linear % input_tile2;
        const int x = blockIdx.x * output_tile2 + shared_x - radius2;
        const int y = blockIdx.y * output_tile2 + shared_y - radius2;
        tile[shared_y][shared_x] =
            x >= 0 && x < width && y >= 0 && y < height
                ? input[y * width + x]
                : 0.0f;
    }
    __syncthreads();

    const int x = blockIdx.x * output_tile2 + tx;
    const int y = blockIdx.y * output_tile2 + ty;
    if (x < width && y < height) {
        float sum = 0.0f;
        for (int fy = 0; fy < kernel2; ++fy) {
            for (int fx = 0; fx < kernel2; ++fx) {
                sum += filter2_output_memory[fy * kernel2 + fx] *
                       tile[ty + fy][tx + fx];
            }
        }
        output[y * width + x] = sum;
    }
}

std::vector<float> convolution_3d_cpu(const std::vector<float>& input,
                                      const std::vector<float>& filter,
                                      int radius, int width, int height,
                                      int depth) {
    const int size = 2 * radius + 1;
    if (radius < 0 || width <= 0 || height <= 0 || depth <= 0 ||
        input.size() !=
            static_cast<std::size_t>(width) * height * depth ||
        filter.size() !=
            static_cast<std::size_t>(size) * size * size) {
        throw std::invalid_argument("invalid 3D convolution dimensions");
    }
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

std::vector<float> convolution_2d_cpu(const std::vector<float>& input,
                                      const std::vector<float>& filter,
                                      int radius, int width, int height) {
    const int size = 2 * radius + 1;
    if (radius < 0 || width <= 0 || height <= 0 ||
        input.size() != static_cast<std::size_t>(width) * height ||
        filter.size() != static_cast<std::size_t>(size) * size) {
        throw std::invalid_argument("invalid 2D convolution dimensions");
    }
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

void verify(const std::vector<float>& actual,
            const std::vector<float>& expected, const char* label) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(std::string(label) + ": size mismatch");
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 5.0e-4f + 5.0e-4f * std::abs(expected[i]);
        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error(std::string(label) +
                                     ": mismatch at index " +
                                     std::to_string(i));
        }
    }
}

void validate_reference(const std::vector<float>& values,
                        std::size_t expected_size, const char* label) {
    if (values.size() != expected_size) {
        throw std::runtime_error(std::string(label) + ": size mismatch");
    }
    bool any_nonzero = false;
    for (const float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string(label) +
                                     ": non-finite CPU result");
        }
        any_nonzero = any_nonzero || value != 0.0f;
    }
    if (!any_nonzero) {
        throw std::runtime_error(std::string(label) +
                                 ": unexpectedly all-zero CPU result");
    }
}

int main(int argc, char** argv) {
    try {
        constexpr int width3 = 13;
        constexpr int height3 = 11;
        constexpr int depth3 = 9;
        const std::size_t elements3 =
            static_cast<std::size_t>(width3) * height3 * depth3;
        std::vector<float> input3(elements3);
        for (int z = 0; z < depth3; ++z) {
            for (int y = 0; y < height3; ++y) {
                for (int x = 0; x < width3; ++x) {
                    input3[(z * height3 + y) * width3 + x] =
                        static_cast<float>((x * 3 + y * 5 + z * 7) % 29 -
                                           14) /
                        16.0f;
                }
            }
        }
        std::vector<float> filter3(kernel3 * kernel3 * kernel3);
        for (std::size_t i = 0; i < filter3.size(); ++i) {
            filter3[i] = static_cast<float>(static_cast<int>(i % 7) - 3) /
                         64.0f;
        }
        filter3[filter3.size() / 2] += 0.75f;
        const std::vector<float> expected3 = convolution_3d_cpu(
            input3, filter3, radius3, width3, height3, depth3);
        validate_reference(expected3, elements3, "3D CPU reference");

        constexpr int width2 = 19;
        constexpr int height2 = 18;
        const std::size_t elements2 =
            static_cast<std::size_t>(width2) * height2;
        std::vector<float> input2(elements2);
        for (int y = 0; y < height2; ++y) {
            for (int x = 0; x < width2; ++x) {
                input2[y * width2 + x] =
                    static_cast<float>((x * 11 + y * 3) % 31 - 15) / 16.0f;
            }
        }
        std::vector<float> filter2(kernel2 * kernel2);
        for (std::size_t i = 0; i < filter2.size(); ++i) {
            filter2[i] = static_cast<float>(static_cast<int>(i % 5) - 2) /
                         32.0f;
        }
        filter2[filter2.size() / 2] += 0.5f;
        const std::vector<float> expected2 =
            convolution_2d_cpu(input2, filter2, radius2, width2, height2);
        validate_reference(expected2, elements2, "2D CPU reference");

        if (cpu_only_requested(argc, argv) || !has_cuda_device()) {
            verify(convolution_3d_cpu(input3, filter3, radius3, width3,
                                      height3, depth3),
                   expected3, "3D CPU repeat");
            verify(convolution_2d_cpu(input2, filter2, radius2, width2,
                                      height2),
                   expected2, "2D CPU repeat");
            report_cpu_only("ch07_ex08-11_convolution");
            return EXIT_SUCCESS;
        }

        device_buffer<float> d_input3(elements3), d_filter3(filter3.size()),
            d_output3(elements3);
        CUDA_CHECK(cudaMemcpy(d_input3.get(), input3.data(),
                              elements3 * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_filter3.get(), filter3.data(),
                              filter3.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        std::vector<float> actual3(elements3);

        const dim3 basic_block(4, 4, 4);
        const dim3 basic_grid((width3 + basic_block.x - 1) / basic_block.x,
                              (height3 + basic_block.y - 1) / basic_block.y,
                              (depth3 + basic_block.z - 1) / basic_block.z);
        convolution_3d_basic<<<basic_grid, basic_block>>>(
            d_input3.get(), d_filter3.get(), d_output3.get(), radius3, width3,
            height3, depth3);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual3.data(), d_output3.get(),
                              elements3 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        verify(actual3, expected3, "basic 3D kernel");

        CUDA_CHECK(cudaMemcpyToSymbol(filter3_constant_memory, filter3.data(),
                                      filter3.size() * sizeof(float)));
        convolution_3d_constant<<<basic_grid, basic_block>>>(
            d_input3.get(), d_output3.get(), width3, height3, depth3);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual3.data(), d_output3.get(),
                              elements3 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        verify(actual3, expected3, "constant 3D kernel");

        CUDA_CHECK(cudaMemcpyToSymbol(filter3_tiled_memory, filter3.data(),
                                      filter3.size() * sizeof(float)));
        const dim3 tiled_block(input_tile3, input_tile3, input_tile3);
        const dim3 tiled_grid(
            (width3 + output_tile3 - 1) / output_tile3,
            (height3 + output_tile3 - 1) / output_tile3,
            (depth3 + output_tile3 - 1) / output_tile3);
        convolution_3d_tiled<<<tiled_grid, tiled_block>>>(
            d_input3.get(), d_output3.get(), width3, height3, depth3);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual3.data(), d_output3.get(),
                              elements3 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        verify(actual3, expected3, "tiled 3D kernel");

        device_buffer<float> d_input2(elements2), d_output2(elements2);
        CUDA_CHECK(cudaMemcpy(d_input2.get(), input2.data(),
                              elements2 * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(filter2_output_memory, filter2.data(),
                                      filter2.size() * sizeof(float)));
        const dim3 output_block(output_tile2, output_tile2);
        const dim3 output_grid(
            (width2 + output_tile2 - 1) / output_tile2,
            (height2 + output_tile2 - 1) / output_tile2);
        convolution_2d_output_threads<<<output_grid, output_block>>>(
            d_input2.get(), d_output2.get(), width2, height2);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> actual2(elements2);
        CUDA_CHECK(cudaMemcpy(actual2.data(), d_output2.get(),
                              elements2 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        verify(actual2, expected2, "output-thread 2D kernel");

        std::cout << "ch07_ex08-11_convolution: all four kernels agree with CPU.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
