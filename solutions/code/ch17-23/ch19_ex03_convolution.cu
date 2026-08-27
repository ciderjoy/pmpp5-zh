#include "common/cuda_check.hpp"
#include "ch17_23_cpu_algorithms.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using pmpp_examples::ConvolutionGradients;
using pmpp_examples::ConvolutionShape;

__global__ void convolution_forward(const float* input, const float* filter,
                                    float* output, int batch,
                                    int output_channels, int input_channels,
                                    int height, int width, int kernel) {
    const int output_height = height - kernel + 1;
    const int output_width = width - kernel + 1;
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(batch) *
                              output_channels * output_height * output_width;
    if (index >= total) {
        return;
    }
    const int output_col = static_cast<int>(index % output_width);
    index /= output_width;
    const int output_row = static_cast<int>(index % output_height);
    index /= output_height;
    const int output_channel = static_cast<int>(index % output_channels);
    const int sample = static_cast<int>(index / output_channels);
    float sum = 0.0f;
    for (int input_channel = 0; input_channel < input_channels;
         ++input_channel) {
        for (int p = 0; p < kernel; ++p) {
            for (int q = 0; q < kernel; ++q) {
                const int input_row = output_row + p;
                const int input_col = output_col + q;
                if (input_row < height && input_col < width) {
                    const std::size_t input_index =
                        ((static_cast<std::size_t>(sample) * input_channels +
                          input_channel) *
                             height +
                         input_row) *
                            width +
                        input_col;
                    const std::size_t filter_index =
                        ((static_cast<std::size_t>(output_channel) *
                              input_channels +
                          input_channel) *
                             kernel +
                         p) *
                            kernel +
                        q;
                    sum += input[input_index] * filter[filter_index];
                }
            }
        }
    }
    const std::size_t output_index =
        ((static_cast<std::size_t>(sample) * output_channels +
          output_channel) *
             output_height +
         output_row) *
            output_width +
        output_col;
    output[output_index] = sum;
}

__global__ void conv_backward_x(const float* upstream, const float* filter,
                                float* input_gradient, int batch,
                                int output_channels, int input_channels,
                                int height, int width, int kernel) {
    const int output_height = height - kernel + 1;
    const int output_width = width - kernel + 1;
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total =
        static_cast<std::size_t>(batch) * input_channels * height * width;
    if (index >= total) {
        return;
    }
    const int input_col = static_cast<int>(index % width);
    index /= width;
    const int input_row = static_cast<int>(index % height);
    index /= height;
    const int input_channel = static_cast<int>(index % input_channels);
    const int sample = static_cast<int>(index / input_channels);
    float sum = 0.0f;
    for (int output_channel = 0; output_channel < output_channels;
         ++output_channel) {
        for (int p = 0; p < kernel; ++p) {
            for (int q = 0; q < kernel; ++q) {
                const int output_row = input_row - p;
                const int output_col = input_col - q;
                if (output_row >= 0 && output_row < output_height &&
                    output_col >= 0 && output_col < output_width) {
                    const std::size_t output_index =
                        ((static_cast<std::size_t>(sample) * output_channels +
                          output_channel) *
                             output_height +
                         output_row) *
                            output_width +
                        output_col;
                    const std::size_t filter_index =
                        ((static_cast<std::size_t>(output_channel) *
                              input_channels +
                          input_channel) *
                             kernel +
                         p) *
                            kernel +
                        q;
                    sum += upstream[output_index] * filter[filter_index];
                }
            }
        }
    }
    const std::size_t input_index =
        ((static_cast<std::size_t>(sample) * input_channels + input_channel) *
             height +
         input_row) *
            width +
        input_col;
    input_gradient[input_index] = sum;
}

__global__ void conv_backward_f(const float* upstream, const float* input,
                                float* filter_gradient, int batch,
                                int output_channels, int input_channels,
                                int height, int width, int kernel) {
    const int output_height = height - kernel + 1;
    const int output_width = width - kernel + 1;
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(output_channels) *
                              input_channels * kernel * kernel;
    if (index >= total) {
        return;
    }
    const int q = static_cast<int>(index % kernel);
    index /= kernel;
    const int p = static_cast<int>(index % kernel);
    index /= kernel;
    const int input_channel = static_cast<int>(index % input_channels);
    const int output_channel = static_cast<int>(index / input_channels);
    float sum = 0.0f;
    for (int sample = 0; sample < batch; ++sample) {
        for (int output_row = 0; output_row < output_height; ++output_row) {
            for (int output_col = 0; output_col < output_width; ++output_col) {
                const std::size_t input_index =
                    ((static_cast<std::size_t>(sample) * input_channels +
                      input_channel) *
                         height +
                     output_row + p) *
                        width +
                    output_col + q;
                const std::size_t output_index =
                    ((static_cast<std::size_t>(sample) * output_channels +
                      output_channel) *
                         output_height +
                     output_row) *
                        output_width +
                    output_col;
                sum += input[input_index] * upstream[output_index];
            }
        }
    }
    const std::size_t filter_index =
        ((static_cast<std::size_t>(output_channel) * input_channels +
          input_channel) *
             kernel +
         p) *
            kernel +
        q;
    filter_gradient[filter_index] = sum;
}

__global__ void conv_backward_b(const float* upstream, float* bias_gradient,
                                int batch, int output_channels,
                                int output_height, int output_width) {
    const int output_channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_channel >= output_channels) {
        return;
    }
    float sum = 0.0f;
    for (int sample = 0; sample < batch; ++sample) {
        for (int row = 0; row < output_height; ++row) {
            for (int col = 0; col < output_width; ++col) {
                const std::size_t index =
                    ((static_cast<std::size_t>(sample) * output_channels +
                      output_channel) *
                         output_height +
                     row) *
                        output_width +
                    col;
                sum += upstream[index];
            }
        }
    }
    bias_gradient[output_channel] = sum;
}

struct GpuConvolutionResult {
    std::vector<float> output;
    ConvolutionGradients gradients;
};

GpuConvolutionResult run_convolution_gpu(
    const std::vector<float>& input, const std::vector<float>& filter,
    const std::vector<float>& upstream, const ConvolutionShape& shape) {
    const std::vector<float> validated_output =
        pmpp_examples::convolution_forward_cpu(input, filter, shape);
    const ConvolutionGradients validated_gradients =
        pmpp_examples::convolution_backward_cpu(input, filter, upstream, shape);
    (void)validated_gradients;
    device_buffer<float> device_input(input.size());
    device_buffer<float> device_filter(filter.size());
    device_buffer<float> device_upstream(upstream.size());
    device_buffer<float> device_output(validated_output.size());
    device_buffer<float> device_input_gradient(input.size());
    device_buffer<float> device_filter_gradient(filter.size());
    device_buffer<float> device_bias_gradient(
        static_cast<std::size_t>(shape.output_channels));
    CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(),
                          input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_filter.get(), filter.data(),
                          filter.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_upstream.get(), upstream.data(),
                          upstream.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    constexpr std::size_t threads = 256;
    const auto blocks_for = [](std::size_t count) {
        constexpr std::size_t block_threads = 256;
        return static_cast<unsigned>((count + block_threads - 1) /
                                     block_threads);
    };
    convolution_forward<<<blocks_for(validated_output.size()), threads>>>(
        device_input.get(), device_filter.get(), device_output.get(),
        shape.batch, shape.output_channels, shape.input_channels, shape.height,
        shape.width, shape.kernel);
    CUDA_CHECK(cudaGetLastError());
    conv_backward_x<<<blocks_for(input.size()), threads>>>(
        device_upstream.get(), device_filter.get(),
        device_input_gradient.get(), shape.batch, shape.output_channels,
        shape.input_channels, shape.height, shape.width, shape.kernel);
    CUDA_CHECK(cudaGetLastError());
    conv_backward_f<<<blocks_for(filter.size()), threads>>>(
        device_upstream.get(), device_input.get(),
        device_filter_gradient.get(), shape.batch, shape.output_channels,
        shape.input_channels, shape.height, shape.width, shape.kernel);
    CUDA_CHECK(cudaGetLastError());
    conv_backward_b<<<blocks_for(
                          static_cast<std::size_t>(shape.output_channels)),
                      threads>>>(device_upstream.get(),
                                 device_bias_gradient.get(), shape.batch,
                                 shape.output_channels, shape.output_height(),
                                 shape.output_width());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuConvolutionResult result;
    result.output.resize(validated_output.size());
    result.gradients.input.resize(input.size());
    result.gradients.filter.resize(filter.size());
    result.gradients.bias.resize(
        static_cast<std::size_t>(shape.output_channels));
    CUDA_CHECK(cudaMemcpy(result.output.data(), device_output.get(),
                          result.output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.gradients.input.data(),
                          device_input_gradient.get(),
                          result.gradients.input.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.gradients.filter.data(),
                          device_filter_gradient.get(),
                          result.gradients.filter.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.gradients.bias.data(),
                          device_bias_gradient.get(),
                          result.gradients.bias.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return result;
}

void run_cpu_assertions(const std::vector<float>& input,
                        const std::vector<float>& filter,
                        const std::vector<float>& upstream,
                        const ConvolutionShape& shape) {
    const std::vector<float> output =
        pmpp_examples::convolution_forward_cpu(input, filter, shape);
    const ConvolutionGradients gradients =
        pmpp_examples::convolution_backward_cpu(input, filter, upstream, shape);
    pmpp_examples::expect(output.size() == upstream.size(),
                          "CPU convolution output shape");
    pmpp_examples::expect(gradients.input.size() == input.size() &&
                              gradients.filter.size() == filter.size() &&
                              gradients.bias.size() ==
                                  static_cast<std::size_t>(
                                      shape.output_channels),
                          "CPU convolution gradient shapes");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool cpu_only = pmpp_examples::validate_cli(argc, argv);
        const ConvolutionShape shape{2, 2, 2, 4, 5, 3};
        std::vector<float> input(
            pmpp_examples::convolution_input_count(shape));
        std::vector<float> filter(
            pmpp_examples::convolution_filter_count(shape));
        std::vector<float> upstream(
            pmpp_examples::convolution_output_count(shape));
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(static_cast<int>(i % 13) - 6) *
                       0.08f;
        }
        for (std::size_t i = 0; i < filter.size(); ++i) {
            filter[i] = static_cast<float>(static_cast<int>(i % 9) - 4) *
                        0.06f;
        }
        for (std::size_t i = 0; i < upstream.size(); ++i) {
            upstream[i] = static_cast<float>(static_cast<int>(i % 7) - 3) *
                          0.09f;
        }
        run_cpu_assertions(input, filter, upstream, shape);
        const std::vector<float> reference_output =
            pmpp_examples::convolution_forward_cpu(input, filter, shape);
        const ConvolutionGradients reference_gradients =
            pmpp_examples::convolution_backward_cpu(input, filter, upstream,
                                                     shape);
        if (cpu_only || !has_cuda_device()) {
            report_cpu_only("ch19_ex03_convolution");
            return EXIT_SUCCESS;
        }
        const GpuConvolutionResult actual =
            run_convolution_gpu(input, filter, upstream, shape);
        pmpp_examples::expect_near(actual.output, reference_output, 1.0e-5f,
                                   2.0e-5f, "GPU convolution forward");
        pmpp_examples::expect_near(actual.gradients.input,
                                   reference_gradients.input, 1.0e-5f,
                                   3.0e-5f, "GPU convolution dX");
        pmpp_examples::expect_near(actual.gradients.filter,
                                   reference_gradients.filter, 1.0e-5f,
                                   3.0e-5f, "GPU convolution dF");
        pmpp_examples::expect_near(actual.gradients.bias,
                                   reference_gradients.bias, 1.0e-5f,
                                   3.0e-5f, "GPU convolution db");
        std::cout << "ch19_ex03_convolution: GPU comparisons passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ch19_ex03_convolution: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
