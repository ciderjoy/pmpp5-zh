#include "common/cuda_check.hpp"
#include "ch17_23_cpu_algorithms.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

__global__ void subsample_forward_nchw(const float* input, const float* bias,
                                       float* output, int batch, int channels,
                                       int height, int width, int window) {
    const int output_height = height / window;
    const int output_width = width / window;
    const std::size_t total =
        static_cast<std::size_t>(batch) * channels * output_height *
        output_width;
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int output_col = static_cast<int>(index % output_width);
    index /= output_width;
    const int output_row = static_cast<int>(index % output_height);
    index /= output_height;
    const int channel = static_cast<int>(index % channels);
    const int sample = static_cast<int>(index / channels);

    float sum = 0.0f;
    for (int p = 0; p < window; ++p) {
        for (int q = 0; q < window; ++q) {
            const int input_row = output_row * window + p;
            const int input_col = output_col * window + q;
            if (input_row < height && input_col < width) {
                const std::size_t input_index =
                    ((static_cast<std::size_t>(sample) * channels + channel) *
                         height +
                     input_row) *
                        width +
                    input_col;
                sum += input[input_index];
            }
        }
    }
    const float activation =
        sum / static_cast<float>(window * window) + bias[channel];
    const std::size_t output_index =
        ((static_cast<std::size_t>(sample) * channels + channel) *
             output_height +
         output_row) *
            output_width +
        output_col;
    output[output_index] = 1.0f / (1.0f + expf(-activation));
}

std::vector<float> subsample_gpu(const std::vector<float>& input,
                                 const std::vector<float>& bias, int batch,
                                 int channels, int height, int width,
                                 int window) {
    const std::vector<float> validated = pmpp_examples::subsample_forward_cpu(
        input, bias, batch, channels, height, width, window);
    if (validated.empty()) {
        return validated;
    }
    device_buffer<float> device_input(input.size());
    device_buffer<float> device_bias(bias.size());
    device_buffer<float> device_output(validated.size());
    CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(),
                          input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_bias.get(), bias.data(),
                          bias.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>(
        (validated.size() + threads - 1) / threads);
    subsample_forward_nchw<<<blocks, threads>>>(
        device_input.get(), device_bias.get(), device_output.get(), batch,
        channels, height, width, window);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> output(validated.size());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output.get(),
                          output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return output;
}

void run_cpu_assertions(const std::vector<float>& input,
                        const std::vector<float>& bias) {
    const std::vector<float> output = pmpp_examples::subsample_forward_cpu(
        input, bias, 2, 2, 5, 6, 2);
    pmpp_examples::expect(output.size() == 2U * 2U * 2U * 3U,
                          "CPU subsampling output shape");
    pmpp_examples::expect(
        std::all_of(output.begin(), output.end(), [](float value) {
            return value > 0.0f && value < 1.0f && std::isfinite(value);
        }),
        "CPU subsampling sigmoid range");
    bool rejected = false;
    try {
        (void)pmpp_examples::subsample_forward_cpu(input, bias, 2, 2, 5, 6,
                                                    0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    pmpp_examples::expect(rejected, "CPU subsampling window validation");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool cpu_only = pmpp_examples::validate_cli(argc, argv);
        constexpr int batch = 2;
        constexpr int channels = 2;
        constexpr int height = 5;
        constexpr int width = 6;
        constexpr int window = 2;
        std::vector<float> input(static_cast<std::size_t>(batch) * channels *
                                 height * width);
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) *
                       0.125f;
        }
        const std::vector<float> bias{0.15f, -0.2f};
        run_cpu_assertions(input, bias);
        const std::vector<float> reference =
            pmpp_examples::subsample_forward_cpu(
                input, bias, batch, channels, height, width, window);
        if (cpu_only || !has_cuda_device()) {
            report_cpu_only("ch19_ex01_subsample_forward");
            return EXIT_SUCCESS;
        }
        const std::vector<float> actual = subsample_gpu(
            input, bias, batch, channels, height, width, window);
        pmpp_examples::expect_near(actual, reference, 1.0e-6f, 1.0e-5f,
                                   "GPU subsampling");
        std::cout << "ch19_ex01_subsample_forward: GPU comparison passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ch19_ex01_subsample_forward: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
