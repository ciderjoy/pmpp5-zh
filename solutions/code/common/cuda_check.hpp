#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t status, const char* expression,
                       const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                             ": " + expression + ": " +
                             cudaGetErrorString(status));
}

#define CUDA_CHECK(expression) \
    cuda_check((expression), #expression, __FILE__, __LINE__)

inline bool has_cuda_device() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaSuccess) {
        return count > 0;
    }
    cudaGetLastError();
    return false;
}

inline bool cpu_only_requested(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--cpu-only") {
            return true;
        }
    }
    return false;
}

inline void report_cpu_only(const char* example) {
    std::cout << example
              << ": CUDA device unavailable or --cpu-only requested; "
                 "CPU reference checks passed.\n";
}

template <class T>
class device_buffer {
public:
    explicit device_buffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_),
                                  count_ * sizeof(T)));
        }
    }

    ~device_buffer() {
        if (data_) {
            cudaFree(data_);
        }
    }

    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    T* get() const { return data_; }
    std::size_t size() const { return count_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};
