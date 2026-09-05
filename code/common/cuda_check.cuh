#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

// CUDA Runtime API 通过返回值报告错误，不会自动抛出 C++ 异常。
// 把表达式、源文件和行号一起打印出来，能避免只看到
// “invalid argument”却不知道是哪一次 CUDA 调用失败。
inline void check_cuda(cudaError_t status,
                       const char* expression,
                       const char* file,
                       int line) {
    if (status == cudaSuccess) {
        return;
    }

    std::cerr << "CUDA error at " << file << ':' << line << '\n'
              << "  expression: " << expression << '\n'
              << "  reason: " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) \
    check_cuda((expression), #expression, __FILE__, __LINE__)

