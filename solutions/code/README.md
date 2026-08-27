# 可执行示例

本目录存放题解中算法与 CUDA 片段的独立、可构建版本。每个源文件都包含 `main`、输入检查、
边界保护和 CPU 参考验证；CUDA 程序接受 `--cpu-only`，没有 NVIDIA GPU 时仍可运行参考路径。
源文件是为本题解独立编写的实现，不是从底本插图复制的代码。

## 构建

构建目录必须位于仓库外。在仓库根目录执行：

```powershell
solutions\code\build.ps1 `
  -BuildDirectory D:\translation\tmp\pmpp5-solutions-code-vs2022-fresh `
  -CudaArchitecture 75
```

Windows 脚本用 CMake 构建普通 C++，再用 `nvcc` 和 VS 2022 的 x64 主机编译器逐个构建
CUDA 文件，并默认运行其 `--cpu-only` 参考路径。默认架构 75 只用于本机无 NVIDIA GPU 时
的编译验收，不代表性能建议。已有 NVIDIA GPU 时，应把参数改成该设备支持的计算能力；
参数也接受带后缀的目标，例如 Hopper 特定功能可用 `90a`。脚本要求使用新的仓库外构建
目录；若已有缓存的生成器、平台或 VS 实例不一致，脚本会停止并要求换目录，不会擅自删除缓存。

Windows 安全策略拦截新生成的未签名 CUDA 包装程序时，可保留完整编译与 CTest，只跳过
逐个运行包装程序：

```powershell
solutions\code\build.ps1 `
  -BuildDirectory D:\translation\tmp\pmpp5-solutions-code-build-compile `
  -CudaArchitecture 75 -SkipCudaRun
```

`-SkipCTest` 只跳过 CTest，`-SkipCudaRun` 只跳过 CUDA 包装程序运行；兼容参数 `-SkipRun`
同时跳过两者。若要由 CMake 自身启用 CUDA 语言，可手动配置：

```powershell
cmake -S solutions\code -B D:\translation\tmp\pmpp5-solutions-code-cmake-cuda `
  -G "Visual Studio 17 2022" -A x64 `
  -DPMPP_ENABLE_CUDA_WITH_CMAKE=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build D:\translation\tmp\pmpp5-solutions-code-cmake-cuda --config Release
```

CMake 的 CUDA 路径默认关闭，因为 CUDA Toolkit 的 Visual Studio 集成不是直接 `nvcc`
编译的必要条件；在已经安装集成的环境中可显式设置 `PMPP_ENABLE_CUDA_WITH_CMAKE=ON`。

MPI 示例文件以 `_mpi.cpp` 结尾；只有在 CMake 找到 MPI C++ 实现时才会加入构建。
构建产物、缓存和运行日志不得放入或提交到本仓库。

## 代码与题解映射

目录共有 27 个 CUDA 程序和 6 个普通 C++/MPI 程序。详细映射见：

- `ch02-09/README.md`：第 2--9 章的 14 个题解代码清单映射到 9 个 CUDA 程序；
- `ch10-16/README.md`：第 10--16 章的 24 个题解代码清单映射到 9 个 CUDA 程序；
- `ch17-23/README.md`：第 17--23 章的 15 个代码清单和 27 道题映射到 9 个 CUDA 程序、
  2 个第 23 章 C++/MPI 程序及 CPU 参考测试；
- `appendices/appendix_a_floating_point.cpp`：附录 A 的 ULP、归约次序和迭代次数验证。

## 本机验收结果

在 CUDA Toolkit 13.3、MSVC 14.44 和 Microsoft MPI 环境中，验收目录
`D:\translation\tmp\pmpp5-solutions-code-msvc-final-20260826` 的 CTest 为 6/6，通过项包含
三个章节 CPU 参考、附录 A、分区计数和两秩 halo 交换。全部 27 个 `.cu` 文件均以 `sm_75`
编译链接；Smith--Waterman 另以 `sm_90` 编译，验证 DPX 条件分支。安全策略收紧前的基准
验收中，27 个可执行文件的 `--cpu-only` 路径均实际退出 0。

最终源码在新默认目录 `D:\translation\tmp\pmpp5-solutions-code-vs2022` 再次完成 6 个
C++/MPI 目标及 27 个 CUDA 程序的 fresh compile。用户关闭 Smart App Control 后，
`SmartAppControlState` 查询为 `Off`，策略状态为 0；该目录的 CTest 6/6 全部通过。27 个
CUDA 可执行文件随后逐一以 `--cpu-only` 启动，枚举 27、实际启动 27、通过 27、失败 0、
超时 0、非空标准错误 0，且每个程序都明确报告 CPU reference checks passed。

本机没有 NVIDIA GPU，设备内核与 Compute Sanitizer 未实际运行，不能把 CPU 参考运行表述为
GPU 运行。
