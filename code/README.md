# 第二、三章 CUDA 教学示例

这里的程序把书中为了突出概念而省略的主机端代码补齐，使每个示例都可以独立编译、
运行和验证。内核算法、参数顺序、二维索引和数值公式与译稿正文保持一致；输入数据、
CPU 基准实现、错误检查和结果输出属于便于学习的扩展。

## 学习顺序

1. `ch2/vec_add.cu`
   - 对照 CPU 循环与 CUDA 线程；
   - 学习 `cudaMalloc`、`cudaMemcpy`、内核启动和 `cudaFree`；
   - 理解一维全局索引和最后一个线程块的边界检查。
2. `ch3/rgb_to_grayscale.cu`
   - 学习二维网格和二维线程块；
   - 理解行主序二维数组及交错 RGB 数据的线性化；
   - 复现书中 `76×62` 图像、`16×16` 线程块和 408 个多余线程。
3. `ch3/image_blur.cu`
   - 保持“一个线程计算一个输出像素”的映射；
   - 让每个线程遍历 `3×3` 邻域；
   - 比较角点、边缘和内部像素不同的有效邻居数量。
4. `ch3/matrix_multiply.cu`
   - 让一个二维线程计算一个输出矩阵元素；
   - 手动跟踪矩阵行与矩阵列的点积；
   - 默认复现书中 `4×4` 矩阵和 `2×2` 线程块的小型示例。

这些是朴素教学实现，不追求最终性能。共享内存分块矩阵乘法等优化将在后续章节出现。

## 环境要求

- 支持 CUDA 的 NVIDIA GPU；
- NVIDIA 驱动；
- CUDA Toolkit，其中需要 `nvcc`；
- 支持 C++17 的主机编译器。

检查编译器：

```bash
nvcc --version
```

## 批量编译和运行

在 `code` 目录中执行：

```bash
make
make run
```

可执行文件生成在 `code/build/`：

```text
build/vec_add
build/rgb_to_grayscale
build/image_blur
build/matrix_multiply
```

如果 `nvcc` 不在 `PATH` 中，可以显式指定：

```bash
make NVCC=/usr/local/cuda/bin/nvcc
```

如需为特定 GPU 指定目标架构，也可以传入 `CUDA_ARCH`，例如：

```bash
make CUDA_ARCH=sm_89
```

具体的 `sm_XX` 值取决于 GPU 的 compute capability。CUDA Toolkit 对 GCC、Clang
或 MSVC 的受支持版本也不同；如果 `nvcc` 报告主机编译器不受支持，应以所安装
Toolkit 的兼容性说明为准。这里的 Makefile 使用 GNU make 和 POSIX shell；Windows
原生 PowerShell/CMD 环境可以直接采用下一节的 `nvcc` 命令。

## 单独编译

每个 `.cu` 文件也可以直接编译：

```bash
nvcc -std=c++17 -O2 ch2/vec_add.cu -o vec_add
nvcc -std=c++17 -O2 ch3/rgb_to_grayscale.cu -o rgb_to_grayscale
nvcc -std=c++17 -O2 ch3/image_blur.cu -o image_blur
nvcc -std=c++17 -O2 ch3/matrix_multiply.cu -o matrix_multiply
```

上面的源文件路径以当前目录是 `code/` 为前提。各文件通过相对于源文件自身的路径
包含 `common/cuda_check.cuh`，因此不需要额外的 `-I` 参数；如果从其他目录编译，
需要相应地写出 `.cu` 文件的正确路径。

## 建议运行方式

### 第二章：向量加法

```bash
./build/vec_add
./build/vec_add 1003
```

`1003` 不是 256 的整数倍，适合观察网格向上取整和多余线程的边界检查。程序会打印
线程总数、被 `if (i < n)` 屏蔽的线程数、若干结果，以及 GPU 与 CPU 的逐元素校验。

### 第三章：RGB 转灰度

```bash
./build/rgb_to_grayscale
./build/rgb_to_grayscale 31 19
```

默认尺寸为 `width=76`、`height=62`。输入是程序生成的确定性 RGB 图案，不依赖
OpenCV 或图片文件。程序会打印书中示例像素 `(row,col)=(16,0)` 的：

```text
gray_offset = 16 * 76 + 0 = 1216
rgb_offset  = 1216 * 3 = 3648
```

灰度公式与正文一致：

```text
L = 0.299R + 0.587G + 0.114B
```

这里计算的是经过传递函数编码的 RGB 信号的 luma；程序故意保留书中转换为
`unsigned char` 时截断小数部分的行为。CPU 与 GPU 可能对浮点乘加采用不同的融合
方式，因此校验允许最多 1 个灰度级的差异；更大的差异通常意味着需要检查算法、
参数顺序或索引。这个容差校验不能检测所有恰好偏差 1 的实现错误。

### 第三章：图像模糊

```bash
./build/image_blur
./build/image_blur 12 9
```

默认输入很小，会以数字矩阵形式打印模糊前后的图像。`BLUR_SIZE=1` 表示半径为 1，
因此窗口边长是 `2*BLUR_SIZE+1=3`。角点只有 4 个有效样本，非角点边缘通常有
6 个，内部像素有 9 个；分母必须使用实际有效样本数。

### 第三章：矩阵乘法

```bash
./build/matrix_multiply
./build/matrix_multiply 17 8
```

参数依次是方阵宽度和方形线程块边长。默认 `4 2` 便于逐项查看；`17 8` 则适合观察
矩阵宽度不是线程块边长整数倍时的边界线程。程序会打印线程 `(row,col)=(0,0)`
如何在 `k` 循环中计算第一个点积。

## 阅读代码时重点观察

- `blockIdx` 标识线程块，`threadIdx` 标识块内线程；
- 执行配置使用 `(x,y,z)` 顺序，而图和矩阵坐标通常写成 `(row,col)`，也就是
  `(y,x)`；
- 行主序二维索引是 `row*width+col`；
- 网格尺寸使用向上取整，所以内核必须做边界检查；
- 内核启动是异步的：`cudaGetLastError()` 检查启动错误，
  `cudaDeviceSynchronize()` 或事件同步负责暴露执行期间的错误；
- 通过 CPU 基准答案检查 GPU 结果，比只打印“看起来合理”的几个数可靠得多。

程序会在 CUDA API 或内核执行失败时立即退出。输入尺寸仍受主机内存、GPU 显存和
当前设备网格上限约束；这些示例面向教学，不负责把超大输入自动切分成多个批次。
