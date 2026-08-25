# 《大规模并行处理器编程（第 5 版）》中文学习译稿

这是基于 _Programming Massively Parallel Processors: A Hands-on Approach, Fifth Edition_ 的个人学习译稿工程。原书作者、出版商及其他权利人保留全部版权。本工程仅用于个人学习、术语讨论和排版实验；未经权利人授权，不应公开发布或商业传播。

## 当前进度

- 已建立 XeLaTeX/CTeX 编译骨架。
- 已完成第一章“引言”的初译。
- 已完成第二章“异构数据并行计算”的初译，包含 CUDA C++ 代码示例和练习。
- 已完成第三章“多维网格与数据”的初译，完成底本对照、CUDA C++ 代码核对、图示重绘和 Tectonic 编译验证。
- 已完成第四章“计算架构与调度”的初译，包含线程块调度、warp、控制流分歧、占用率、设备属性查询、练习和参考文献。
- 已完成第五章“内存架构与数据局部性”的初译，包含 Roofline 模型、CUDA 内存类型、分块矩阵乘法、边界检查、动态共享内存、练习和参考文献。
- 已完成第六章“性能考量”的初译，包含全局内存访问合并、DRAM 通道与 bank、向量加载和存储、共享内存 bank 冲突、线程粗化、循环展开、双缓冲、优化检查清单、练习和参考文献。
- 已完成第七章“卷积”的初译，包含一维和二维卷积、边界条件、ghost cell、常量内存与缓存、共享内存分块、halo cell、CUDA C++ 代码、练习和概念图重绘。
- 已完成第八章“模板计算与线程粗化”的初译，包含有限差分背景、基本模板内核、共享内存分块、线程粗化、寄存器分块和练习。
- 已完成第九章“直方图”的初译，包含竞态条件、CUDA C++ 原子操作、原子吞吐量分析、私有化、线程粗化、线程级私有化、练习和概念图重绘。
- 已完成第十章“归约”的初译，包含归约树、控制流与内存访问分歧、共享内存、warp 级原语、两阶段 warp 内归约、多块原子归约、线程粗化、练习和 21 幅图示重绘。
- 第一至第十章已按原书顺序纳入 `main.tex`。当前整书译稿可由 Tectonic 编译生成 PDF，并已完成交叉引用和版面警告检查。
- 已完成第十七章初译。
- 已完成第二十章“大语言模型”的初译，包含 Transformer、注意力、CUDA softmax、KV cache、FlashAttention、batching、speculative decoding、MQA、GQA、MLA、PagedAttention、MoE、练习和参考文献。
- 第一至第八章及第二十章已纳入 `main.tex`；第九至十九章暂缺，但第二十章通过显式章节计数保持原书编号。
- 第二十章中的 19 幅图目前均采用中文概念示意或代码清单临时重绘，后续可替换为经许可的原版图像或更精确的矢量图。
- 已完成章节中的原书插图目前以中文概念示意重绘，后续可替换为经许可的图像或更精确的矢量图。
- 其余章节按原书目录预留，采用“每章一个 `.tex` 文件”的方式持续推进。

## 编译

可以使用 Tectonic（会自动获取所需宏包）：

```bash
tectonic main.tex
```

也可以使用包含 `ctex`、`geometry`、`amsmath`、`amssymb`、`booktabs`、 `enumitem`、`graphicx`、`xcolor`、`array`、`tabularx`、`listings`、`hyperref` 和 `bookmark` 的 XeLaTeX 环境，例如 macOS 上的 MacTeX 或配置了自动安装宏包的 TinyTeX：

```bash
latexmk -xelatex -interaction=nonstopmode -file-line-error main.tex
```

生成的文件为 `main.pdf`。文档使用 TeX Live 自带的 Fandol 中文字体，避免依赖特定操作系统字体。清理辅助文件：

```bash
latexmk -C
```

## 翻译约定

- CUDA、GPU、CPU、OpenMP、MPI、NCCL、NVSHMEM 等 API、硬件和缩写保持英文。
- 首次出现的关键术语采用“中文（英文）”形式，后续尽量使用统一中文译名。
- 代码、函数名、变量名和命令使用等宽字体，不翻译其标识符。
- 公式保留原书含义；页码和图号按中文版重新生成，不机械照搬原书页码。
- 原书中可能存在的编辑错误或版本差异，以译者注标出，不擅自修改原文语义。

## 目录

当前已纳入完成初译的前十章，第十七章，第二十章。
在 `main.tex` 中加入相应的 `\include`，并按原书章节编号调整计数器即可。
