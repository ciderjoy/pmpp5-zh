# 《大规模并行处理器编程（第 5 版）》中文学习译稿

这是基于 _Programming Massively Parallel Processors: A Hands-on Approach, Fifth Edition_ 的个人学习译稿工程。原书作者、出版商及其他权利人保留全部版权。本工程仅用于个人学习、术语讨论和排版实验；未经权利人授权，不应公开发布或商业传播。

## 当前进度

- 已建立 XeLaTeX/CTeX 编译骨架。
- 已完成第 1--22 章初译并纳入 `main.tex`，覆盖基础概念、并行模式和高级模式与应用。
- 第十八章“图遍历”包含 BFS 的顶点中心/边中心实现、方向优化、前沿、私有化、协作组、17 幅编号图、练习图、3 道练习和 5 条参考文献。
- 第十九章“卷积神经网络”包含卷积层前向路径、CUDA 内核、GEMM 显式/隐式展开和 cuDNN，共重绘或重排 11 幅编号图，并完成公式、表格、练习和参考文献核对。
- 第 20 章“大语言模型”已按原书章节号接入主文档，包含 Transformer、注意力、KV cache、FlashAttention 及相关 CUDA 实现。
- 第 21 章“静电势图”已完成全章翻译，覆盖散布/聚集、线程粗化、内存访问合并、截断求和、分箱和溢出列表，并重绘或重排图 21.1--21.14。
- 第 22 章“算法选择、问题分解与问题表述”已完成全章翻译，覆盖算法性质与算法复杂度权衡、输出中心/输入中心分解、阿姆达尔定律、问题表述和批处理延迟/吞吐量，并用 TikZ 重绘图 22.1--22.2。
- 当前译稿使用 Tectonic 编译；原书插图暂以中文概念图或可复制的 CUDA 代码清单重绘，后续可替换为经许可的原版图。
- 详细章节状态、验收记录和待办事项见 [`TRANSLATION_STATUS.md`](TRANSLATION_STATUS.md)。

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
