# 翻译进度

本文件记录译稿的阶段性进度。章节标题沿用原书目录，中文标题在正文定稿时再统一校订。

## 已完成

- [x] 第 1 章：Introduction —— 引言（初译，含参考文献和两幅概念示意图）
- [x] 第 2 章：Heterogeneous data-parallel computing —— 异构数据并行计算（初译，已完成对照校订、CUDA 代码核对和图示重绘）
- [x] 第 3 章：Multidimensional grids and data —— 多维网格与数据（初译，已完成底本对照、CUDA 代码核对、图示重绘和编译/视觉验收）
- [x] 第 4 章：Compute architecture and scheduling —— 计算架构与调度（初译，已完成底本对照、CUDA 代码核对、图示重绘和编译/视觉验收）
- [x] 第 5 章：Memory architecture and data locality —— 内存架构与数据局部性（初译，已完成底本对照、CUDA 代码核对、图示重绘和编译/视觉验收）

## 待处理

### 第 1 部分：基础概念

- [ ] 第 6 章：Performance considerations —— 性能考量

### 第 2 部分：并行模式

- [ ] 第 7 章：Convolution —— 卷积
- [ ] 第 8 章：Stencil computation —— 模板计算
- [ ] 第 9 章：Histogram —— 直方图
- [ ] 第 10 章：Reduction —— 归约
- [ ] 第 11 章：Scan —— 扫描
- [ ] 第 12 章：Filter —— 过滤
- [ ] 第 13 章：Merge —— 归并
- [ ] 第 14 章：Sorting —— 排序
- [ ] 第 15 章：Advanced optimizations for matrix multiplication —— 矩阵乘法的高级优化

### 第 3 部分：高级模式与应用

- [ ] 第 16 章：Dynamic programming and wavefront parallelism —— 动态规划与波前并行
- [ ] 第 17 章：Sparse matrix computation —— 稀疏矩阵计算
- [ ] 第 18 章：Graph traversal —— 图遍历
- [ ] 第 19 章：Convolutional neural networks —— 卷积神经网络
- [ ] 第 20 章：Large language models —— 大语言模型
- [ ] 第 21 章：Electrostatic potential map —— 静电势图
- [ ] 第 22 章：Algorithm selection, problem decomposition, and problem formulation —— 算法选择、问题分解与问题表述
- [ ] 第 23 章：Multi-GPU programming —— 多 GPU 编程
- [ ] 第 24 章：Conclusion and future outlook —— 总结与未来展望

### 附录

- [ ] 附录 A：Numerical considerations —— 数值计算考量
- [ ] 附录 B：Deep learning basics —— 深度学习基础
- [ ] 附录 C：CUDA memories, address spaces, and pointers —— CUDA 内存、地址空间与指针

## 后续章节通用验收清单

- [ ] 从原 PDF 提取并核对正文、公式、代码和参考文献。
- [ ] 统一本章新增术语，并回写主术语表。
- [ ] 将图表重绘为可编译的矢量图，或明确标注暂缺原因。
- [ ] 通过 XeLaTeX 至少连续编译两次，清理未定义引用和交叉引用警告。
- [ ] 对照原页逐节检查数字、单位、函数名和算法名称。

## 第四、第五章本轮验收记录

- 已在 `main.tex` 中启用第四章和第五章的独立 `.tex` 文件。
- 已用 Tectonic 完整编译；章节图、代码清单、脚注、练习和参考文献均进入最终 PDF。
- 已检查交叉引用和引用键；当前构建无未定义引用、缺失数学定界符、LaTeX 错误或 overfull/underfull 警告。
- 原书插图以中文概念示意重绘，图号使用译稿中的 `4-1`--`4-10` 和 `5-1`--`5-15` 标签。
- 对底本中动态共享内存参数单位、`BLOCK_WIDTH`/`BLOCK_SIZE` 命名不一致，以及练习 11 的主机/设备指针类型差异，均保留代码并添加译者注。
