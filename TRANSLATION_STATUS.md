# 翻译进度

本文件记录译稿的阶段性进度。章节标题沿用原书目录，中文标题在正文定稿时再统一校订。

## 已完成

- [x] 第 1 章：Introduction —— 引言（初译，含参考文献和两幅概念示意图）
- [x] 第 2 章：Heterogeneous data-parallel computing —— 异构数据并行计算（初译，已完成对照校订、CUDA 代码核对和图示重绘）
- [x] 第 3 章：Multidimensional grids and data —— 多维网格与数据（初译，已完成底本对照、CUDA 代码核对、图示重绘和编译/视觉验收）
- [x] 第 4 章：Compute architecture and scheduling —— 计算架构与调度（初译，已完成底本对照、CUDA 代码核对、图示重绘和编译/视觉验收）
- [x] 第 5 章：Memory architecture and data locality —— 内存架构与数据局部性（初译，已完成底本对照、CUDA 代码核对、图示重绘和编译/视觉验收）
- [x] 第 6 章：Performance considerations —— 性能考量（初译，已完成底本对照、CUDA 代码核对、图示重绘和编译/视觉验收）
- [x] 第 20 章：Large language models —— 大语言模型（初译，已完成底本对照、公式、CUDA 代码、临时图示、练习、参考文献和整书编译/视觉验收）

## 待处理

### 第 2 部分：并行模式

- [x] 第 7 章：Convolution —— 卷积（初译，已完成底本对照、CUDA 代码核对、图示重绘和编译/视觉验收）
- [x] 第 8 章：Stencil computation —— 模板计算与线程粗化（初译，已完成底本对照、CUDA 代码核对、图示重绘和编译/视觉验收）
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
- [x] 第 20 章：Large language models —— 大语言模型（已先行完成；第 9--19 章暂缺）
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

## 第二十章本轮记录

- 已核对原书物理页 505--540，对应印刷页 477--512；正文包含 20.1--20.8、
  20.9 练习、15 个编号公式、19 幅图和 17 条参考文献。
- 已完成 Transformer 架构、多头注意力、CUDA softmax、KV caching、FlashAttention、
  batching、speculative decoding、MQA、GQA、MLA、PagedAttention 和 MoE 的中文初译。
- 图 20.1--20.3、20.5--20.8、20.17--20.19 采用中文概念示意临时重绘；图 20.4、
  20.9--20.16 采用保留 CUDA 标识符的可复制代码清单。
- 已在 `main.tex` 中加入第三部分，并显式设置章节计数以跳过第 9--19 章而显示
  第二十章。
- 已使用 Tectonic 完整编译，最终 `main.pdf` 共 200 页；构建日志无 LaTeX 错误、
  未定义引用、未定义标签、overfull/underfull 或 rerun 警告。
- 已视觉抽查第二十章章首页、图 20.2、图 20.5、图 20.8、图 20.9、图 20.16、
  图 20.17、图 20.18、图 20.19、公式 20.15、练习页和参考文献页，版面正常。

## 第四、第五章本轮验收记录

- 已在 `main.tex` 中启用第四章和第五章的独立 `.tex` 文件。
- 已用 Tectonic 完整编译；章节图、代码清单、脚注、练习和参考文献均进入最终 PDF。
- 已检查交叉引用和引用键；当前构建无未定义引用、缺失数学定界符、LaTeX 错误或 overfull/underfull 警告。
- 原书插图以中文概念示意重绘，图号使用译稿中的 `4-1`--`4-10` 和 `5-1`--`5-15` 标签。
- 对底本中动态共享内存参数单位、`BLOCK_WIDTH`/`BLOCK_SIZE` 命名不一致，以及练习 11 的主机/设备指针类型差异，均保留代码并添加译者注。

## 第六章本轮验收记录

- 已完成全局内存访问合并、DRAM 通道与 bank、向量加载和存储、共享内存 bank 冲突、线程粗化、循环展开、双缓冲及优化检查清单等内容的底本对照。
- 已核对本章 CUDA C++ 代码、公式、练习和参考文献，并将 13 幅原书概念图重绘为可编译的中文示意图。
- 已将第六章接入 `main.tex`，使用 Tectonic 完整编译整书；最终 PDF 共 133 页。
- 已检查构建日志；当前无未定义引用、缺失数学定界符、LaTeX 错误、overfull/underfull 或其他包警告。

## 第七章本轮验收记录

- 已完成一维/二维卷积、零填充及其他边界条件、ghost cell、常量内存与缓存、共享内存分块、halo cell、算术强度分析和 11 道练习的底本对照。
- 已核对本章 CUDA C++ 代码、公式和索引关系，并将原书图 7.1--7.15 重绘为可编译的中文概念示意图。
- 已对底本图 7.7、图 7.12 和图 7.15 中发现的指针/数组下标、共享内存元素类型、常量内存变量名及 halo 偏移问题添加译者注并修订代码示例。
- 已将第七章接入 `main.tex`，并置于第八章之前；使用 Tectonic 完整编译整书，最终 PDF 共 168 页，第七章从印刷页 126 开始，第八章从印刷页 146 开始。
- 已检查第七章图号、章节交叉引用、代码清单和关键图示版面；当前构建无未定义引用、缺失数学定界符、LaTeX 错误、overfull/underfull 或其他包警告。

## 第八章本轮验收记录

- 已完成有限差分与规则网格背景、基本模板扫掠、内存带宽分析、共享内存分块、线程粗化、寄存器分块、练习等内容的底本对照。
- 已核对 4 份 CUDA C++ 代码清单、公式和练习，并将 12 幅原书概念图重绘为可编译的中文示意图。
- 第八章已在 `main.tex` 中紧接第七章接入第二部分“并行模式”，章节编号和页码连续。
- 已对原书图 8.12 中 `inNext_s`/`inNext` 的编辑错误及正文图号误指添加译者注。
- 已用 Tectonic 完整编译整书，最终 PDF 共 168 页；目录、第二部分扉页、第八章全文和术语表均已完成渲染检查。
- 已检查构建日志；当前无未定义引用、缺失数学定界符、LaTeX 错误、overfull/underfull 或其他包警告。
