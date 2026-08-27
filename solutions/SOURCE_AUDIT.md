# 来源核验与公开副本审计

审计日期：2026-08-26。

## 口径

- 125 道主问题均有至少一条 `\sourcecheck`，共 126 条、27 个唯一 URL。
- 来源用于核验定义、算法不变量、API 语义和工程边界；题解采用独立推导与短引/转述，
  不以网络材料替代解答。
- 优先级依次为：标准/官方文档、论文作者主页、大学或政府机构仓储、项目官方源代码、DOI
  书目页。
- 不绕过付费墙、登录、验证码、robots 限制或其他访问控制。找不到合法公开全文时只保留 DOI
  元数据；不会把受限出版物抓取或提交到仓库。

## 在线可用性

最终 URL 审计中，27 个唯一地址均返回 HTTP 200 或 202。审计 CSV 位于仓库外：

`D:\translation\tmp\pmpp5-solutions-url-audit\url-audit.csv`

唯一地址包括：

- NVIDIA 官方 CUDA Programming Guide、Best Practices Guide、Runtime API、浮点指南、
  Nsight Compute、cuDNN、CUTLASS、CCCL/CUB、CUDA Samples 和 RAPIDS cuGraph；
- MPI Forum 的 MPI 4.1 标准与 IEEE 754-2019 标准页面；
- FlashAttention 与 online softmax 的作者公开预印本；
- Merge Path 作者主页公开版、Mines Paris 的 supernode partitioning 机构仓储版；
- NASA 的 Saad JDS 报告、NVIDIA Research 的 SpMV/scan 论文公开版；
- Smith--Waterman、Kogge--Stone、Brent--Kung 和稳定并行归并的 DOI 书目记录；
- NVIDIA CUDA Samples、cuda-convnet2 等项目的官方开源代码。

## 仓库外公开副本

为避免链接失效影响复核，已把合法公开的核验副本下载到：

`D:\translation\tmp\pmpp5-solutions-sources`

其中包括 CUDA Programming/Best Practices/浮点/Hopper 指南、MPI 4.1、FlashAttention、
Bell--Garland SpMV、Merrill 单遍扫描、Saad JDS、ModernGPU merge 教程、NVIDIA 卷积样例、
cuda-convnet2 代码，以及以下两份精确公开原文：

- Green、McColl、Bader，GPU Merge Path 作者版，10 页，SHA-256
  `709245D66C5F2BCA0B1F16AD1C051F96261FFB544CA6C20D44FF8F6DF02F6BBE`；
- Irigoin、Triolet，Supernode Partitioning，Mines Paris A-179，11 页，SHA-256
  `19AC498D1470EBD894354FB6A8B43D40937306AC31C264D67F4FA76A997870C9`。

下载目录、抽取文本、HTTP 响应头和散列均为验收材料，不属于题解源码，不会提交。

## 受限来源

OpenAlex 对稳定并行归并 DOI `10.1007/978-3-642-33518-1_25` 与 Smith--Waterman DOI
`10.1016/0022-2836(81)90087-5` 标记为 closed；IEEE 两篇前缀网络论文也仅保留 DOI
权威书目信息。题解没有声称下载这些受限全文，也没有从非授权镜像摘录内容。
