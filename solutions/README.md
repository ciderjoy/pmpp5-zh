# 《大规模并行处理器编程（第 5 版）》习题详解

本目录是一份与根目录中文译稿相互独立的习题详解。覆盖第 1--24 章与附录 A--C；
没有习题的第 1、22、24 章及附录 B、C 仍保留标题并标明“本章无习题”。

## 编译

在仓库根目录执行，且把构建与缓存放在仓库外：

```powershell
$env:TECTONIC_CACHE_DIR='D:\translation\tmp\pmpp5-solutions-tectonic-cache-fresh'
tectonic --outdir D:\translation\tmp\pmpp5-solutions-build-fresh `
  --keep-logs --keep-intermediates solutions\solutions.tex
```

输出文件为 `D:\translation\tmp\pmpp5-solutions-build-fresh\solutions.pdf`。

## 覆盖统计

全书共 125 道主问题，已完成 125 道，未完成 0 道；另有 124 个独立小问。逐章题目数、
小题数和验收记录见 `SOLUTIONS_STATUS.md`。“小题”只统计题目中单独标号且需要分别作答的
问项；选择题选项以及仅用于引出下级问项的容器标签不重复计数。

每道主问题均包含“题目、解题思路、详细解答、结论、进一步讨论、变体练习与答案要点、
工程实践、联网核验与对抗审查”。125 个“进一步讨论”各由 4 个连续讲解段落组成，共 500 段；
每段从问题情境出发，解释机制与前因后果，再用公式、例子或反例落地，最后联系性能测量、
体系结构、数值或并发风险及可继续研究的问题。原始 LaTeX 正文中每段为 220--394 个字符，
平均 257.4 个字符，且均至少包含 3 个完整句子；不是把同一组模板机械复用到各题。题解正文
共有 126 条来源核验
（第 16 章第 5 题使用两条），
优先采用 NVIDIA、MPI Forum、IEEE、NASA、作者主页和机构仓储等一手来源。公开资料只作
独立核验与短引/转述，未把受限出版物或长段原文复制进仓库；下载的核验副本位于仓库外。
27 个唯一 URL 的可用性、公开副本和访问边界见 `SOURCE_AUDIT.md`。

## 可执行代码

`solutions/code/` 收录 27 个独立 CUDA 程序、6 个普通 C++/MPI 程序及逐题映射。完整构建：

```powershell
solutions\code\build.ps1 `
  -BuildDirectory D:\translation\tmp\pmpp5-solutions-code-vs2022-fresh `
  -CudaArchitecture 75
```

脚本先构建并运行 CTest（安装 Microsoft MPI 时包含两秩 halo 测试），再逐个调用 `nvcc`
编译链接所有 CUDA 程序，最后运行每个程序的 `--cpu-only` 参考路径。第 16 章 DPX 示例还可
单独用 `sm_90` 编译以检查 Hopper 条件分支。脚本在 Windows 上固定使用 VS 2022 x64，
并拒绝复用生成器、平台或 VS 实例不匹配的 CMake 缓存；`-SkipCTest`、`-SkipCudaRun` 可分别
跳过 CTest 或 CUDA 包装程序运行，兼容参数 `-SkipRun` 同时跳过二者。完整参数见
`solutions/code/README.md`。

## 验证边界

数值结果使用独立计算复核；CUDA 代码检查边界、索引、同步、竞态和内存生命周期。
本机构建环境使用 CUDA Toolkit 13.3 的 `nvcc`、MSVC 14.44 和 Microsoft MPI 完成编译、
链接及 CPU/MPI 实际运行。本机没有 NVIDIA GPU，因此没有声称设备内核已经运行；设备结果
由 CPU 参考、边界样例、静态并发审查和实际 `nvcc` 编译共同验证。

最终验收构建位于仓库外的
`D:\translation\tmp\pmpp5-solutions-release-narrative-final-20260826\solutions.pdf`：连续两次
Tectonic 编译后 `.aux` 与 `.toc` 散列稳定，共 286 页。全部 286 页均通过 18 张接触表逐页
检查，并对标题、目录、TikZ、长代码、矩阵/表格、FlashAttention、MPI 和浮点页面作原分辨率
复核。日志严格扫描为 0 个诊断命中，中文字体均已嵌入。

代码方面，基准验收目录 `D:\translation\tmp\pmpp5-solutions-code-final3-20260826` 曾完成
CTest 6/6、CUDA 27/27 编译链接及 27/27 `--cpu-only` 实际运行；DPX 示例另以 `sm_90`
编译通过。最终源码在新默认目录 `D:\translation\tmp\pmpp5-solutions-code-vs2022` 再次
fresh build，6 个 C++/MPI 目标及 27 个 CUDA 程序全部编译链接。用户关闭 Smart App Control
后，系统查询为 `SmartAppControlState=Off`、策略状态为 0；该目录的 CTest 6/6 全部通过，
27 个 CUDA 包装程序的 `--cpu-only` 路径也逐一实际运行，27/27 退出码为 0、标准错误为空。
本机没有 NVIDIA GPU，因此不能把这些 CPU 参考运行表述为 GPU 设备运行。
