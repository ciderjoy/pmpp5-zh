# 《大规模并行处理器编程（第 5 版）》中文学习译稿

这是基于 _Programming Massively Parallel Processors: A Hands-on Approach, Fifth Edition_ 的个人学习译稿工程。原书作者、出版商及其他权利人保留全部版权。本工程仅用于个人学习、术语讨论和排版实验；未经权利人授权，不应公开发布或商业传播。


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
