# PMPP 插图批量提取

本工具采用两段式流程：

1. `detect` 根据 `FIGURE x.y` 图题自动生成候选裁剪区域和浏览画廊。
2. 人工检查并修正坐标后，将确认结果加入 `figure-crops.json`，再用 `extract`
   稳定、批量地生成正式图片。

自动检测只用于减少定位工作，不应未经检查直接覆盖译稿图片。复合矢量图、跨栏图、
无图题插图和正文相邻很近的图片仍可能需要调整。

检测器会排除页眉，并在同一页有连续多幅图时，以前一幅图的图题作为后一幅图的
上边界。需要微调时可使用 `--minimum-top`、`--vertical-gap` 和
`--caption-clearance` 参数。

## 初始化

在项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

此后始终使用项目内的 Python：

```bash
.venv/bin/python tools/figure_extractor.py --help
```

## 重新生成已确认图片

生成清单中的全部图片：

```bash
.venv/bin/python tools/figure_extractor.py extract --force
```

只生成指定图片：

```bash
.venv/bin/python tools/figure_extractor.py extract --force 2-5 3-1
```

先输出到临时目录，不覆盖正式图片：

```bash
.venv/bin/python tools/figure_extractor.py extract \
  --force \
  --output-root /tmp/pmpp-figures \
  --gallery /tmp/pmpp-figures/index.html

open /tmp/pmpp-figures/index.html
```

## 自动扫描某一章

例如扫描第三章：

```bash
.venv/bin/python tools/figure_extractor.py detect \
  --chapter 3 \
  --output-dir build/figure-candidates/ch3

open build/figure-candidates/ch3/index.html
```

该目录包含：

- `N-M.png`：自动裁剪预览；
- `candidates.json`：候选页码和裁剪坐标；
- `index.html`：并排检查所有候选图的浏览页面。

## 确认并修正候选

当前正式图片以 600 DPI 输出；清单中的已确认坐标仍使用 300 DPI 像素，
并以 PDF 的 CropBox（正文裁切框）左上角为原点。`figure-crops.json` 中的
`coordinate_dpi` 记录坐标基准，`dpi` 记录正式输出分辨率：

```json
{
  "id": "3-1",
  "page": 76,
  "crop": {
    "x": 480,
    "y": 310,
    "width": 1440,
    "height": 830
  },
  "output": "figs/ch3/3-1.png"
}
```

字段含义：

- `page`：PDF 物理页，从 1 开始；
- `coordinate_dpi`：清单中裁剪坐标的基准 DPI；
- `dpi`：正式 PNG 图片的输出 DPI；
- `x`、`y`：以 `coordinate_dpi` 为基准的裁剪区域左上角；
- `width`、`height`：以 `coordinate_dpi` 为基准的裁剪区域尺寸；
- `output`：项目根目录下的正式输出路径。

工具会先根据 `coordinate_dpi` 将清单坐标换算为 PDF point，再按照 `dpi`
渲染 PNG。因此提高输出 DPI 不会改变已经确认的物理裁剪区域。

本书的 MediaBox 比 CropBox 四周各宽 72pt。直接使用 `pdftocairo` 渲染整张
MediaBox 后量出的 300 DPI 坐标，需要把 `x`、`y` 各减去 300，才能写入本工具的
清单；宽度和高度保持不变。使用 `detect` 生成的候选坐标已经是 CropBox 坐标，
无需转换。

在画廊中检查候选后，把正确条目复制到
`tools/figure-crops.json`。坐标不准确时，只需要修改这四个数，然后运行：

```bash
.venv/bin/python tools/figure_extractor.py extract --force 3-1
```

调整原则：

- 保留图内标题、坐标、代码、箭头和说明文字；
- 去掉原书的 `FIGURE x.y` 和英文图注；
- 去掉页眉及图下方正文；
- 对图 2-5 这样的复合图，代码和右下角示意图应作为一个整体保留。

## 在译稿中引用

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figs/ch3/3-1.png}
  \caption{译文中的中文图注}
  \label{fig:3-1}
\end{figure}
```

无编号的 note 插图不要使用 `figure` 和 `caption`，直接放入说明框：

```latex
\begin{center}
  \includegraphics[width=0.7\textwidth]{figs/ch2/RGB.png}
\end{center}
```

## 最终检查

```bash
tectonic -X compile --keep-logs main.tex
```

确认编译退出码为 `0`，并检查最终 PDF 中图的宽度、清晰度和浮动位置。
