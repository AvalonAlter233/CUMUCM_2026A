# Q1/Q2 Match Q3/Q4 Figure Style Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将问题一、二的 10 张论文图统一为问题三、四已确认的字体与网格样式。

**Architecture:** 仅在 `问题一_绘图.py` 与 `问题二_绘图.py` 内替换字体配置、标题字体和 `_style_axis` 网格策略，不抽取公共模块。热图轴显式关闭网格，普通数据轴默认显示主网格。

**Tech Stack:** Python 3、Matplotlib、现有 Excel 结果数据。

## Global Constraints

- 总标题、子图标题、坐标轴标题使用 `STZhongsong` 并加粗。
- 正文中文使用宋体，纯数字刻度使用 `Times New Roman`，数学公式使用 `STIX`。
- 普通数据轴主网格使用 `#AAA5A8`、虚线、线宽 `0.55`、透明度 `0.70`。
- 两组热图轴及色条不显示网格。
- 不改变数据、模型、配色、布局、图名和输出数量。
- 按用户要求不运行测试，只重绘并目视核验。

---

### Task 1: Apply the approved style locally

**Files:**
- Modify: `问题一_绘图.py`
- Modify: `问题二_绘图.py`

**Interfaces:**
- Consumes: existing Matplotlib axes and figure builders.
- Produces: `_style_axis(axis, show_grid=True)` with the same typography and grid behavior as problems three and four.

- [ ] Replace the sans-serif font configuration with Song-style serif configuration and STIX math text.
- [ ] Add `HEADING_FONT = "STZhongsong"` with an installed-font fallback.
- [ ] Update `_style_axis` to apply the approved grid, Times New Roman numeric ticks, and bold heading font to axis titles and labels.
- [ ] Update all total titles and explicit medium-weight subplot titles to use the heading font and bold weight.
- [ ] Route heatmap axes through `_style_axis(..., show_grid=False)` and keep their colorbars grid-free.

### Task 2: Regenerate and inspect outputs

**Files:**
- Modify: `figures/问题一/*.png`
- Modify: `figures/问题二/*.png`

**Interfaces:**
- Consumes: updated plotting scripts and existing result files.
- Produces: five 600 dpi PNG files for each problem.

- [ ] Run `python 问题一_绘图.py` and `python 问题二_绘图.py`.
- [ ] Confirm each output directory contains exactly five PNG files and no other formats.
- [ ] Inspect representative ordinary, heatmap, and validation figures for title weight, grid depth, missing glyphs, clipping, and heatmap grid exclusion.
