# Portable Plotting Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让问题一至四的绘图脚本脱离 Codex skills，在普通 Python 环境中直接生成图片。

**Architecture:** 从四个脚本中删除 skills 路径注入、外部对齐审计与对应临时代码。保留 Matplotlib 原生绘制和紧边界 PNG 导出，不新增本地审计模块或可选回退。

**Tech Stack:** Python 3、Matplotlib、NumPy、openpyxl、项目内求解模块。

## Global Constraints

- 不改变模型、数据、字体、网格、布局、图名和输出数量。
- 不复制或可选导入 Codex skills 中的源码。
- 按用户此前要求不运行 pytest；通过静态搜索和四个脚本实际出图验证。

---

### Task 1: Remove all skills dependencies

**Files:**
- Modify: `问题一_绘图.py`
- Modify: `问题二_绘图.py`
- Modify: `问题三_绘图.py`
- Modify: `问题四_绘图.py`

**Interfaces:**
- Consumes: existing figure builders.
- Produces: `save_publication_figure(figure, base_name)` implemented only with Matplotlib.

- [ ] Delete `sys` and `TemporaryDirectory` imports where they are only used by the skills audit.
- [ ] Delete `NATURE_FIGURE_SCRIPTS`, `sys.path.insert`, and `require_matplotlib_panel_alignment` imports.
- [ ] Remove the `exclude_axes` argument and the entire `TemporaryDirectory` audit block from each `save_publication_figure`.
- [ ] Remove `_alignment_exclude_axes`, `_alignment_row_groups`, `_extra_qa_axes` assignments and update every save call to pass only `figure, base_name`.

### Task 2: Verify standalone reproduction

**Files:**
- Regenerate: `figures/问题一/*.png`
- Regenerate: `figures/问题二/*.png`
- Regenerate: `figures/问题三/*.png`
- Regenerate: `figures/问题四/*.png`

**Interfaces:**
- Consumes: four cleaned plotting scripts.
- Produces: 20 PNG files at approximately 600 dpi.

- [ ] Search all `*_绘图.py` files and confirm no `.codex`, `skills`, `audit_panel_alignment`, `TemporaryDirectory`, or `_alignment_` remains.
- [ ] Run the four plotting scripts directly with Python and require exit code 0.
- [ ] Confirm every problem directory contains exactly five PNG files and no other formats.
- [ ] Inspect representative ordinary and heatmap figures for missing glyphs, clipping, font changes, and grid changes.
