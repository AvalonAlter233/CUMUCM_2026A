# Q3/Q4 Typography and Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让问题三、四的 10 张论文图采用中文宋体、英文数字 Times New Roman，仅标题类文字加粗，并将普通坐标轴主网格适度加深。

**Architecture:** 保留两份现有绘图脚本结构，只调整共享的 Matplotlib 全局字体参数和 `_style_axis` 网格参数。测试直接检查 rcParams、生成图中文字对象的字重和网格线属性；热图排除逻辑保持不变。

**Tech Stack:** Python 3、Matplotlib、pytest、现有 Excel/JSON/NPZ 结果数据。

## Global Constraints

- 中文使用宋体类字体，英文和数字优先使用 `Times New Roman`，数学公式使用 `STIX`。
- 仅总标题、子图标题和坐标轴标题加粗；刻度、图例、注释与数据标签保持常规字重。
- 普通数据轴主网格使用 `#AAA5A8`、虚线、线宽 `0.55`、透明度 `0.70`，并位于数据下方。
- 问题三图三、问题四图三的热图轴及色条不显示网格，配套剖面轴显示主网格。
- 不改变模型、数据、图数、图名、配色、线型、版式和论文正文；只输出 600 dpi PNG。

---

### Task 1: Define typography and grid regression contracts

**Files:**
- Modify: `tests/test_problem3_plots.py`
- Modify: `tests/test_problem4_plots.py`

**Interfaces:**
- Consumes: `plotting.mpl.rcParams`, all five plot builders in each plotting module.
- Produces: regression assertions for font fallback order, title/label weights, normal-weight supporting text, and exact grid styling.

- [ ] **Step 1: Update publication-contract assertions**

In both test files, replace the current single-family assertions with:

```python
assert plotting.mpl.rcParams["font.family"][:2] == ["Times New Roman", "STSong"]
assert plotting.mpl.rcParams["mathtext.fontset"] == "stix"
assert plotting.mpl.rcParams["axes.titleweight"] == "bold"
assert plotting.mpl.rcParams["axes.labelweight"] == "bold"
```

- [ ] **Step 2: Add exact grid-style assertions**

Add a helper and use it on an ordinary data axis:

```python
def _assert_paper_grid(axis):
    xline = next(line for line in axis.get_xgridlines() if line.get_visible())
    yline = next(line for line in axis.get_ygridlines() if line.get_visible())
    for line in (xline, yline):
        assert line.get_color().upper() == "#AAA5A8"
        assert line.get_linestyle() == "--"
        assert line.get_linewidth() == pytest.approx(0.55)
        assert line.get_alpha() == pytest.approx(0.70)
```

Retain the existing assertions that the two heatmap axes have no visible grid and their companion profile axes do.

- [ ] **Step 3: Extend text-weight assertions**

For every figure builder, assert the total title, each subplot title, and each x/y label are bold. For a representative figure, assert tick labels, legend text, annotations, and data labels are not bold:

```python
assert all(axis.title.get_fontweight() == "bold" for axis in figure.axes)
assert all(axis.xaxis.label.get_fontweight() == "bold" for axis in figure.axes)
assert all(axis.yaxis.label.get_fontweight() == "bold" for axis in figure.axes)
assert all(label.get_fontweight() == "normal" for axis in figure.axes for label in axis.get_xticklabels())
assert all(text.get_fontweight() == "normal" for legend in figure.legends for text in legend.get_texts())
```

- [ ] **Step 4: Run focused tests to verify RED**

Run: `python -m pytest tests/test_problem3_plots.py tests/test_problem4_plots.py -q`

Expected: failures because `font.family` is still only `STSong`, axis titles/labels lack the new global bold settings, and the grid still uses `#C8C3C5`, `0.45`, `0.55`.

### Task 2: Implement the approved typography and grid style

**Files:**
- Modify: `问题三_绘图.py`
- Modify: `问题四_绘图.py`
- Test: `tests/test_problem3_plots.py`
- Test: `tests/test_problem4_plots.py`

**Interfaces:**
- Consumes: `AVAILABLE_CJK_FONTS: list[str]`, `_style_axis(axis, show_grid=True)`.
- Produces: a glyph-fallback font family list and consistent title/label/grid defaults for every figure builder.

- [ ] **Step 1: Restrict Chinese fallbacks to Song-style fonts**

Use this candidate tuple in both plotting scripts:

```python
CJK_FONT_CANDIDATES = (
    "STSong", "SimSun", "Source Han Serif SC", "Noto Serif CJK SC",
)
```

- [ ] **Step 2: Configure mixed typography and title weights**

Change the relevant rcParams in both plotting scripts to:

```python
"font.family": ["Times New Roman", *AVAILABLE_CJK_FONTS],
"mathtext.fontset": "stix",
"axes.titleweight": "bold",
"axes.labelweight": "bold",
```

Keep `_set_top_title(..., fontweight="bold")`; do not add bold settings to legends, ticks, annotations, or data labels.

- [ ] **Step 3: Apply the approved grid parameters**

In both `_style_axis` functions, use:

```python
axis.grid(
    True,
    which="major",
    color="#AAA5A8",
    linestyle="--",
    linewidth=0.55,
    alpha=0.70,
)
```

Keep `axis.set_axisbelow(True)` and the existing `show_grid=False` calls for heatmap axes.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `python -m pytest tests/test_problem3_plots.py tests/test_problem4_plots.py -q`

Expected: all focused tests pass with no missing-glyph warnings.

### Task 3: Regenerate and verify the final figures

**Files:**
- Modify: `figures/问题三/*.png`
- Modify: `figures/问题四/*.png`

**Interfaces:**
- Consumes: existing result workbooks, diagnostics, internal-field file, and updated plot builders.
- Produces: exactly five 600 dpi PNG files for each problem.

- [ ] **Step 1: Regenerate both figure sets**

Run: `python 问题三_绘图.py`

Run: `python 问题四_绘图.py`

Expected: each script reports five successfully generated PNG figures.

- [ ] **Step 2: Run full automated verification**

Run: `python -m pytest -q`

Run: `git diff --check`

Expected: all tests pass; `git diff --check` reports no whitespace errors.

- [ ] **Step 3: Audit output inventory**

Verify that `figures/问题三` and `figures/问题四` each contain exactly five `.png` files and no `.pdf`, `.svg`, `.tif`, or `.tiff` files; verify each PNG reports 600 dpi metadata within format tolerance.

- [ ] **Step 4: Visually inspect representative outputs**

Inspect both threshold figures, both heatmap figures, and at least one robustness figure. Confirm Chinese serif appearance, Times-style Latin/numerals, bold total/subplot/axis titles, normal-weight supporting text, darker grids below the data, heatmap grid exclusion, and absence of clipping or乱码.
