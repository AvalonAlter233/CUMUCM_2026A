# Q3/Q4 Grid and Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复问题三图二的 $t_*$ 遮挡，并为问题三、四全部非热图数据轴添加轻量网格，同时只输出 PNG。

**Architecture:** 两份绘图脚本各自保留现有结构，在 `_style_axis` 上增加统一的网格开关。网格默认开启，两个 `pcolormesh` 热图轴显式关闭；问题三阈值标注改用轴坐标定位和白底框避免遮挡。

**Tech Stack:** Python 3、Matplotlib、NumPy、pytest、现有 Excel/JSON 数据。

## Global Constraints

- 不改动模型、输入数据、图名、图数、配色和论文正文。
- 最终仅导出 600 dpi PNG，不生成 PDF、SVG 或灰度预览。
- 热图面板及色条无网格；其余数据坐标轴均显示轻量主网格。

---

### Task 1: Add failing visual-contract tests

**Files:**
- Modify: `tests/test_problem3_plots.py`
- Modify: `tests/test_problem4_plots.py`

**Interfaces:**
- Consumes: `plot_threshold_evidence(data)`, `plot_field_evolution(data)`, `plot_moving_field(data)` and all existing figure builders.
- Produces: regression assertions for grid visibility, heatmap exclusion, annotation coordinate system/bounding box, and PNG-only output contract.

- [ ] **Step 1: Write the failing tests**

Add helpers that inspect major gridline visibility. Assert every ordinary data axis has visible x/y gridlines, both heatmap axes have none, and the problem-three annotation uses `axes fraction` text coordinates with a non-null bounding-box patch.

Use this helper in both test modules:

```python
def _has_visible_major_grid(axis):
    return (
        any(line.get_visible() for line in axis.get_xgridlines())
        and any(line.get_visible() for line in axis.get_ygridlines())
    )
```

For problem three, build all five figures, assert both axes of figures 1, 2, 4, and 5 have grids, assert figure 3 axis 0 has no grid and axis 1 has a grid, then locate the annotation with `"t_*" in text.get_text()` and assert `xycoords == "data"`, `_textcoords == "axes fraction"`, and `get_bbox_patch() is not None`. For problem four, apply the same grid assertions to figures 1, 2, 4, and 5, with figure 3 axis 0 excluded and axis 1 included.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_problem3_plots.py tests/test_problem4_plots.py -q`

Expected: failures because `_style_axis` disables grids and the annotation lacks a bounding box/axis-coordinate positioning.

- [ ] **Step 3: Commit the regression tests**

Run: `git add -- tests/test_problem3_plots.py tests/test_problem4_plots.py && git commit -m "test: define q3 q4 plot grid contract"`

### Task 2: Implement grid policy and annotation avoidance

**Files:**
- Modify: `问题三_绘图.py`
- Modify: `问题四_绘图.py`

**Interfaces:**
- Consumes: existing Matplotlib axes from each plot builder.
- Produces: `_style_axis(axis: plt.Axes, *, show_grid: bool = True) -> None`.

- [ ] **Step 1: Implement the minimal style change**

Set axes below data; when `show_grid=True`, enable the major grid with `color="#C8C3C5"`, `linestyle="--"`, `linewidth=0.45`, `alpha=0.55`; otherwise disable it. Call `_style_axis(axes[0], show_grid=False)` only for each heatmap panel and keep default styling for the companion profile axis.

```python
def _style_axis(axis: plt.Axes, *, show_grid: bool = True) -> None:
    axis.set_axisbelow(True)
    axis.grid(
        show_grid,
        which="major",
        color="#C8C3C5",
        linestyle="--",
        linewidth=0.45,
        alpha=0.55,
    )
    axis.tick_params(length=3, width=0.7)
    for name in ("left", "bottom"):
        axis.spines[name].set_color(COLOR_DARK)
        axis.spines[name].set_linewidth(0.75)
```

- [ ] **Step 2: Move the problem-three annotation**

Use `xycoords="data"`, `textcoords="axes fraction"`, `xytext=(0.12, 0.70)`, a white bounding box with `alpha=0.9`, and a coral connector line to the threshold intersection.

```python
axes[1].annotate(
    f"$t_*= {target:.4f}$ h",
    xy=(target, CRITICAL_MOISTURE),
    xycoords="data",
    xytext=(0.12, 0.70),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "-", "color": COLOR_CORAL, "lw": 0.8},
    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    color=COLOR_CORAL,
    fontsize=7.5,
)
```

- [ ] **Step 3: Run focused tests to verify GREEN**

Run: `python -m pytest tests/test_problem3_plots.py tests/test_problem4_plots.py -q`

Expected: all focused tests pass.

- [ ] **Step 4: Commit implementation**

Run: `git add -- 问题三_绘图.py 问题四_绘图.py && git commit -m "fix: clarify q3 threshold and add plot grids"`

### Task 3: Regenerate and inspect PNG outputs

**Files:**
- Modify: `figures/问题三/*.png`
- Modify: `figures/问题四/*.png`

**Interfaces:**
- Consumes: existing result workbooks, diagnostics, and plotting scripts.
- Produces: exactly five PNG files per problem at 600 dpi.

- [ ] **Step 1: Regenerate figures**

Run: `python 问题三_绘图.py`

Run: `python 问题四_绘图.py`

Expected: each command reports five generated 600 dpi PNG files.

- [ ] **Step 2: Verify output formats**

Run a PowerShell inventory of `figures/问题三` and `figures/问题四` and assert that each contains five `.png` files and no `.pdf`, `.svg`, `.tif`, or `.tiff` files.

- [ ] **Step 3: Run visual/file audits**

Run the available figure checks, then open problem-three figure 2 and both field-evolution figures to verify the annotation, grids, heatmap exclusions, typography, and cropping at final size.

- [ ] **Step 4: Commit regenerated PNGs**

Run: `git add -- figures/问题三 figures/问题四 && git commit -m "chore: regenerate q3 q4 png figures"`

### Task 4: Remove process-only files and run final verification

**Files:**
- Delete: `docs/superpowers/specs/2026-09-12-q3-q4-grid-and-annotation-design.md`
- Delete: `docs/superpowers/plans/2026-09-12-q3-q4-grid-and-annotation.md`
- Delete: `使用指南.md`

**Interfaces:**
- Consumes: completed code, tests, and PNG outputs.
- Produces: clean project deliverables without process-only documents.

- [ ] **Step 1: Remove only the three process-only files**

Use `apply_patch` to delete the exact files listed above; preserve all original files.

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest -q`

Expected: all tests pass with zero failures.

- [ ] **Step 3: Verify the working tree and deliverables**

Run: `git diff --check`, `git status --short`, PNG inventory, and targeted image inspection. Confirm no process-only files remain and no output format other than PNG was added.

- [ ] **Step 4: Commit cleanup if needed**

Run: `git add -A -- docs/superpowers 使用指南.md && git commit -m "chore: remove temporary process docs"`
