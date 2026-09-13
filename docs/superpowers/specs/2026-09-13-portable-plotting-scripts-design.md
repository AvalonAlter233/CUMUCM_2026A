# 四问绘图脚本可复现性改造设计

## 目标

- 四个绘图脚本不再访问 `~/.codex/skills`，也不导入 `audit_panel_alignment`。
- 仅依赖常规 Python 包与项目内求解模块，在未安装 Codex skills 的电脑上可以出图。
- 保持现有字体、网格、数据、布局、图名和 PNG 输出效果不变。

## 方案

采用最简方案：删除四个脚本中的 skills 路径注入、对齐审计导入及临时审计文件逻辑。`save_publication_figure` 保留 Matplotlib 原生的 `figure.canvas.draw()` 与 `figure.savefig(..., bbox_inches="tight")`，继续负责布局渲染和紧边界导出。

同时删除只服务于外部审计的 `sys`、`TemporaryDirectory`、`exclude_axes` 参数和 `_alignment_*` 属性，避免留下无效兼容代码。不复制审计模块，不设置可选 skills 回退。

## 验证

1. 搜索四个绘图脚本，确认不存在 `.codex`、`skills`、`audit_panel_alignment`、`TemporaryDirectory` 或 `_alignment_*`。
2. 依次运行四个绘图脚本，确认均能直接生成各自的 5 张 PNG。
3. 核对四个目录均只有 5 张约 600 dpi PNG，并抽查图片外观没有变化或裁切。

## 完成标准

- 四个脚本仅使用 Python/第三方常规包和项目内模块。
- 四个脚本运行成功，共生成 20 张 PNG。
- 不修改模型、数据和既有绘图样式。
