# 论文概念图工作区

概念图相关的图稿、绘图源码、字体、参考原图和审查记录集中维护在本目录。
专用分支：`docs/concept-figure`。

| 目录 | 内容 | 入口 |
|---|---|---|
| `classifier_guidance_method/` | 本项目的方法图，当前为统一原图字体、带 Strong / Weak 小网络的 v2 | [可编辑 PPT](classifier_guidance_method/classifier_guidance_method_v2.pptx)、[使用说明](classifier_guidance_method/README.md) |
| `advfd_editable/` | AdvFD Figure 3 原图、早期可编辑重绘及其来源记录 | [参考说明](advfd_editable/README.md) |

## 当前图稿

- 联合学习弱头与系数：[SVG](classifier_guidance_method/method_joint_v2.svg) · [PDF](classifier_guidance_method/method_joint_v2.pdf) · [PNG](classifier_guidance_method/method_joint_v2.png)。
- 仅学习系数：[SVG](classifier_guidance_method/method_schedule_v2.svg) · [PDF](classifier_guidance_method/method_schedule_v2.pdf) · [PNG](classifier_guidance_method/method_schedule_v2.png)。
- 配套文字：[图注与符号](classifier_guidance_method/CAPTION.md) · [自审记录](classifier_guidance_method/REVIEW.md)。

修改版全部使用 Comic Sans MS；参考底稿保留早期的 Comic Neue / DejaVu Serif 重绘，供追溯使用。两者的字体和许可分别存放在各自的 `fonts/` 中。

## 修改与重建

可直接修改 PPT 中的文字和图形。需要重新生成全部格式时，修改 `classifier_guidance_method/build_figure.py` 中的布局和模块，或修改 `drawing.py` 中的绘制逻辑；`embedded_fonts.py` 负责 PPT 字体嵌入。运行生成脚本会覆盖导出文件，手工修改的 PPT 请另存后再重建。

在仓库根目录执行：

```bash
python -m pip install -r paper/figures/requirements.txt
python paper/figures/classifier_guidance_method/build_figure.py
```

若需要重建 AdvFD 参考底稿：

```bash
python paper/figures/advfd_editable/build_figure.py
```

Git 跟踪绘图源码、最新 v2 图稿、对象场景、审查记录及必要参考资源。自动生成的重复文件、分发 ZIP、字体的本机路径配置和 Python 缓存由本目录的 `.gitignore` 排除。

`review_checks.json` 保留交付时的检查及源文件哈希；重新绘图会更新对象检查及源文件哈希。完整的交付检查经过独立办公软件渲染和人工目视核对，详见自审记录。
