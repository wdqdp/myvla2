"""Draw the paper inference overview as editable SVG, PDF and PNG.

Run: MPLCONFIGDIR=/tmp/myvla2_mpl python 论文准备/draw_inference_framework.py
Coordinates and wording are intentionally explicit to make paper edits easy.
"""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/myvla2_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/myvla2_figure_cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MPath
import numpy as np


OUT = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
REG = font_manager.FontProperties(fname=FONT)
HEAVY = font_manager.FontProperties(fname=BOLD)
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})

C = {
    "ink": "#223449", "muted": "#657387", "border": "#D5DEE8",
    "blue": "#3979B5", "blue_bg": "#EEF5FC",
    "teal": "#258B83", "teal_bg": "#EBF7F3",
    "orange": "#C68536", "orange_bg": "#FFF4E5",
    "purple": "#8064A5", "purple_bg": "#F4F0FA",
    "red": "#B65E60", "white": "#FFFFFF",
}

fig, ax = plt.subplots(figsize=(22, 12.4))
fig.patch.set_facecolor("white")
ax.set(xlim=(0, 24), ylim=(0, 13.5))
ax.axis("off")
fig.subplots_adjust(left=.018, right=.982, bottom=.025, top=.985)


def text(x, y, s, size=12, color=None, bold=False, ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color or C["ink"],
                   fontproperties=HEAVY if bold else REG, ha=ha, va=va,
                   linespacing=1.5, zorder=6, **kw)


def box(x, y, w, h, title=None, subtitle=None, face="white", edge=None,
        title_size=13, sub_size=10.5, dashed=False, lw=1.25, radius=.13):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.02,rounding_size={radius}",
                       facecolor=face, edgecolor=edge or C["border"], linewidth=lw,
                       linestyle=(0, (5, 3)) if dashed else "-", zorder=2)
    ax.add_patch(p)
    if title and subtitle:
        text(x+w/2, y+h*.65, title, title_size, bold=True)
        text(x+w/2, y+h*.29, subtitle, sub_size, color=C["muted"])
    elif title:
        text(x+w/2, y+h/2, title, title_size, bold=True)
    return p


def arrow(points, color=None, dashed=False, lw=1.5, head=12):
    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(points)-1)
    a = FancyArrowPatch(path=MPath(points, codes), arrowstyle="-|>",
                        mutation_scale=head, linewidth=lw,
                        color=color or C["muted"],
                        linestyle=(0, (4, 3)) if dashed else "-",
                        capstyle="round", joinstyle="round", zorder=4)
    ax.add_patch(a)


def label(x, y, s, color=None, size=10, **kwargs):
    text(x, y, s, size=size, color=color or C["muted"],
         bbox={"facecolor":"white", "edgecolor":"none", "pad":1.3}, **kwargs)


# Overall title: this is a method figure, not a software deployment diagram.
text(.45, 13.06, "动态触觉驱动的故障诊断与闭环恢复", 23, bold=True, ha="left")
text(.45, 12.57, "推理框架草稿  ·  触觉语义 → 故障理解与历史恢复推理 → 异步执行闭环", 12,
     color=C["muted"], ha="left")

# (a) The independent tactile captioner.
box(.4, 5.25, 4.65, 6.85, face="#FAFCFE")
text(.68, 11.75, "(a) 动态触觉元描述", 15, bold=True, ha="left")
text(2.72, 11.26, "双指触觉 · 最近 30 帧", 11.5, color=C["muted"])

# Small synthetic contact patches: explicitly an illustration, not measured data.
for j, cx in enumerate([1.07, 2.98]):
    for k in [2, 1, 0]:
        ax.add_patch(Rectangle((cx + .065*k, 9.92 + .065*k), 1.05, .8,
                               facecolor="#F4E7D5", edgecolor="#CBAE87", lw=.8, zorder=3))
    xx, yy = np.meshgrid(np.arange(7), np.arange(5))
    zz = np.exp(-((xx-3-j*.35)**2/5 + (yy-2)**2/2.2))
    ax.scatter(cx+.10+xx.ravel()*.14, 10.0+yy.ravel()*.15,
               c=zz.ravel(), cmap="YlOrBr", s=16, vmin=0, vmax=1,
               edgecolors="none", zorder=4)
    text(cx+.53, 9.68, "左指" if j == 0 else "右指", 10)
arrow([(2.72, 9.44), (2.72, 9.12)], C["teal"])
box(.88, 8.35, 3.69, .76, "逐帧 CNN → 时序 TCN", face=C["teal_bg"], edge=C["teal"], title_size=12)
arrow([(2.72, 8.32), (2.72, 8.0)], C["teal"])
box(.88, 7.3, 3.69, .68, "元字段分类与文本化", face=C["teal_bg"], edge=C["teal"], title_size=12)
arrow([(2.72, 7.27), (2.72, 6.96)], C["teal"])
box(.74, 5.66, 3.98, 1.28, "Touch 元描述文本", "接触面积 / 受力 / 旋转", face="white", edge=C["teal"], title_size=13)

# (b) One shared model with task-specific calls, not several parallel VLMs.
box(5.58, 5.25, 17.98, 6.85, face="#FCFCFE", edge=C["purple"], dashed=True, lw=1.5)
text(5.9, 11.75, "(b) 共享 π0.5 模型：按任务调用同一主干", 15, bold=True, ha="left")

box(6.05, 10.12, 4.55, .99, "Front / Wrist 图像", "SigLIP 视觉编码", face=C["blue_bg"], edge=C["blue"], title_size=12)
box(10.96, 10.12, 3.17, .99, "当前 qpos + Task", "状态离散化与文本编码", face=C["blue_bg"], edge=C["blue"], title_size=11.5, sub_size=10)
box(14.49, 10.12, 8.56, .99, "按任务选用上下文", "Touch / H100→11 运动历史 / 当前 plan / 恢复记忆", face=C["teal_bg"], edge=C["teal"], title_size=12, sub_size=10.5)

# Touch enters task-conditioned context, rather than the action branch directly.
arrow([(4.75, 6.37), (5.29, 6.37), (5.29, 11.37), (17.02, 11.37), (17.02, 11.13)], C["teal"], lw=1.6)
label(8.0, 11.39, "触觉语义", C["teal"], size=10)

box(6.07, 8.57, 16.98, 1.0, "共享 PaliGemma 多模态表征", "不同任务使用不同输入组合；动作分支输入视觉、当前状态及阶段指令 / plan", face=C["purple_bg"], edge=C["purple"], title_size=15, sub_size=10.5)
for xx in [8.32, 12.55, 18.77]:
    arrow([(xx, 10.09), (xx, 9.61)], C["purple"])

# Group the two actual heads as one stage-transition function.
box(6.06, 5.65, 4.55, 2.36, face=C["blue_bg"], edge=C["blue"])
text(8.33, 7.68, "阶段切换模块", 13, bold=True)
box(6.31, 6.48, 1.98, .78, "恢复触发", "need_recovery", edge=C["blue"], title_size=11, sub_size=9)
box(8.43, 6.48, 1.93, .78, "恢复终止", "adjustment_end", edge=C["blue"], title_size=11, sub_size=9)
text(8.33, 5.98, "共享池化特征 · 两个独立二分类头", 9.5, color=C["muted"])
arrow([(8.33, 8.55), (8.33, 8.05)], C["blue"])

# Autoregressive reasoning: two sequential tasks with a shared language decoder.
box(11.02, 5.65, 7.32, 2.36, face=C["orange_bg"], edge=C["orange"])
text(14.68, 7.68, "故障诊断与恢复推理（共享语言解码）", 12, bold=True)
box(11.25, 6.63, 2.25, .66, "Failure 诊断", edge=C["orange"], title_size=11.5)
box(15.25, 6.63, 2.85, .66, "Plan 恢复规划", edge=C["orange"], title_size=11.5)
box(11.25, 5.85, 3.40, .48, "外部记忆：已执行 plan → 后续 failure", edge=C["orange"], title_size=8.5)
arrow([(12.37, 6.61), (12.37, 6.36)], C["orange"], head=10)
arrow([(14.67, 6.09), (16.65, 6.09), (16.65, 6.60)], C["orange"], head=10)
label(15.43, 6.09, "历史反馈", C["orange"], size=8.8)
arrow([(14.68, 8.55), (14.68, 8.05)], C["orange"])

# Action branch and its learned plan conditioning.
box(18.86, 6.59, 4.19, 1.42, "共享 Action Expert", "Flow matching\n正常 / 恢复动作", face=C["purple_bg"], edge=C["purple"], title_size=12.5, sub_size=10)
arrow([(20.95, 8.55), (20.95, 8.05)], C["purple"])
arrow([(18.12, 6.95), (18.83, 6.95)], C["orange"], head=11)
label(18.48, 7.37, "plan 条件", C["orange"], size=8.5)
box(19.02, 5.65, 3.87, .54, "H30 关节动作块", edge=C["purple"], title_size=11)
arrow([(20.95, 6.56), (20.95, 6.22)], C["purple"], head=10)

# (c) Runtime loop, displayed separately from the model components.
box(.4, .72, 23.16, 3.83, face="#FAFCFE")
text(.70, 4.17, "(c) 异步执行与恢复闭环", 15, bold=True, ha="left")
text(23.12, 4.17, "动作执行期间后台评估；触发后立即中断动作块", 11, color=C["muted"], ha="right")

nodes = [
    (1.0, 2.22, 3.6, 1.0, "正常执行", "执行动作块 + 后台 need", C["blue_bg"], C["blue"]),
    (6.0, 2.22, 3.6, 1.0, "中断与保持", "need=true：截断旧动作", "#FCEFEF", C["red"]),
    (11.0, 2.22, 4.3, 1.0, "诊断 → 更新记忆 → 新 plan", "等待诊断与规划完成", C["orange_bg"], C["orange"]),
    (17.0, 2.22, 5.5, 1.0, "执行恢复动作", "按 plan 执行 + 后台 adjustment_end", C["teal_bg"], C["teal"]),
]
for xx, yy, ww, hh, tt, ss, ff, ee in nodes:
    box(xx, yy, ww, hh, tt, ss, face=ff, edge=ee, title_size=12, sub_size=10)
for x1, x2 in [(4.63, 5.96), (9.63, 10.96), (15.33, 16.96)]:
    arrow([(x1, 2.72), (x2, 2.72)], C["ink"])

# Logical coupling from model outputs to the control loop.
arrow([(8.33, 5.62), (8.33, 4.59)], C["blue"], dashed=True)
label(8.32, 4.91, "阶段判断", C["blue"], size=10)
arrow([(14.7, 5.62), (14.7, 4.59)], C["orange"], dashed=True)
label(14.7, 4.91, "故障与计划", C["orange"], size=10)
arrow([(20.95, 5.62), (20.95, 4.59)], C["purple"], dashed=True)
label(20.95, 4.91, "动作发布", C["purple"], size=10)

# End is not a task success head: it returns to execution, where need runs again.
arrow([(19.75, 2.19), (19.75, 1.47), (2.8, 1.47), (2.8, 2.18)], C["teal"], lw=1.7)
label(11.55, 1.47, "adjustment_end=true：结束当前恢复并继续原任务；若仍有故障，进入下一轮恢复", C["teal"], size=11)

# Observation feedback closes the physical loop.
arrow([(22.54, 2.71), (23.78, 2.71), (23.78, 12.31), (.18, 12.31), (.18, 10.64), (.96, 10.64)],
      C["muted"], dashed=True, lw=1.2, head=10)
label(20.0, 12.32, "环境反馈：更新图像 / qpos / 时序触觉", size=10.5)

text(.55, .28, "实线：信息或动作流   虚线：反馈 / 控制关联     ·     共享模型按请求串行推理，异步发生在物理动作执行与后台评估之间。", 10, color=C["muted"], ha="left")

stem = OUT / "模型推理框架_草稿"
fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=.12)
fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.12)
fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=.12)
plt.close(fig)
print(f"Generated {stem.name}.svg / .pdf / .png")
