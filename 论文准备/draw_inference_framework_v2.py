"""Editable paper figure; run from any directory with Python + matplotlib + Pillow.

python 论文准备/draw_inference_framework_v2.py --lang both
Optional real photographs belong in 框架图/figure_assets/; see 精修版说明.md.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/myvla2_mpl_v2')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/myvla2_figure_cache_v2')
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
from matplotlib.path import Path as MPath
from PIL import Image

ROOT = Path(__file__).resolve().parent / '框架图'
C = dict(ink='#202C3B', muted='#687686', line='#DCE3E9', blue='#457BAA',
         blue_bg='#EDF4FA', teal='#278C83', teal_bg='#EAF5F2',
         amber='#BF8548', amber_bg='#FCF3E7', red='#C56F6E', red_bg='#FAEDEC',
         slate='#8B98A6', grey_bg='#F4F6F8', white='#FFFFFF', purple='#8172A7')


def draw(lang: str, assets: Path):
    chinese = lang == 'zh'
    def tr(en, zh):
        return zh if chinese else en
    reg = font_manager.FontProperties(fname=(
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc' if chinese else
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    bold = font_manager.FontProperties(fname=(
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc' if chinese else
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'pdf.compression': 9, 'axes.unicode_minus': False})
    fig, ax = plt.subplots(figsize=(14.4, 8.72))
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 26.4), ylim=(0, 16.0))
    ax.axis('off')
    fig.subplots_adjust(left=.008, right=.992, bottom=.01, top=.99)
    texts = []

    def txt(x, y, s, size=12, color=None, weight=False, ha='center', va='center', z=8, **kw):
        t = ax.text(x, y, s, fontsize=size, color=color or C['ink'],
                    fontproperties=bold if weight else reg, ha=ha, va=va,
                    linespacing=1.36, zorder=z, **kw)
        texts.append(t)
        return t

    def card(x, y, w, h, fc='white', ec=None, lw=1.0, radius=.12, dash=False, z=2):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                          facecolor=fc, edgecolor=ec or C['line'], linewidth=lw,
                          linestyle=(0, (4, 3)) if dash else '-', zorder=z)
        ax.add_patch(p)
        return p

    def line(points, color=None, lw=1.25, dash=False, z=4):
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color or C['line'], linewidth=lw,
                linestyle=(0, (4, 3)) if dash else '-', solid_capstyle='round', zorder=z)

    def arrow(points, color=None, lw=1.5, dash=False, head=11, z=5):
        path = MPath(points, [MPath.MOVETO] + [MPath.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=head,
                                    linewidth=lw, color=color or C['muted'],
                                    linestyle=(0, (4, 3)) if dash else '-',
                                    capstyle='round', joinstyle='round', zorder=z))

    def tag(x, y, w, text, color, fill=None, size=11.5):
        card(x, y, w, .43, fill or 'white', color, .8, .08, z=5)
        txt(x+w/2, y+.215, text, size, color, weight=True)

    def note(x, y, text, color=None, size=11.5, **kw):
        return txt(x, y, text, size, color or C['muted'],
                   bbox=dict(facecolor='white', edgecolor='none', pad=1.5), **kw)

    def placeholder(x, y, w, h, name, pid, caption, kind='camera'):
        path = assets / name
        if path.is_file():
            with Image.open(path) as im:
                image = im.convert('RGB').copy()
            clip = card(x, y, w, h, fc='none', ec=C['line'], lw=.9, radius=.08, z=7)
            iw, ih = image.size
            scale = max(w/iw, h/ih)
            dw, dh = iw*scale, ih*scale
            artist = ax.imshow(image, extent=(x+(w-dw)/2, x+(w+dw)/2,
                                               y+(h-dh)/2, y+(h+dh)/2),
                               aspect='auto', interpolation='antialiased', zorder=3)
            artist.set_clip_path(clip)
        else:
            card(x, y, w, h, C['grey_bg'], '#B4C0CC', .9, .08, dash=True)
            # Deliberate photo placeholder, not fabricated experimental imagery.
            if kind == 'camera':
                cx, cy = x+w/2, y+h*.68
                card(cx-.22, cy-.12, .44, .26, fc='none', ec='#AAB7C4', radius=.035, lw=.9)
                ax.add_patch(Circle((cx, cy+.01), .065, facecolor='none', edgecolor='#AAB7C4', lw=.8, zorder=4))
            else:
                cx, cy = x+w/2, y+h*.68
                for ix in range(5):
                    for iy in range(3):
                        ax.add_patch(Circle((cx+(ix-2)*.12, cy+(iy-1)*.12), .022,
                                            facecolor='#AAB7C4', edgecolor='none', zorder=4))
            txt(x+w/2, y+h*.36, caption, 11, C['muted'])
            tag(x+.10, y+h-.42, .61, pid, C['slate'], C['white'], 9.5)

    # Wide paper overview: tactile semantics → shared reasoning → actions.
    txt(.40, 15.58, tr('(a) Tactile description', '(a) 时序触觉元描述'),
        16, weight=True, ha='left')
    txt(8.0, 15.58, tr('(b) Shared reasoning and action model', '(b) 共享推理与动作模型'),
        16, weight=True, ha='left')
    line([(.4,15.05),(6.9,15.05)], C['teal'], 2.1)
    line([(8.0,15.05),(25.8,15.05)], C['blue'], 2.1)

    # Tactile module. Empty photo slots carry stable IDs for replacement.
    txt(3.68, 14.53, tr('Paired fingertip signals · last T frames', '双指触觉信号 · 最近T帧'),
        12, C['muted'])
    placeholder(.72, 12.48, 2.76, 1.65, 'tactile_left.png', 'P2L',
                tr('Left tactile clip', '左指触觉序列'), 'tactile')
    placeholder(3.94, 12.48, 2.76, 1.65, 'tactile_right.png', 'P2R',
                tr('Right tactile clip', '右指触觉序列'), 'tactile')
    # A small temporal strip replaces the old fabricated contact heatmaps.
    for j in range(9):
        card(1.23+j*.56, 11.98, .40, .17, C['teal_bg'], C['teal'], .65, .025)
    txt(1.20, 11.66, 't − T + 1', 10.5, C['muted'], ha='left')
    txt(6.15, 11.66, 't', 10.5, C['muted'], ha='right')
    arrow([(3.70,11.42),(3.70,11.05)], C['teal'])
    card(.94, 10.20, 5.52, .84, C['teal_bg'], C['teal'], 1.1)
    txt(3.70,10.62,tr('Shared spatial CNN','逐帧共享空间CNN'),14,weight=True)
    arrow([(3.70,10.18),(3.70,9.83)],C['teal'])
    card(.94,8.65,5.52,1.15,C['teal_bg'],C['teal'],1.1)
    txt(3.70,9.38,tr('Temporal convolution','时序卷积TCN'),14,weight=True)
    # Stylized temporal dependencies, no unmeasured signals or data curves.
    for j in range(7):
        xx=2.25+j*.48
        ax.add_patch(Circle((xx,8.99),.047,facecolor=C['teal'],edgecolor='none',zorder=4))
    for a,b in [(0,2),(2,4),(4,6),(0,4),(2,6)]:
        line([(2.25+a*.48,9.03),(2.25+b*.48,9.03)], C['teal'], .7)
    arrow([(3.70,8.63),(3.70,8.26)],C['teal'])
    card(.72,6.37,5.98,1.87,'white',C['teal'],1.25)
    txt(3.71,7.79,tr('Discrete tactile text','离散元触觉描述文本'),14,weight=True)
    for xx, ww, s in [(1.13,.90,'area'),(2.23,.65,'Fx'),(3.08,.65,'Fy'),
                       (3.93,.65,'Fz'),(4.78,1.50,'rotation')]:
        tag(xx,7.01,ww,s,C['teal'],C['teal_bg'],11)
    txt(3.71,6.62,tr('Reference: left-finger sensor frame','参考系：左指传感器局部坐标系'),10.8,C['muted'])

    # One shared VLA, with task-dependent inputs. No duplicated VLM modules.
    card(7.92,6.12,17.90,8.58,fc='#FCFDFE',ec='#C5D3E1',lw=1.1,radius=.17)
    placeholder(8.28,12.56,2.55,1.58,'front.png','P1F',tr('Front view','正面相机画面'))
    placeholder(11.10,12.56,2.55,1.58,'wrist.png','P1W',tr('Wrist view','腕部相机画面'))
    txt(10.97,12.26,tr('Visual observations · SigLIP','当前视觉观测 · SigLIP编码'),11.4,C['muted'])
    card(14.12,12.56,3.68,1.58,C['blue_bg'],C['line'],.85)
    txt(15.96,13.58,tr('State + task','状态与任务'),13,weight=True)
    txt(15.96,12.99,tr('qpos · instruction','qpos · 任务指令'),11.2,C['muted'])
    card(18.27,12.56,7.12,1.58,C['teal_bg'],C['line'],.85)
    txt(21.83,13.62,tr('Task-specific context','按任务选择上下文'),13,weight=True)
    txt(21.83,12.98,tr('Touch · motion history\ncurrent plan · recovery memory',
                       'Touch · 运动历史\n当前plan · 恢复记忆'),11.4,C['muted'])
    for xx in (10.97,15.96,21.83):
        arrow([(xx,12.09),(xx,11.74)],C['blue'],lw=1.15)
    arrow([(6.73,7.36),(7.40,7.36),(7.40,14.40),(21.83,14.40),(21.83,14.16)],C['teal'],lw=1.3)
    note(15.1,14.41,tr('Tactile semantics','触觉语义'),C['teal'],11)
    card(8.28,10.66,17.11,1.03,C['blue_bg'],C['blue'],1.3)
    txt(16.84,11.28,tr(r'Shared $\pi_{0.5}$ / PaliGemma backbone',r'共享$\pi_{0.5}$ / PaliGemma主干'),15.5,weight=True)
    txt(16.84,10.88,tr('Task-conditioned calls · shared parameters','按任务分别调用 · 参数共享'),11.2,C['muted'])

    # Reasoning: failure enters an external history record before the next plan.
    card(8.28,6.52,8.90,3.60,C['amber_bg'],'#E4CFB5',.9)
    txt(12.73,9.88,tr('Shared language decoding','共享语言解码'),11.5,C['amber'],weight=True)
    card(8.60,8.40,3.67,1.10,'white',C['amber'],1.1)
    txt(10.44,9.09,tr('Failure diagnosis','故障诊断'),13.5,weight=True)
    txt(10.44,8.65,tr('Global-frame failure','全局坐标系故障'),10.8,C['muted'])
    card(13.18,8.40,3.67,1.10,'white',C['amber'],1.1)
    txt(15.02,9.09,tr('Recovery planning','恢复策略推理'),13.5,weight=True)
    txt(15.02,8.65,tr('Direction + magnitude','恢复方向与程度'),10.8,C['muted'])
    arrow([(10.44,10.63),(10.44,10.14)],C['blue'],lw=1.2)
    arrow([(15.02,10.63),(15.02,10.14)],C['blue'],lw=1.2)
    card(8.60,6.80,8.25,1.04,'white','#D6B68E',.95)
    txt(12.73,7.54,tr('Recovery memory','恢复历史记忆'),12.5,weight=True)
    txt(12.73,7.09,tr('Executed plan  →  subsequent failure','已执行plan  →  后续failure'),11.5,C['amber'])
    arrow([(10.44,8.37),(10.44,7.87)],C['amber'])
    arrow([(15.02,7.87),(15.02,8.37)],C['amber'])

    # Action and phase heads are part of the same shared model.
    card(18.00,8.46,3.63,1.59,C['blue_bg'],C['blue'],1.1)
    txt(19.82,9.62,'Action Expert',14,weight=True)
    txt(19.82,9.04,tr('Flow matching\nH30 action chunk','Flow matching\nH30关节动作块'),11.1,C['muted'])
    arrow([(19.82,10.63),(19.82,10.08)],C['blue'],lw=1.2)
    arrow([(16.89,8.97),(17.96,8.97)],C['amber'],lw=1.5)
    note(17.43,9.37,'plan',C['amber'],10.7)
    placeholder(22.28,8.46,3.11,1.59,'robot.png','P3',tr('Robot interaction','真机交互画面'))
    arrow([(21.66,9.26),(22.25,9.26)],C['blue'])
    txt(21.71,8.17,tr('Actions use vision, state and task / plan','动作输入：视觉、状态与任务/plan'),10.6,C['muted'])
    card(18.00,6.52,7.39,1.22,'white',C['blue'],1.05)
    line([(21.70,6.73),(21.70,7.53)],C['line'],1.0)
    txt(19.84,7.40,tr('Trigger recovery','触发恢复'),12.4,weight=True)
    txt(19.84,6.88,'need_recovery',10.8,C['muted'])
    txt(23.55,7.40,tr('End recovery','结束恢复'),12.4,weight=True)
    txt(23.55,6.88,'adjustment_end',10.8,C['muted'])
    arrow([(25.48,11.16),(25.65,11.16),(25.65,7.99),(23.55,7.99),(23.55,7.78)],C['blue'],lw=1.1)
    txt(21.70,6.27,tr('Two independent binary heads','两个独立二分类头'),10.8,C['muted'])

    # Compact environmental feedback: returns to visual input and touch context.
    arrow([(25.42,9.26),(26.08,9.26),(26.08,14.83),(10.97,14.83),(10.97,14.18)],
          C['slate'],lw=1.0,dash=True,head=9)
    note(18.8,14.86,tr('Updated observations','环境反馈：更新观测'),C['muted'],10.8)

    # Lower panel: execution lane versus assessment lane. This is not GPU parallelism.
    line([(.4,5.76),(25.80,5.76)],C['line'],1.0)
    txt(.4,5.25,tr('(c) Asynchronous execution–recovery loop','(c) 异步执行与恢复闭环'),
        16,weight=True,ha='left')
    txt(25.65,5.25,tr('Monitor during motion · interrupt within a chunk',
                       '动作执行期间评估 · 支持块内中断'),11.6,C['muted'],ha='right')
    # A dashed coupling links the phase outputs to the timeline without long crossings.
    arrow([(23.55,6.06),(23.55,5.79)],C['blue'],lw=1.0,dash=True,head=9)
    txt(1.35,3.33,tr('Motion','动作执行'),12.5,weight=True)
    txt(1.35,1.92,tr('Assessment','状态评估'),12.5,weight=True)
    segments=[(3.10,7.75,C['blue_bg'],C['blue']),
              (7.75,11.93,C['grey_bg'],C['slate']),
              (11.93,19.25,C['teal_bg'],C['teal']),
              (19.25,25.35,C['blue_bg'],C['blue'])]
    labels=[tr('Execute','正常执行'),tr('Hold & reason','保持并推理'),
            tr('Recover','执行恢复'),tr('Resume task','继续原任务')]
    for (a,b,fill,edge),label in zip(segments,labels):
        txt((a+b)/2,4.18,label,13,weight=True)
        card(a,2.94,b-a,.79,fill,edge,.85,.055)
    # Chunk boundaries remain visible, even where a chunk is interrupted.
    for a,b,color in [(3.1,7.75,C['blue']),(11.93,19.25,C['teal']),
                      (19.25,25.35,C['blue'])]:
        x=a+.70
        while x < b-.06:
            line([(x,2.98),(x,3.69)],color,.65)
            x+=.70
    txt(9.84,3.33,'HOLD',12,C['muted'],weight=True)
    # Assessment dots are repeated calls, not parallel heads at each time step.
    for x in [3.50,4.28,5.06,5.84,6.62,7.40]:
        ax.add_patch(Circle((x,1.94),.074,facecolor=C['blue'],edgecolor='white',lw=.4,zorder=5))
    txt(5.43,2.47,'need_recovery',11.4,C['blue'])
    card(7.97,1.56,3.75,.76,C['amber_bg'],C['amber'],.9,.055)
    txt(9.84,1.94,tr('Diagnose → plan','诊断 → 记忆 → plan'),11.2,C['amber'],weight=True)
    for x in [12.35,13.10,13.85,14.60,15.35,16.10,16.85,17.60,18.35,18.98]:
        ax.add_patch(Circle((x,1.94),.074,facecolor=C['teal'],edgecolor='white',lw=.4,zorder=5))
    txt(15.60,2.47,'adjustment_end',11.4,C['teal'])
    for x in [19.70,20.49,21.28,22.07,22.86,23.65,24.44,25.17]:
        ax.add_patch(Circle((x,1.94),.074,facecolor=C['blue'],edgecolor='white',lw=.4,zorder=5))
    txt(22.30,2.47,'need_recovery',11.4,C['blue'])
    # Distinct gates show immediate interruption and return to normal execution.
    for x,color,label in [(7.75,C['red'],'need = true'),(19.25,C['teal'],'end = true')]:
        line([(x,1.48),(x,4.45)],color,1.6,dash=True,z=6)
        ax.add_patch(Circle((x,3.34),.115,facecolor=color,edgecolor='white',lw=.8,zorder=7))
        note(x,4.65,label,color,11.2)
    arrow([(3.10,1.08),(25.35,1.08)],C['slate'],lw=1.0,head=8)
    txt(25.55,1.08,'t',11,C['muted'],ha='left')
    note(14.0,.66,tr('Resume, re-evaluate, and recover again if a fault remains.',
                      '恢复结束后继续原任务；若仍有故障，再次触发下一轮恢复。'),C['muted'],11)
    txt(.42,.13,tr('Solid: information / action flow     Dashed: feedback / stage control     •     Shared-model calls are serialized; monitoring overlaps physical motion.',
                   '实线：信息/动作流    虚线：反馈/阶段控制    ·    共享模型按请求串行推理；异步发生在物理执行与后台评估之间。'),
        9.8,C['muted'],ha='left')

    # Sanity-check clipping; SVG keeps text editable, PDF embeds font subsets.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=fig.bbox
    for t in texts:
        bb=t.get_window_extent(renderer)
        if bb.x0 < bounds.x0-1 or bb.x1 > bounds.x1+1 or bb.y0 < bounds.y0-1 or bb.y1 > bounds.y1+1:
            raise RuntimeError(f'Text outside canvas: {t.get_text()}')
    stem=ROOT / ('模型推理框架_精修版_中文' if chinese else '模型推理框架_精修版_英文')
    for suffix in ('.svg','.pdf','.png'):
        fig.savefig(stem.with_suffix(suffix), dpi=260, facecolor='white',
                    metadata={'Creator':'myvla2 paper figure v2'})
    plt.close(fig)
    print(f'Generated {stem.name}: SVG / PDF / PNG')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lang',choices=['en','zh','both'],default='both')
    parser.add_argument('--asset-dir',type=Path,default=ROOT/'figure_assets')
    args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    for language in (['zh','en'] if args.lang == 'both' else [args.lang]):
        draw(language,args.asset_dir)
