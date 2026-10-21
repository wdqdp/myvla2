"""Editable paper figure; run from any directory with Python + matplotlib + Pillow.

python 论文准备/draw_inference_framework_v4.py --lang both
Optional real photographs belong in 框架图/figure_assets/; see 框架图_v4说明.md.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle, Polygon
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
    fig, ax = plt.subplots(figsize=(14.4, 9.81))
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 26.4), ylim=(0, 18.0))
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

    # A: preserve the tactile descriptor, allowing room for the expanded B panel.
    txt(.40,17.58,tr('(a) Tactile description','(a) 时序触觉元描述'),16,weight=True,ha='left')
    txt(8.0,17.58,tr('(b) Shared reasoning and action model','(b) 共享推理与动作模型'),16,weight=True,ha='left')
    line([(.4,17.05),(6.9,17.05)],C['teal'],2.1)
    line([(8.0,17.05),(25.8,17.05)],C['blue'],2.1)
    txt(3.68,16.53,tr('Paired fingertip signals · last T frames','双指触觉信号 · 最近T帧'),12,C['muted'])
    placeholder(.72,14.48,2.76,1.65,'tactile_left.png','P2L',tr('Left tactile clip','左指触觉序列'),'tactile')
    placeholder(3.94,14.48,2.76,1.65,'tactile_right.png','P2R',tr('Right tactile clip','右指触觉序列'),'tactile')
    for j in range(9):
        card(1.23+j*.56,13.98,.40,.17,C['teal_bg'],C['teal'],.65,.025)
    txt(1.20,13.66,'t − T + 1',10.5,C['muted'],ha='left')
    txt(6.15,13.66,'t',10.5,C['muted'],ha='right')
    arrow([(3.70,13.42),(3.70,13.05)],C['teal'])
    card(.94,12.20,5.52,.84,C['teal_bg'],C['teal'],1.1)
    txt(3.70,12.62,tr('Shared spatial CNN','逐帧共享空间CNN'),14,weight=True)
    arrow([(3.70,12.18),(3.70,11.83)],C['teal'])
    card(.94,10.65,5.52,1.15,C['teal_bg'],C['teal'],1.1)
    txt(3.70,11.38,tr('Temporal convolution','时序卷积TCN'),14,weight=True)
    for j in range(7):
        ax.add_patch(Circle((2.25+j*.48,10.99),.047,facecolor=C['teal'],edgecolor='none',zorder=4))
    for aa,bb in [(0,2),(2,4),(4,6),(0,4),(2,6)]:
        line([(2.25+aa*.48,11.03),(2.25+bb*.48,11.03)],C['teal'],.7)
    arrow([(3.70,10.63),(3.70,10.26)],C['teal'])
    card(.94,9.40,5.52,.84,C['teal_bg'],C['teal'],1.1)
    txt(3.70,9.82,tr('Field classifiers','元字段分类'),14,weight=True)
    arrow([(3.70,9.38),(3.70,8.86)],C['teal'])
    card(.72,6.37,5.98,2.46,'white',C['teal'],1.25)
    txt(3.71,8.30,tr('Discrete tactile text','离散元触觉描述文本'),14,weight=True)
    for xx,ww,s in [(1.13,.90,'area'),(2.23,.65,'Fx'),(3.08,.65,'Fy'),(3.93,.65,'Fz'),(4.78,1.50,'rotation')]:
        tag(xx,7.48,ww,s,C['teal'],C['teal_bg'],11)
    txt(3.71,6.90,tr('Reference: left-finger sensor frame','参考系：左指传感器局部坐标系'),10.8,C['muted'])

    # B: input legend above a shared model; functions grouped by their runtime role.
    inputs = {
        'C': ('#457BAA', '#EDF4FA'),
        'T': ('#278C83', '#EAF5F2'),
        'H': ('#8873A9', '#F2EDF8'),
        'P': ('#A88024', '#FBF5DF'),
        'F': ('#C57B54', '#FCF0E9'),
    }

    def chips(cx, y, keys, scale=1.0):
        w, h, gap = .45*scale, .34*scale, .12*scale
        x=cx-(len(keys)*w+(len(keys)-1)*gap)/2
        for k in keys:
            edge, fill=inputs[k]
            card(x,y,w,h,fill,edge,.85,.045,z=6)
            txt(x+w/2,y+h/2,k,10.8*scale,edge,weight=True)
            x+=w+gap

    def input_card(x,y,w,h,key,title,subtitle):
        edge,fill=inputs[key]
        card(x,y,w,h,fill,edge,.95,.10)
        tag(x+.14,y+h-.48,.51,key,edge,'white',10.5)
        txt(x+.86,y+h-.28,title,11.6,edge,weight=True,ha='left')
        txt(x+w/2,y+.32,subtitle,10.4,C['muted'])

    # Inputs, above a single shared model.
    card(8.25,14.00,8.00,2.40,inputs['C'][1],inputs['C'][0],1.05)
    tag(8.48,15.84,.52,'C',inputs['C'][0],'white',10.5)
    txt(9.24,16.05,tr('Common inputs','通用输入'),12.8,inputs['C'][0],weight=True,ha='left')
    placeholder(8.48,14.22,1.88,1.25,'front.png','P1F',tr('Front','正面相机'))
    placeholder(10.59,14.22,1.88,1.25,'wrist.png','P1W',tr('Wrist','腕部相机'))
    txt(14.35,15.12,tr('Current 7D qpos','当前7维qpos'),11.8,weight=True)
    txt(14.35,14.62,tr('Task instruction','任务指令'),11.8,weight=True)
    input_card(16.68,15.32,4.20,1.08,'T',tr('Tactile text','触觉元描述'),tr('Current Touch','当前Touch'))
    input_card(21.17,15.32,4.20,1.08,'H',tr('Motion history','运动历史'),'qpos H100 → 11')
    input_card(16.68,14.00,4.20,1.08,'P',tr('Current plan','当前恢复计划'),tr('none if unavailable','无当前计划时为none'))
    input_card(21.17,14.00,4.20,1.08,'F',tr('Global failures','全局故障序列'),tr('Current + past failures','当前与历史failure'))
    arrow([(6.73,7.61),(7.40,7.61),(7.40,16.64),(18.78,16.64),(18.78,16.42)],C['teal'],lw=1.3)
    note(13.8,16.64,tr('Tactile semantics','触觉语义'),C['teal'],10.8)

    card(7.92,6.06,17.90,7.52,fc='#FCFDFE',ec='#C5D3E1',lw=1.1,radius=.17)
    for xx in (12.25,21.0):
        arrow([(xx,13.97),(xx,13.61)],C['blue'],lw=1.1)
    card(8.25,12.94,17.12,.61,C['blue_bg'],C['blue'],1.2)
    txt(16.81,13.25,tr(r'Shared $\pi_{0.5}$ / PaliGemma backbone',r'共享$\pi_{0.5}$ / PaliGemma主干'),14.7,weight=True)

    # Stage control is separate from the monitoring heads. A need-triggered
    # transition explicitly includes hold/diagnosis/planning before Adjustment.
    txt(10.79,12.49,tr('Hold & reason','保持动作并推理'),11.5,C['amber'],weight=True)
    card(14.12,11.50,11.25,1.10,C['grey_bg'],C['line'],.9,.08)
    card(14.40,11.83,2.70,.58,C['blue_bg'],C['blue'],.9,.06)
    txt(15.75,12.12,'Execution',11.8,weight=True)
    card(22.35,11.83,2.70,.58,C['teal_bg'],C['teal'],.9,.06)
    txt(23.70,12.12,'Adjustment',11.8,weight=True)
    arrow([(17.13,12.12),(22.32,12.12)],C['amber'],lw=1.2)
    note(19.73,12.42,tr('need = true: hold → plan','need = true：保持 → 诊断/规划'),C['amber'],9.6)
    arrow([(23.70,11.80),(23.70,11.63),(15.75,11.63),(15.75,11.80)],C['teal'],lw=1.1,head=8)
    note(19.73,11.63,'adj = true',C['teal'],9.5)

    # Left column: the same diagnosis and planning inputs, and failure-only memory.
    card(8.25,6.28,5.08,5.73,C['amber_bg'],'#E4CFB5',.9)
    txt(10.79,11.72,tr('On need = true','need = true 时调用'),10.5,C['amber'])
    chips(10.79,11.20,'CTHP')
    card(8.48,10.10,4.62,.86,'white',C['amber'],1.0)
    txt(10.79,10.53,tr('Failure diagnosis','Failure故障诊断'),12.6,weight=True)
    arrow([(10.79,10.07),(10.79,9.72)],inputs['F'][0],lw=1.2)
    card(8.48,8.78,4.62,.91,inputs['F'][1],inputs['F'][0],.9,.06)
    txt(10.79,9.24,tr('Global failure sequence','全局failure序列'),11.0,inputs['F'][0],weight=True)
    arrow([(10.79,8.75),(10.79,8.36)],inputs['F'][0],lw=1.1)
    chips(10.79,7.87,'CF')
    card(8.48,6.80,4.62,.86,'white',C['amber'],1.0)
    txt(10.79,7.23,tr('Recovery planning','Plan恢复规划'),12.6,weight=True)
    txt(10.79,6.50,tr('Direction + magnitude','恢复方向与程度'),10.4,C['muted'])

    # Assessment loop: get a fresh observation, select a head using the active
    # phase, and resample if false. This is not a synchronous action→judge chain.
    txt(16.85,11.21,tr('Repeated assessment','持续状态判断'),11.5,C['blue'],weight=True)
    card(14.47,10.64,4.76,.44,'white',C['line'],.85,.06)
    txt(16.85,10.86,tr('Fresh observation','获取最新观测'),10.8,C['muted'])
    note(16.85,10.38,tr('Select by phase','按当前阶段选头'),C['muted'],9.7)
    arrow([(15.30,10.61),(15.30,10.28)],C['blue'],lw=1.1,head=8)
    arrow([(18.40,10.61),(18.40,10.28)],C['teal'],lw=1.1,head=8)
    for x,head,phase,color,fill in [(14.13,'need','Execution',C['blue'],C['blue_bg']),
                                     (17.23,'adj','Adjustment',C['teal'],C['teal_bg'])]:
        card(x,9.11,2.34,1.14,fill,color,1.0,.09)
        txt(x+1.17,9.99,head,12.3,color,weight=True)
        txt(x+1.17,9.65,phase,9.6,C['muted'])
        chips(x+1.17,9.21,'CTHP',.88)
    arrow([(15.30,9.08),(15.30,8.78),(16.85,8.78),(16.85,8.68)],C['muted'],lw=1.0,head=8)
    line([(18.40,9.08),(18.40,8.78),(16.85,8.78)],C['muted'],1.0)
    ax.add_patch(Polygon([(16.85,8.68),(17.47,8.31),(16.85,7.94),(16.23,8.31)],
                         closed=True,facecolor='white',edgecolor=C['blue'],lw=1.0,zorder=3))
    txt(16.85,8.31,'true?',10.4,C['blue'],weight=True)
    arrow([(16.20,8.31),(13.74,8.31),(13.74,10.86),(14.44,10.86)],C['blue'],lw=1.25)
    txt(13.93,9.56,tr('false: resample','false：再次采样'),9.5,C['blue'],rotation=90,
        bbox=dict(facecolor='white',edgecolor='none',pad=1.0))

    # One Action Expert feeds a queue; a separately executing physical chunk
    # is the target of the interrupt, not the model itself.
    txt(23.21,11.21,tr('Action generation & motion','动作生成与执行'),11.5,C['blue'],weight=True)
    card(21.06,9.48,4.31,1.59,C['blue_bg'],C['blue'],1.1)
    txt(23.215,10.81,'Action Expert',12.4,weight=True)
    txt(21.34,10.30,'execution',10.2,C['muted'],ha='left')
    chips(24.08,10.13,'C',.93)
    txt(21.34,9.83,'adjustment',10.2,C['muted'],ha='left')
    chips(24.08,9.66,'CP',.93)
    arrow([(23.215,9.45),(23.215,9.19)],C['blue'],lw=1.1,head=8)
    card(21.65,8.77,3.17,.40,C['blue_bg'],C['blue'],.8,.045)
    for j in range(1,10):
        xx=21.65+j*.317
        line([(xx,8.80),(xx,9.14)],C['blue'],.6)
    note(25.00,8.97,'H30',C['blue'],9.7)
    arrow([(23.215,8.74),(23.215,8.35)],C['blue'],lw=1.1,head=8)
    cut=24.20
    card(21.06,7.73,4.31,.59,C['grey_bg'],C['blue'],.9,.04)
    ax.add_patch(Rectangle((21.07,7.74),cut-21.07,.57,facecolor=C['blue_bg'],edgecolor='none',zorder=3))
    for j in range(1,8):
        xx=21.06+j*.52
        line([(xx,7.77),(xx,8.28)],C['blue'] if xx<cut else C['slate'],.6)
    txt(22.58,8.025,tr('Executing','逐步执行中'),10.3,C['blue'],weight=True,
        bbox=dict(facecolor=C['blue_bg'],edgecolor='none',pad=.8))
    line([(cut,7.65),(cut,8.38)],C['red'],1.6,z=7)
    ax.add_patch(Circle((cut,8.025),.075,facecolor=C['red'],edgecolor='white',lw=.5,zorder=8))
    arrow([(23.215,7.70),(23.215,7.37)],C['blue'],lw=1.0,head=8)
    placeholder(21.62,6.20,3.19,1.14,'robot.png','P3',tr('Robot interaction','真机交互画面'))

    # A true assessment interrupts actual motion and requests a stage update.
    arrow([(17.50,8.31),(20.10,8.31),(20.10,8.025),(21.03,8.025)],C['red'],lw=1.6)
    note(19.24,8.53,tr('true → interrupt','true → 中断动作'),C['red'],9.5)
    arrow([(20.10,8.31),(20.10,11.47)],C['red'],lw=1.0,dash=True,head=8)
    txt(20.27,10.15,tr('update phase','更新阶段'),9.4,C['red'],rotation=90,
        bbox=dict(facecolor='white',edgecolor='none',pad=1.2))

    # Local overlap inset replaces the old unsupported "asynchronous" label.
    txt(16.82,7.53,tr('One chunk · repeated checks','一个动作块内多次判断'),10.4,C['muted'],weight=True)
    card(14.66,6.91,4.44,.22,C['grey_bg'],C['blue'],.7,.025)
    ax.add_patch(Rectangle((14.67,6.92),3.66,.20,facecolor=C['blue_bg'],edgecolor='none',zorder=3))
    for xx in [15.4,16.2,17.0,17.8]:
        line([(xx,6.94),(xx,7.10)],C['blue'],.55)
    for xx in [15.0,16.1,17.2]:
        ax.add_patch(Circle((xx,6.56),.045,facecolor=C['blue'],edgecolor='none',zorder=5))
    ax.add_patch(Circle((18.33,6.56),.057,facecolor=C['red'],edgecolor='none',zorder=5))
    arrow([(18.33,6.64),(18.33,7.16)],C['red'],lw=.95,head=7)
    txt(16.1,6.28,'false',9.1,C['muted'])
    txt(18.33,6.28,'true',9.1,C['red'])

    # Lower panel: execution lane versus assessment lane. This is not GPU parallelism.
    line([(.4,5.76),(25.80,5.76)],C['line'],1.0)
    txt(.4,5.25,tr('(c) Asynchronous execution–recovery loop','(c) 异步执行与恢复闭环'),
        16,weight=True,ha='left')
    txt(25.65,5.25,tr('Monitor during motion · interrupt within a chunk',
                       '动作执行期间评估 · 支持块内中断'),11.6,C['muted'],ha='right')
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
    txt(9.84,1.94,tr('Diagnose → plan','诊断 → 规划'),11.2,C['amber'],weight=True)
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
    stem=ROOT / ('模型推理框架_v4_中文' if chinese else '模型推理框架_v4_英文')
    for suffix in ('.svg','.pdf','.png'):
        fig.savefig(stem.with_suffix(suffix), dpi=260, facecolor='white',
                    metadata={'Creator':'myvla2 paper figure v4'})
    # Standalone B-panel crop for reviewing the new input/stage layout.
    crop = matplotlib.transforms.Bbox.from_extents(7.75, 5.91, 26.10, 17.95)
    crop_inches = crop.transformed(ax.transData).transformed(fig.dpi_scale_trans.inverted())
    crop_stem = ROOT / ('B区_异步判断_v4_中文' if chinese else 'B区_异步判断_v4_英文')
    for suffix in ('.svg','.pdf','.png'):
        fig.savefig(crop_stem.with_suffix(suffix), dpi=260, facecolor='white',
                    bbox_inches=crop_inches, pad_inches=0,
                    metadata={'Creator':'myvla2 paper figure v4'})
    plt.close(fig)
    print(f'Generated {stem.name} and B-panel crop: SVG / PDF / PNG')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lang',choices=['en','zh','both'],default='both')
    parser.add_argument('--asset-dir',type=Path,default=ROOT/'figure_assets')
    args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    for language in (['zh','en'] if args.lang == 'both' else [args.lang]):
        draw(language,args.asset_dir)
