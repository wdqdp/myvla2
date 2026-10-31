# Method

TARO旨在利用动态触觉理解接触故障，并依据故障历史推理恢复策略，实现任务执行中的多轮自主恢复。如图1(d)所示，系统在正常执行与恢复两个阶段间切换。正常执行时，Need与物理动作执行异步运行，判断是否需要恢复；触发后，系统中断动作，结合视觉与触觉诊断全局故障，并根据全局故障历史生成恢复计划。恢复时，动作模型执行该计划，End异步判断当前恢复操作是否完成，完成后继续原任务。若再次检测到故障，系统将新的故障反馈纳入历史，开展下一轮恢复。

![图1：TARO的触觉描述、故障理解、共享模型与异步部署](框架图/cvpr2027.png)

*图1. TARO总体框架。（a）触觉描述器以CNN–TCN编码双指几何与力学信号，输出离散触觉文本。（b）结合视觉，将左指传感器局部坐标系下的触觉语义解释为固定场景坐标系下的故障。（c）共享视觉语言主干支持Need/End二分类及Failure/Plan自回归生成，并为流匹配动作专家提供条件。（d）正常执行期间由Need异步监测，触发后诊断故障并生成计划；恢复期间由End异步判断当前操作是否完成，随后继续原任务。输入标记C、T、M、P、F分别表示通用输入、触觉描述、运动历史、计划和全局故障历史；运动历史仅用于End。异步指状态评估与物理运动重叠，共享模型调用仍串行调度。*

## 3.1 轻量化时序触觉描述器

### 触觉信号

如图1(a)所示，为例减轻对原始触觉图像成像外观的依赖，促进跨传感器泛化。我们使用以下三类触觉信号：mesh_3d为当前接触的三维点云，mesh_3d_flow为初始点云到当前点云的位移向量场，force为三维力分布。每类信号的维度均为 \(35\times20\times3\)，归一化后按固定左右顺序沿通道拼接为18通道网格 \(x_t\)，取最近 \(T=30\) 帧作为输入窗口 \(\mathcal T_t\)。

### 空间与时序编码

为同时捕获接触分布与动态变化，缓解单帧观测容易产生的接触状态歧义问题，我们以共享CNN逐帧提取空间特征 \(e_\tau=E_{\mathrm{CNN}}(x_\tau)\)，再通过时序卷积网络（TCN）聚合接触演化：
\[
u_t=
\left[E_{\mathrm{TCN}}(e_{t-T+1},\ldots,e_t)\right]_T.
\]
末位置表征 \(u_t\) 汇集截至当前时刻的窗口信息。网络层数、特征维度及卷积配置见附录。

### 字段预测与文本化

本文设计的触觉描述体系目标是用尽可能少的字段概括尽可能多的常见接触状态，我们围绕接触面积、三轴受力与平面内旋转设计area、Fx、Fy、Fz和rotation五个元字段。其中，Fx/Fy兼顾双指同向与异向受力，Fz描述法向受力及双指幅值差异。五个独立分类头从 \(u_t\) 预测离散标签，再按固定字段顺序通过模板生成触觉文本 \(c_t\)。由于相向安装的左右传感器对同一接触旋转呈现相反方向，我们统一以左指传感器局部坐标系 \(\mathcal F_L\) 定义方向标签。例如，同一旋转在左指视角下为顺时针、右指视角下为逆时针时，统一标注为rotation=clockwise。完整字段取值与文本模板见附录。

## 3.2 两阶段动作执行与异步状态判断

### 正常执行与恢复的动作条件

如图1(c)所示，TARO的动作生成沿用π0.5的视觉语言主干与动作专家，通过流匹配迭代去噪生成动作块。图中的C表示通用输入 \(\mathcal C_t=(I_t^{\mathrm{front}},I_t^{\mathrm{wrist}},q_t,\ell)\)，包括主摄与腕部视角的当前图像、当前机器人状态和任务指令。两路图像经共享SigLIP编码，与文本化的当前状态及任务指令共同构成视觉语言前缀。

正常执行与恢复共用同一动作专家，通过通用输入 \(\mathcal C_t\) 文本部分的`Mode: execution`与`Mode: adjustment`区分两个阶段。恢复阶段额外输入当前计划 \(p_k\)，以引导恢复动作：
\[
A_t\sim
\begin{cases}
\pi_\theta(A\mid\mathcal C_t),&s_t=\mathrm E,\\
\pi_\theta(A\mid\mathcal C_t,p_k),&s_t=\mathrm R.
\end{cases}
\]
其中 \(s_t\) 表示当前阶段，\(\mathrm E\) 和 \(\mathrm R\) 分别表示正常执行与恢复，\(k\) 表示恢复决策序号。

### Need/End的输入组织与分类结构

正常执行以推进原任务为目标，恢复以纠正故障为目标，因此我们将两种不同操作意图建模为独立的执行与恢复阶段。为决定阶段切换，我们在共享PaliGemma主干上增加Need和End两个参数独立的二分类头，如图1(c)所示。Need在正常执行阶段判断是否需要恢复，使用通用输入、当前触觉元描述和计划上下文；End在恢复阶段额外读取运动历史，结合plan中的程度要求判断何时结束当前恢复操作，从而控制恢复动作的幅度：
\[
\begin{aligned}
\mathcal X_t&=\left(\mathcal C_t,c_t,p_t^{\mathrm{ctx}}\right),\\
\mathcal X_t^{\mathrm{End}}&=\left(\mathcal C_t,c_t,M_t,p_k\right).
\end{aligned}
\]
图1(d)用C、T、M、P标记两阶段各分支的输入，分别对应通用输入、触觉描述、运动历史与计划。其中 \(c_t\) 为当前触觉元描述，\(p_t^{\mathrm{ctx}}\) 为正常阶段最近一次恢复计划，首次尚无计划时使用none；\(p_k\) 为恢复阶段正在执行的计划。运动历史（motion history）\(M_t\) 仅用于End，下标\(t\)表示当前观测时刻。其采用H100窗口，由截至时刻\(t\)的最近100帧机器人状态均匀采样为11点并离散量化为文本，末项为当前状态，按本文训练数据同步采样率30 Hz计覆盖约3.3秒；回合起始的不足部分用首帧填充。推理时，我们依据时间戳将机器人回传状态重采样至与训练一致的30 Hz时间网格，从最近100个状态中均匀选取11点，并离散量化为文本。该历史为End提供此前已执行的运动信息，使其结合计划程度判断恢复进度。

如图1(c)中Mean Pool分支所示，共享主干输出的有效token表征经平均池化得到 \(\bar z_t\)，再分别通过两个判断头的两层线性投影映射到二维空间，经softmax得到恢复触发概率 \(P_{\mathrm{Need}}\) 和恢复操作完成概率 \(P_{\mathrm{End}}\)。
正常阶段以 \(P_{\mathrm{Need}}\geq\eta_{\mathrm{Need}}\) 触发恢复，恢复阶段以 \(P_{\mathrm{End}}\geq\eta_{\mathrm{End}}\) 结束当前操作。Need的触发阈值固定为 \(\eta_{\mathrm{Need}}=0.5\)；End的结束阈值在独立验证集上选取，以End Macro-F1最高为准，并在真机测试中保持固定。

### 与物理动作执行的异步协调

如图1(d)所示，模型部署于模型服务端（model server），机器人端（robot client）负责采集观测、发送推理请求和控制机械臂。机器人端以30 Hz发布率执行返回的动作块，并根据更新后的观测约以3 Hz请求当前阶段对应的状态判断。模型服务端串行处理推理请求，物理动作执行与状态评估在时间上重叠，因此判断能够在动作块尚未执行完时触发阶段切换。

Need触发后，机器人端中断当前动作并暂停，待模型服务端完成故障诊断与恢复规划后执行恢复动作。End触发后，机器人端结束当前恢复，根据最新观测请求原任务动作并继续执行。若仍检测到故障，则结合全局故障历史开展下一轮恢复。

异步控制的具体实现见附录A.5。

## 3.3 全局故障诊断与自回归恢复规划

### Failure与Plan的输入输出

Need触发后，模型根据当前图像、机器人状态、触觉元描述及最近的恢复计划，自回归输出全局接触故障：
\[
f_k\sim P_\theta(f\mid\mathcal X_t)
=
P_\theta
\left(f\mid\mathcal C_t,c_t,p_t^{\mathrm{ctx}}\right).
\]
故障诊断与Need使用相同输入，分别生成故障描述和二分类概率。如图1(b)所示，触觉元描述以左指传感器局部坐标系 \(\mathcal F_L\) 为参考，其方向随夹爪姿态变化。微调后的视觉语言主干结合图像中的夹爪与场景信息，在主摄参考的固定场景坐标系 \(\mathcal F_G\) 下推理故障，使不同轮次的恢复反馈具有一致的空间含义。Failure输出采用以下模板：

```text
failure[grasp_position=<label>; shear=<label>;
        grasp_force=<label>; grasp_balance=<label>;
        rotate=<label>]
```

字段含义、合法标签与坐标约定见附录A.3。

生成全局故障后，立即自回归推理恢复策略。恢复策略包含恢复动作和每个动作的幅度这两部分。恢复动作的幅度需要根据此前故障历史逐轮修正。例如，抓取未知重心物体时，如果未抓到重心，物体会向重心方向偏转。模型理解偏转方向后会推理出对应调整方向。若经过一次恢复，重新夹取后物体的偏转方向与前一次相反，表明前一次动作幅度可能过量，后续应减小动作幅度。如果没有故障历史无法完成这样的推理。因此，我们保留全局故障历史，为恢复程度的选择提供跨轮次反馈。

我们将本次诊断 \(f_k\) 与此前最近最多四条全局故障组成全局故障历史（global failure history）：
\[
\mathcal F_k=
\left(f_{\max(1,k-4)},\ldots,f_k\right).
\]
该历史最多包含五条故障，按时间排列，末项为当前故障。如图1(d)的Plan分支所示，规划仅使用通用输入C与全局故障历史F，分别对应 \(\mathcal C_t\) 和 \(\mathcal F_k\)：
\[
p_k\sim P_\theta(p\mid\mathcal C_t,\mathcal F_k).
\]
Plan以预定义字段组织恢复动作，输出模板为：

```text
plan[horizontal=<direction> <degree>;
     vertical=<direction> <degree>;
     rotate=<direction> <degree>;
     gripper_distance=<direction> <degree>]
```

其中<direction>与<degree>分别表示动作方向与幅度，不执行的字段仅填none。具体字段含义及方向、幅度类别见附录A.4。

### 复用Need前向生成Failure

Need触发后即进行Failure生成，两者使用相同输入。为避免重复计算并降低诊断延迟，我们复用Need前向计算的结果生成Failure。图1(c)中的Need池化分支与自回归解码分支共用该次主干计算。共享主干首先编码输入 \(\mathcal X_t\)，得到token表征 \(Z_t\) 和各层的KV cache \(\mathcal K_0\)：
\[
(Z_t,\mathcal K_0)
=
G_\theta^{\mathrm{prefill}}(\mathcal X_t).
\]
Need使用池化特征 \(\bar z_t\) 完成二分类；同时，前缀最后一个有效token的表征 \(h_{\mathrm{last}}\) 通过PaliGemma原有的词表投影得到Failure首token的logits：
\[
z_1=h_{\mathrm{last}}E^\top,
\]
其中 \(E\) 为输入输出共享的词嵌入矩阵。

当Need为false时，模型服务端结束本次评估，不生成Failure；为true时，模型服务端先向机器人端返回恢复触发事件，使其立即中断动作，再继续Failure的自回归解码。首先由 \(z_1\) 选择首token \(y_1\)，随后逐步将已选token与缓存输入主干：
\[
(h_i,\mathcal K_i)
=
G_\theta^{\mathrm{step}}(y_i;\mathcal K_{i-1}),
\qquad
z_{i+1}=h_iE^\top.
\]
每一步追加当前token的K/V，并读取已有前缀与生成历史，直至输出结束。

### Plan生成与文本约束

模型根据图像、机器人状态、任务指令和全局故障历史，自回归生成恢复计划：
\[
P_\theta(p_k\mid\mathcal C_t,\mathcal F_k)
=
\prod_{i=1}^{|p_k|}
P_\theta
\left(p_{k,i}\mid\mathcal C_t,\mathcal F_k,p_{k,<i}\right).
\]
Plan与Failure共用PaliGemma解码器。由于两者输入不同，生成Plan时重新编码图像和文本。每轮规划后仅保留最近四条故障；下一轮获得新诊断 \(f_{k+1}\) 后，更新全局故障历史：
\[
\mathcal F_{k+1}=\left(\operatorname{Tail}_4(\mathcal F_k),f_{k+1}\right).
\]
历史在同一任务内保留，在新任务开始时清空。

两个文本任务均采用前缀双向、回答因果的注意力：前缀内部相互可见，生成位置读取完整有效前缀及已生成token。为避免非法标签与格式错误，确保Failure与Plan输出能够被可靠解析，我们将合法字段、类别与输出顺序编码为token前缀树，根据已生成序列确定下一token合法集合 \(\mathcal A_i(y_{<i})\)，并在集合内归一化：
\[
\widetilde P_\theta(y_i\mid X,y_{<i})
=
\frac{\exp z_{i,y_i}}
{\sum_{v\in\mathcal A_i(y_{<i})}\exp z_{i,v}}.
\]
推理采用约束贪心解码，生成符合模板的故障描述与恢复计划。

## 3.4 训练目标与监督

我们先利用属性级标注独立训练触觉描述器并固定其参数，再对VLA进行两阶段训练。Stage A沿用π0.5的流匹配目标，通过正常执行与恢复示教学习原任务动作及plan条件下的恢复动作。

Stage B固定视觉编码器与动作专家，仅更新PaliGemma的LoRA参数及Need/End头。动作生成、Need/End判断和Failure/Plan生成五项任务以相同频次交替训练，保留动作监督以维持原策略的动作能力。Need/End使用分类交叉熵，Failure/Plan使用受语法约束的自回归交叉熵。联合训练目标为
\[
\mathcal L_{\mathrm B}
=\frac15\left(
\mathcal L_{\mathrm{act}}
+\mathcal L_{\mathrm{Need}}
+\mathcal L_{\mathrm{End}}
+\mathcal L_{\mathrm{failure}}
+\mathcal L_{\mathrm{plan}}
\right).
\]

为使模型依据故障历史选择恢复幅度，我们在当前观测不变的情况下构造不同的全局故障历史及对应计划。此外，Black-box中的End反事实监督保持物理观测不变，仅替换plan的程度词并调整结束标签，使模型结合计划程度与运动历史学习停止时机。损失定义、参数更新及样本构造细节见附录A.6。

<!--
正文编写依据：method整理.md及新版四板块框架图cvpr2027.png。原始整理稿保留。
1. 本章按已确认的论文版五字段Touch、五字段failure、四字段plan及failure-only历史撰写；与旧训练和部署产物的差异仍以整理稿1.12、3.14及4.14节为准，不将接口设计视为已完成的新版本评测。
2. Need阈值0.5已有确认；End采用独立验证集上Macro-F1最高的阈值选择规则，测试时固定。该规则已获确认，但实际阈值扫描、最终数值及校准结果尚待完成；正文未填写End阈值数值，不将该规则视为既有真机结果已完成重测。
3. 描述器的识别准确率、推理开销及新版数据规模尚待补齐，本章不填写性能数字或覆盖率结论。采集平台、数据规模、优化超参数及任务测试协议宜放入实验设置。
4. 图1(a–d)依次对应触觉描述器、局部到全局的故障理解、共享模型结构及部署流程；共享模型调用串行，异步不表示多个判断头和Action Expert同时执行模型前向。
5. Black-box反事实中moderately→slightly的较早结束位置由示教帧中点构造，是训练代理标签，不等同于实际位移的一半；源稿中的首token索引核对项亦不作为新增方法写入正文。
-->

# 附录备稿：模型实现与文本接口

本节收录正文移出的网络配置、Touch/Failure/Plan字段及坐标约定，供后续附录排版使用。

## A.1 输入组织与网络配置

每个时刻先拼接mesh_motion=\([D^L,P^L,D^R,P^R]\)，再拼接force=\([F^L,F^R]\)，两者分别包含12和6个通道。各通道按训练集统计量标准化，CNN输入布局为 \(B\times T\times18\times35\times20\)。左右原始信号保留各自坐标系，方向标签统一采用左指参考。

| 模块 | 具体配置 |
| --- | --- |
| 触觉窗口 | 最近30帧，包含当前帧 |
| 空间卷积块1 | 18→32，3×3卷积、BatchNorm、SiLU，随后2×2最大池化 |
| 空间卷积块2 | 32→64，3×3卷积、BatchNorm、SiLU，随后2×2最大池化 |
| 空间卷积块3 | 64→128，3×3卷积、BatchNorm、SiLU，随后全局平均池化 |
| 帧特征映射 | Linear 128→128、LayerNorm、SiLU、Dropout(0.1) |
| 时序输入映射 | Linear 128→192、LayerNorm、SiLU |
| TCN | 四个残差块，膨胀率1、2、4、8；每块含两层核大小3的一维卷积，配合GroupNorm(1)、SiLU与Dropout(0.1) |
| 特征读取 | 取末时间位置，经过LayerNorm与Dropout(0.1) |
| 字段分类 | 五个独立线性头，输出类别数依次为4、5、5、4、3 |

TCN使用窗口内时序聚合；输入仅包含当前及过去观测，其卷积实现采用对称padding。

**动作实现设置。**每次生成30步动作，部署时使用七维有效控制量。训练目标采用相对块起点状态的关节位置，反归一化后恢复绝对目标：
\[
a_{t+h}=q_t+\Delta a_{t,h},
\qquad h=0,\ldots,29.
\]

## A.2 完整字段与类别

| 字段 | 类别 | 语义 |
| --- | --- | --- |
| area | none / small / medium / full | 接触面积 |
| Fx | negative / near_zero / positive / pos_neg / neg_pos | 沿左指参考X轴的双指切向受力组合 |
| Fy | negative / near_zero / positive / pos_neg / neg_pos | 沿左指参考Y轴的双指切向受力组合 |
| Fz | near_zero / negative / left_stronger / right_stronger | 法向受力及双指负向受力幅值差异 |
| rotation | none / clockwise / counterclockwise | 左指传感器参考视角下的平面内旋转 |

Fx/Fy中的positive、negative与near_zero分别表示双指同为正向、负向及近零；pos_neg和neg_pos分别表示左正右负、左负右正。以上为左指参考下的标签语义，不对原始左右通道数值直接套用相同符号判据。

Fz中的negative表示存在负向受力且双指没有明显强弱差异；left_stronger与right_stronger表示对应手指的负向受力幅值明显更大。类别边界采用人工判读，不预设统一数值阈值。rotation的顺、逆时针按固定的左指传感器观察方向标注。

五个字段按固定顺序序列化，例如：

```text
Touch[area=medium; Fx=pos_neg; Fy=near_zero; Fz=left_stronger; rotation=none]
```

## A.3 Failure字段、标签与坐标约定

Failure结合视觉信息与左指传感器局部触觉描述，表达固定场景坐标系 \(\mathcal F_G\) 下的故障。以主摄参考工作平面定义 \(+X\) 为靠近主摄的前方、\(+Y\) 为右方、\(+Z\) 为竖直上方；该坐标系不随夹爪转动。

| 字段 | 合法标签 | 物理含义 |
| --- | --- | --- |
| grasp_position | too left / too right / too front / too back / too high / too low / appropriate | 夹取位置相对目标接触位置的偏差 |
| shear | front / back / left / right / up / down / none | 同向切向受力相关故障的全局方向 |
| grasp_force | too small / too large / appropriate | 整体夹取力度是否合适 |
| grasp_balance | front / back / left / right / up / down / appropriate | 受力偏大的接触位于夹取中心的哪一侧 |
| rotate | yaw_pos / yaw_neg / pitch_pos / pitch_neg / roll_pos / roll_neg / none | 相对目标姿态的旋转偏差 |

grasp_balance描述接触所在侧，不表示力矢量方向，也不固定对应左指或右指传感器。它与grasp_force分别描述受力失衡和整体力度，两者可以同时异常。

rotate中的yaw、pitch、roll分别对应场景固定 \(Z/Y/X\) 轴，正负遵循右手定则；这些名称用于标识偏差轴，不规定欧拉角分解顺序。标签表达当前姿态偏差，不直接表示恢复动作方向。例如，全局 \(Z\) 轴向上时，从上方俯视的逆时针偏转对应yaw_pos。

Failure示例：

```text
failure[grasp_position=too left; shear=none;
        grasp_force=appropriate; grasp_balance=appropriate;
        rotate=none]
```

## A.4 Plan字段、方向与程度

Plan包含四个动作字段。活动字段写为“方向 程度”，不活动字段只写none；字段始终按固定顺序输出。

| 字段 | 方向类别 | 程度类别 | 含义与参考 |
| --- | --- | --- | --- |
| horizontal | front / back / left / right / none | slightly / moderately / significantly | 沿固定场景坐标系的水平移动 |
| vertical | up / down / none | slightly / moderately / significantly | 沿固定场景坐标系的竖直移动 |
| rotate | clockwise / counterclockwise / none | slightly / moderately / significantly | 绕末端腕部旋转关节轴转动，顺逆时针按该轴的固定观察方向定义 |
| gripper_distance | open / close / none | slightly / moderately / significantly | 增大或减小两指间的开口距离 |

Plan中的rotate只描述夹爪开合关节之前的末端腕部旋转关节运动，与Failure中按场景固定轴定义的旋转偏差分别组织。gripper_distance控制开口变化，不直接设置夹取力。

程度词为人工定性标注，不绑定固定毫米数、旋转角度或夹爪间距范围，其实际运动幅度由动作专家与End判断共同实现。方向为none时不附加程度。Plan示例：

```text
plan[horizontal=right slightly; vertical=none;
     rotate=none; gripper_distance=none]
```

## A.5 异步控制实现

控制器在发布每条动作指令前检查停止标志和动作版本。Need或End触发后，停止发送当前动作块的剩余指令并保持最新反馈位置；阶段切换时更新版本号，拒收迟到的动作或评估结果。新阶段的有效动作块返回后才恢复运动。

后台最多保留一个未完成的评估请求。该请求结束后，系统根据更新的观测开展下一次判断，避免旧状态的评估积压。共享模型的推理仍按顺序执行，异步发生在物理运动与状态判断之间。

## A.6 训练目标与监督构造细节

### 数据与参数更新

正常执行与恢复示教通过主从臂遥操作采集。恢复计划的方向与程度由人工标注，Failure由人工结合视觉与触觉判读；Need/End标签依据接触事件和示教阶段边界构造。先按原始回合划分训练、验证和测试集合，各类窗口、历史变体及反事实样本继承所属回合的划分。

Stage A通过LoRA适配语言主干与动作专家，并更新视觉编码器、动作投影和时间MLP；两个Transformer主干的非LoRA参数保持固定。Stage B加载Stage A模型，冻结视觉编码器、动作专家、动作适配参数及语言主干非LoRA参数，仅更新PaliGemma LoRA和Need/End头。四个操作任务分别训练对应VLA，触觉描述器独立训练后固定。

### 触觉字段监督

令 \(\pi_\psi^{(j)}\) 为字段预测分布，\(m_{ij}\) 表示样本 \(i\) 的字段 \(j\) 是否有有效标注，损失为
\[
\mathcal L_{\mathrm{touch}}
=
\frac{
\sum_{i,j}m_{ij}\,
\operatorname{CE}_{w}
\left(\pi_\psi^{(j)}(\cdot\mid\mathcal T_i),y_i^{(j)}\right)}
{\sum_{i,j}m_{ij}}.
\]
每个片段只监督一个已标注字段，其他字段不产生损失。类别权重与训练集类别频次的平方根倒数成比例，按字段归一化，并使用标签平滑。

### 动作与文本损失

动作目标相对块起点的当前qpos定义，归一化后适配为 \(A\in\mathbb R^{30\times32}\)。内部32维中包含七维有效机器人控制量。采样高斯噪声 \(\epsilon\) 与流时间 \(\tau\)，构造
\[
A_\tau=(1-\tau)A+\tau\epsilon,
\qquad
\mathcal L_{\mathrm{act}}
=
\mathbb E
\left[
\frac{
\left\|v_\theta(A_\tau,\tau\mid\mathcal C_t,s_t,p_k)
-(\epsilon-A)\right\|_F^2}
{30\cdot32}
\right].
\]
其中计划仅在恢复阶段进入动作条件。推理从高斯噪声出发，以10步Euler积分生成动作块，反归一化后恢复绝对控制目标。

Need/End使用普通二分类交叉熵。Failure/Plan采用教师强制，按3.3节的合法token概率计算回答部分的损失：
\[
\mathcal L_{\mathrm{text}}
=
-\frac{1}{\sum_i a_i}
\sum_i a_i
\log\widetilde P_\theta(y_i\mid X,y_{<i}),
\]
其中 \(a_i\) 仅标记有效回答token，输入与填充位置不参与损失。训练按action→End→Need→failure→plan循环，每次更新只使用当前任务的一个batch，各项损失权重均为1。

等频任务采样对应的总体目标为
\[
\mathcal L_{\mathrm B}
=\frac15\left(
\mathcal L_{\mathrm{act}}
+\mathcal L_{\mathrm{Need}}
+\mathcal L_{\mathrm{End}}
+\mathcal L_{\mathrm{failure}}
+\mathcal L_{\mathrm{plan}}
\right).
\]
保留动作监督以约束语义适配过程中的共享表征，动作损失通过固定的动作专家回传至PaliGemma LoRA。


### 故障历史与End反事实

规划训练中的历史来自人工全局故障标注与规则变体。构造时固定当前观测，改变故障历史及与其一致的计划目标。在线恢复则使用模型此前实际生成的Failure，按3.3节维护有限历史。

Black-box的End反事实保持同一物理观测，仅替换计划中的slightly与moderately。设示教恢复运动的起止帧为 \(b,e\)，moderately替换为slightly时，用帧中点构造代理结束标签：
\[
y_{\mathrm{End}}^{\mathrm{moderately}\rightarrow\mathrm{slightly}}(u)
=\mathbb I\!\left[u\geq\left\lfloor(b+e)/2\right\rfloor\right],
\qquad b\leq u\leq e.
\]
slightly替换为moderately时，已观测的 \([b,e]\) 区间均标为未完成。帧中点不代表物理位移的一半，反事实样本仅用于End监督，动作训练仍使用真实示教。

# 英文 LaTeX：Method开头与3.1—3.4

```latex
\section{Method}
\label{sec:method}

TARO aims to interpret contact failures from dynamic tactile feedback and generate recovery strategies from failure history to enable autonomous multi-round recovery during task execution. As shown in Fig.~\ref{fig:framework}(d), the system alternates between execution and recovery. During execution, Need runs asynchronously with physical action execution to determine whether recovery is required. A positive decision interrupts the current action, after which the model diagnoses global failures from visual and tactile observations and generates a plan from the global failure history. During recovery, the action model follows this plan while End runs asynchronously with physical action execution to determine whether the current recovery operation is complete. A positive decision interrupts the recovery action and returns the robot to task execution. If another failure is detected, the new diagnosis is added to the history to guide the next recovery round.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{框架图/cvpr2027.png}
    \caption{Overview of TARO. (a) Temporal tactile description. (b) Local-to-global failure interpretation. (c) Shared backbone for state assessment, failure diagnosis, recovery planning, and action generation. (d) Asynchronous assessment during physical execution enables transitions between task execution and recovery. C/T/M/P/F denote common inputs, tactile text, motion history, recovery plan, and global failure history.}
    \label{fig:framework}
\end{figure*}

\subsection{Lightweight Temporal Tactile Descriptor}
\label{sec:tactile_descriptor}

\paragraph{Tactile signals.}
To reduce reliance on tactile image appearance and facilitate cross-sensor generalization, we use three tactile signals (Fig.~\ref{fig:framework}(a)): \texttt{mesh\_3d}, the current 3D contact point cloud; \texttt{mesh\_3d\_flow}, the displacement vector field from the initial point cloud to the current one; and \texttt{force}, the 3D force distribution. Each signal has dimensions $35\times20\times3$. After normalization, signals from both fingers are concatenated along the channel dimension in a fixed left--right order to form an 18-channel grid $x_t$. The latest $T=30$ frames form the input window $\mathcal{T}_t$.

\paragraph{Spatial and temporal encoding.}
To capture spatial contact patterns and their dynamics while mitigating ambiguities in individual frames, a shared CNN extracts frame features $e_\tau=E_{\mathrm{CNN}}(x_\tau)$, which are aggregated by a temporal convolutional network (TCN):
\begin{equation}
u_t=\left[E_{\mathrm{TCN}}(e_{t-T+1},\ldots,e_t)\right]_T.
\end{equation}
The final temporal feature $u_t$ summarizes the window up to the current time. Network depth, feature dimensions, and convolution configurations are provided in the appendix.

\paragraph{Attribute prediction and textual description.}
We represent common contact states using a compact set of discrete tactile attributes. Specifically, we define five attributes covering contact area, three-axis forces, and in-plane rotation: \texttt{area}, \texttt{Fx}, \texttt{Fy}, \texttt{Fz}, and \texttt{rotation}. The \texttt{Fx}/\texttt{Fy} attributes capture tangential forces acting in the same or opposite directions across the fingers, while \texttt{Fz} describes normal forces and differences in their magnitudes. Five independent classification heads predict discrete labels from $u_t$, which are converted into tactile text $c_t$ using a template with a fixed attribute order. Because the opposing sensors observe the same contact rotation in opposite directions, we define directional labels in the left finger sensor's local frame $\mathcal{F}_L$. For example, a rotation appearing clockwise to the left sensor and counterclockwise to the right is labeled \texttt{rotation=clockwise}. Complete attribute values and the text template are provided in the appendix.

\subsection{Two-Phase Action Execution and Asynchronous Assessment}
\label{sec:execution_recovery}

\paragraph{Action conditioning across phases.}
As shown in Fig.~\ref{fig:framework}(c), TARO retains the vision-language backbone and action expert of $\pi_{0.5}$, generating action chunks through iterative denoising with flow matching. Common inputs, denoted by C, are $\mathcal{C}_t=(I_t^{\mathrm{front}},I_t^{\mathrm{wrist}},q_t,\ell)$, comprising current front and wrist images, robot state, and task instruction. Both images are encoded by the shared SigLIP encoder and form the multimodal prefix together with the textualized state and task instruction.

Execution and recovery share the same action expert, with \texttt{Mode: execution} and \texttt{Mode: adjustment} included in the text component of $\mathcal{C}_t$ to distinguish the two phases. Recovery additionally conditions on the current plan $p_k$:
\begin{equation}
A_t\sim
\begin{cases}
\pi_\theta(A\mid\mathcal{C}_t), & s_t=\mathrm{E},\\
\pi_\theta(A\mid\mathcal{C}_t,p_k), & s_t=\mathrm{R},
\end{cases}
\end{equation}
where $s_t$ denotes the current phase, $\mathrm{E}$ and $\mathrm{R}$ denote execution and recovery, respectively, and $k$ indexes recovery decisions.

\paragraph{Need/End inputs and classification.}
Execution aims to advance the original task, while recovery aims to correct failures; we therefore model these distinct manipulation intents as separate execution and recovery phases. To determine phase transitions, we introduce two independent binary heads, Need and End, on the shared PaliGemma backbone (Fig.~\ref{fig:framework}(c)). Need determines whether recovery is required during execution, using common inputs, the current tactile description, and plan context. During recovery, End additionally uses motion history to determine when to stop according to the magnitude specified in the plan, thereby controlling the extent of recovery motion:
\begin{equation}
\begin{aligned}
\mathcal{X}_t&=(\mathcal{C}_t,c_t,p_t^{\mathrm{ctx}}),\\
\mathcal{X}_t^{\mathrm{End}}&=(\mathcal{C}_t,c_t,M_t,p_k).
\end{aligned}
\end{equation}
The C, T, M, and P tags in Fig.~\ref{fig:framework}(d) denote common inputs, tactile text, motion history, and plan context, respectively. Here, $c_t$ is the current tactile description, $p_t^{\mathrm{ctx}}$ is the most recent recovery plan or \texttt{none} if unavailable, and $p_k$ is the active recovery plan. The motion history $M_t$ is used only by End, with $t$ denoting the current observation time. Under the H100 setting, it consists of 11 uniformly sampled states from the latest 100 frames, quantized into text, with the last sample equal to the current state. At the synchronized training-data sampling rate of 30 Hz, this window spans approximately 3.3 seconds; unavailable initial frames are padded with the episode's first state. During inference, timestamped robot feedback is resampled onto the same 30 Hz time grid before selecting and quantizing the 11 states. This history provides evidence of executed motion for assessing recovery progress relative to the planned magnitude.

As shown by the mean-pooling branches in Fig.~\ref{fig:framework}(c), valid backbone token features are averaged into $\bar z_t$ and projected to two logits through two linear layers in the corresponding head. A softmax yields the recovery-trigger probability $P_{\mathrm{Need}}$ or recovery-completion probability $P_{\mathrm{End}}$. Recovery starts when $P_{\mathrm{Need}}\geq\eta_{\mathrm{Need}}$, and the current recovery operation ends when $P_{\mathrm{End}}\geq\eta_{\mathrm{End}}$. We fix $\eta_{\mathrm{Need}}=0.5$ and select $\eta_{\mathrm{End}}$ to maximize End macro-F1 on a held-out validation set, keeping it fixed during real-world testing.

\paragraph{Asynchronous coordination with physical action execution.}
As shown in Fig.~\ref{fig:framework}(d), the model is deployed on a model server, while the robot client collects observations, sends inference requests, and controls the robot. The robot client publishes returned action commands at 30 Hz and requests phase-specific state assessments from updated observations at approximately 3 Hz. The model server processes inference requests serially, while state assessment overlaps physical action execution, enabling phase transitions before an action chunk finishes.

When Need triggers, the robot client interrupts and pauses the current action, then executes recovery actions after the model server completes failure diagnosis and planning. When End triggers, the robot client terminates the current recovery operation and requests task actions from the latest observations to resume execution. Further detected failures initiate additional recovery rounds using the global failure history. Implementation details of asynchronous control are provided in Appendix~A.5.

\subsection{Global Failure Diagnosis and Autoregressive Recovery Planning}
\label{sec:failure_recovery}

\paragraph{Failure and Plan inputs and outputs.}
When Need triggers, the model autoregressively generates a global contact failure description from the current images, robot state, tactile description, and most recent recovery plan:
\begin{equation}
\begin{aligned}
f_k&\sim P_\theta(f\mid\mathcal{X}_t)\\
&=P_\theta(f\mid\mathcal{C}_t,c_t,p_t^{\mathrm{ctx}}).
\end{aligned}
\end{equation}
Failure diagnosis and Need use the same inputs to produce a failure description and a binary classification probability, respectively. As shown in Fig.~\ref{fig:framework}(b), tactile descriptions are expressed in the left sensor's local frame $\mathcal{F}_L$, whose orientation changes with the gripper. The fine-tuned vision-language backbone combines these descriptions with visual information about the gripper and scene to interpret failures in a fixed scene frame $\mathcal{F}_G$ defined with respect to the front camera. This gives feedback from different recovery rounds a consistent spatial meaning. Failure uses the following template:
\begin{quote}
\small\ttfamily
failure[grasp\_position=<label>;\\
shear=<label>; grasp\_force=<label>;\\
grasp\_balance=<label>; rotate=<label>]
\end{quote}
Field definitions, valid labels, and coordinate conventions are provided in Appendix~A.3.

After generating the global failure description, the model autoregressively infers a recovery strategy specifying actions and their magnitudes. Recovery magnitudes must adapt across rounds based on the failure history. For example, an object with an unknown center of mass may tilt toward its center of mass when grasped off-center. The model interprets this tilt to infer a corrective direction. If the tilt reverses after regrasping, the previous correction may have overshot, calling for a smaller subsequent movement. Recognizing this reversal requires feedback from previous attempts. We therefore retain a global failure history to guide magnitude selection across recovery rounds.

The global failure history contains the current diagnosis $f_k$ and up to four preceding failures:
\begin{equation}
\mathcal{F}_k=(f_{\max(1,k-4)},\ldots,f_k).
\end{equation}
It contains at most five failures in chronological order, with the current failure last. As shown by the Plan branch in Fig.~\ref{fig:framework}(d), planning uses only common inputs C and global failure history F, corresponding to $\mathcal{C}_t$ and $\mathcal{F}_k$:
\begin{equation}
p_k\sim P_\theta(p\mid\mathcal{C}_t,\mathcal{F}_k).
\end{equation}
Plan organizes recovery actions into predefined fields:
\begin{quote}
\small\ttfamily
plan[horizontal=<direction> <degree>;\\
vertical=<direction> <degree>;\\
rotate=<direction> <degree>;\\
gripper\_distance=<direction> <degree>]
\end{quote}
Here, \texttt{<direction>} and \texttt{<degree>} specify action direction and magnitude; inactive fields contain only \texttt{none}. Field definitions and direction and magnitude categories are provided in Appendix~A.4.

\paragraph{Reusing Need computation for Failure generation.}
Failure generation immediately follows a positive Need decision and uses the same inputs. To avoid redundant computation and reduce diagnosis latency, we reuse the Need forward pass for Failure generation. The Need pooling branch and autoregressive decoding branch in Fig.~\ref{fig:framework}(c) share this backbone computation. A single prefill encodes $\mathcal{X}_t$ into token representations $Z_t$ and the per-layer KV cache $\mathcal{K}_0$:
\begin{equation}
(Z_t,\mathcal{K}_0)=G_\theta^{\mathrm{prefill}}(\mathcal{X}_t).
\end{equation}
Need classifies the pooled feature $\bar z_t$. PaliGemma's existing vocabulary projection also maps the last valid prefix token representation $h_{\mathrm{last}}$ to logits for the first Failure token:
\begin{equation}
z_1=h_{\mathrm{last}}E^\top,
\end{equation}
where $E$ is the shared input/output embedding matrix.

For a negative Need decision, the model server ends the assessment without generating Failure. For a positive decision, the model server first sends a recovery-trigger event to the robot client to interrupt the current action, then continues autoregressive Failure decoding. The first token $y_1$ is selected from $z_1$; subsequent steps process the selected token together with the cache:
\begin{equation}
\begin{aligned}
(h_i,\mathcal{K}_i)&=G_\theta^{\mathrm{step}}(y_i;\mathcal{K}_{i-1}),\\
z_{i+1}&=h_iE^\top.
\end{aligned}
\end{equation}
Each step appends the current token's keys and values and attends to the existing prefix and generated tokens until the output is complete.

\paragraph{Plan generation and constrained decoding.}
The model autoregressively generates a recovery plan from the images, robot state, task instruction, and global failure history:
\begin{equation}
\begin{aligned}
&P_\theta(p_k\mid\mathcal{C}_t,\mathcal{F}_k)\\
&\quad=\prod_{i=1}^{|p_k|}
P_\theta(p_{k,i}\mid\mathcal{C}_t,\mathcal{F}_k,p_{k,<i}).
\end{aligned}
\end{equation}
Plan and Failure share the PaliGemma decoder. As their inputs differ, planning re-encodes the images and text. After each planning step, we retain the four most recent failures. When the next diagnosis $f_{k+1}$ becomes available, we update the global failure history:
\begin{equation}
\mathcal{F}_{k+1}=(\operatorname{Tail}_4(\mathcal{F}_k),f_{k+1}).
\end{equation}
The history persists within a task and is cleared when a new task begins.

Both text tasks use bidirectional attention within the input prefix and causal attention over the answer. Prefix tokens attend to one another, while generated positions attend to the full valid prefix and preceding answer tokens. To prevent invalid labels and formatting errors and ensure that Failure and Plan outputs can be reliably parsed, we encode valid fields, categories, and output order in a token prefix tree. Given the generated sequence, the tree determines the allowed next-token set $\mathcal{A}_i(y_{<i})$, over which probabilities are normalized:
\begin{equation}
\widetilde P_\theta(y_i\mid X,y_{<i})
=
\frac{\exp z_{i,y_i}}
{\sum_{v\in\mathcal{A}_i(y_{<i})}\exp z_{i,v}}.
\end{equation}
Constrained greedy decoding produces failure descriptions and recovery plans that follow the prescribed templates.

\subsection{Training Objectives and Supervision}
\label{sec:training}

We first train the tactile descriptor independently with attribute-level annotations and freeze it, then adapt the VLA in two stages. Stage A retains the flow-matching objective of $\pi_{0.5}$ to learn task execution and plan-conditioned recovery from demonstrations.

Stage B freezes the visual encoder and action expert, updating only the PaliGemma LoRA parameters and Need/End heads. The five tasks---action generation, Need/End classification, and Failure/Plan generation---are trained in alternation with equal frequency. Retaining action supervision preserves the policy's action capabilities. Need/End use classification cross-entropy, while Failure/Plan use grammar-constrained autoregressive cross-entropy. The joint objective is
\begin{equation}
\begin{aligned}
\mathcal{L}_{\mathrm{B}}=\frac15\bigl(
&\mathcal{L}_{\mathrm{act}}
+\mathcal{L}_{\mathrm{Need}}
+\mathcal{L}_{\mathrm{End}}\\
&+\mathcal{L}_{\mathrm{failure}}
+\mathcal{L}_{\mathrm{plan}}\bigr).
\end{aligned}
\end{equation}

To learn recovery magnitudes from failure history, we construct alternative global failure histories and corresponding plans while holding current observations fixed. We further introduce counterfactual End supervision in Black-box: physical observations remain unchanged, while the plan's magnitude word and completion label are modified. This trains End to determine stopping times from both the planned magnitude and motion history. Loss definitions, parameter updates, and sample construction details are provided in Appendix~A.6.

```
