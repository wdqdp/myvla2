# 方法（Method，逐节确认稿）

本文利用动态触觉理解接触故障，并根据恢复历史推理后续恢复策略，实现任务执行中的在线多轮自主恢复。本文件作为论文方法部分的详细底稿，按“模型结构—训练方式—推理方式—数据采集以及处理”的顺序逐节撰写和确认，后续再压缩为会议论文正文。

当前进度：第一、二节已确认；第三节的动作发布频率、判断频率、采样步数与部署checkpoint已确认，adjustment_end阈值及其校准方案待确认；第四节采集与标注方案已确认，五字段数据规模及部分平台标定信息待补充，见4.14节。代码核对日期：2026-10-22。本文中的Touch、failure与plan采用此前确定的论文版接口；这些接口与部分现有部署代码的差异记录在1.12节，不能将接口草案直接等同于已有checkpoint的输出能力。

## 一、模型结构

### 1.1 整体结构与问题定义

面向接触操作中的执行异常，我们将机器人控制划分为正常执行与恢复两个阶段。正常执行阶段完成原始任务；恢复阶段依据故障诊断结果生成恢复计划，并由动作模型执行相应操作。恢复触发与恢复结束分别由两个状态判断头预测。恢复是否结束与原任务是否成功分别判断，结束一个恢复阶段后，系统仍需继续执行原任务。

模型由两个学习模块及一项外部记忆组成：

1. **时序触觉描述器。**对双指触觉网格进行空间编码和时序聚合，将连续触觉窗口转换为离散元描述文本Touch。
2. **共享视觉—语言—动作模型。**在π0.5的视觉语言主干和动作专家上增加恢复触发、恢复结束两个分类头，并利用共享语言解码器完成故障诊断与恢复规划。
3. **全局故障历史。**保存此前恢复过程中得到的全局failure序列，作为下一次恢复规划的上下文。记忆由控制程序维护，不引入额外的记忆编码网络。

设时刻\(t\)的通用观测为

\[
C_t=(I_t^{\mathrm{front}},I_t^{\mathrm{wrist}},q_t,\ell),
\]

其中\(I_t^{\mathrm{front}}\)和\(I_t^{\mathrm{wrist}}\)分别表示主摄与腕部相机的当前图像，\(q_t\in\mathbb{R}^{7}\)为机器人状态，\(\ell\)为任务指令。七维状态包含六个机械臂关节以及夹爪开合状态。以下进一步定义：

| 符号 | 内容 | 用途 |
| --- | --- | --- |
| \(\mathcal T_t\) | 截至当前时刻的30帧双指触觉网格 | 输入触觉描述器 |
| \(c_t\) | 当前Touch元描述文本 | 接触状态判断与故障诊断 |
| \(H_t\) | 最近100帧qpos采样为11点的运动历史 | 恢复触发、故障诊断与恢复结束判断 |
| \(f_k\) | 第\(k\)次规划所依据的当前全局failure | 定义当前接触故障 |
| \(\mathcal M_k\) | 当前failure之前最近最多4条全局failure | 为恢复规划提供历史反馈 |
| \(p_k\) | 当前恢复计划plan | 条件化恢复动作生成与状态评估 |
| \(s_t\) | 当前阶段，execution或adjustment | 选择动作条件与状态判断任务 |

\(t\)用于表示观测时间，\(k\)用于表示恢复决策序号。触觉窗口、运动历史与故障历史分别记录不同层面的时间信息：触觉窗口描述接触演化，运动历史描述已经发生的机器人运动，故障历史描述多次恢复后的全局接触反馈。

### 1.2 观测表示与坐标约定

**视觉输入。**两路当前图像经过尺寸调整后输入共享SigLIP视觉编码器，当前实现使用\(224\times224\)输入。两路视觉token均进入视觉语言前缀，使模型同时观察物体整体关系与末端局部接触区域。模型使用当前图像，不将历史图像序列作为默认输入。

**机器人状态。**当前qpos按照训练数据统计量进行归一化，并依据π0.5的协议离散成State文本。其作用是提供当前关节与夹爪状态。当前qpos对所有任务均可用；额外的运动历史仅进入指定的判断和诊断任务。

**三个参考系的区分。**

- **左指传感器局部坐标系\(\mathcal F_L\)。**Touch的切向受力方向与平面内旋转以该坐标系为参考；左、右传感器身份固定。
- **主摄参考的固定场景坐标系\(\mathcal F_G\)。**failure的位置、切向方向、受力失衡侧与旋转偏差，以及plan的水平／垂直移动方向均在此定义。该坐标系固定于场景，不随腕部相机或夹爪转动。
- **末端腕部旋转关节参考。**plan中的rotate描述夹爪绕夹爪开合关节之前的末端旋转关节轴转动，其顺、逆时针按固定观察方向定义。

**原始信号与标签参考的区别。**左右手指的原始触觉信号分别保留各自传感器局部坐标系，按固定身份直接拼接，输入前不执行统一坐标变换。Touch标签的方向语义统一以左指传感器局部坐标系定义。因而，标签参考一致不表示左右原始信号已经数值对齐到同一坐标系；描述器根据双指原始观测学习到该语义标签的映射。

**已确认C1：场景坐标系约定。**以主摄参考的工作平面为基础，令\(+X\)沿工作平面靠近主摄、\(+Y\)指向主摄画面右侧、\(+Z\)竖直向上，组成右手坐标系。前／后表示工作平面内的深度方向，与图像中的上／下分开定义。

| 全局轴方向 | 空间标签 | 物理约定 |
| --- | --- | --- |
| \(+X\)／\(-X\) | front／back | 沿工作平面靠近／远离主摄 |
| \(+Y\)／\(-Y\) | right／left | 主摄参考下向右／向左 |
| \(+Z\)／\(-Z\) | up／down | 竖直向上／向下 |

failure的roll、pitch和yaw分别表示绕场景固定X、Y和Z轴的偏差，正负遵循右手定则。这些名称在本文中用于标识固定轴；标签排列不表示特定欧拉角分解顺序。上述轴与方向的对应已由用户确认；具体相机外参、场景坐标标定或人工标注方式在第四节记录。

### 1.3 双指时序触觉描述器

#### 1.3.1 输入信号及通道组织

每个手指传感器提供空间分辨率为\(35\times20\)的三维网格信号，包含三维网格位置\(P\)、网格运动\(D\)以及三轴力场\(F\)。这些原始量保留各自传感器的局部坐标表达，沿通道维直接组合左右手指信号：

\[
M_t=\operatorname{Concat}(D_t^L,P_t^L,D_t^R,P_t^R)
\in\mathbb R^{12\times35\times20},
\]

\[
F_t=\operatorname{Concat}(F_t^L,F_t^R)
\in\mathbb R^{6\times35\times20}.
\]

因此，每个时刻共有18个输入通道。模型取截至当前时刻的30帧窗口，形成

\[
\mathcal T_t=\{(M_\tau,F_\tau)\}_{\tau=t-29}^{t}.
\]

批量输入形状分别为\(B\times30\times12\times35\times20\)与\(B\times30\times6\times35\times20\)。网格位置与运动、力场分别按通道统计量标准化，随后拼接为18通道输入。描述器直接使用网格与力场，不增加合力向量输入分支。

#### 1.3.2 逐帧空间编码

对窗口内各帧应用参数共享的二维CNN：

\[
e_\tau=E_{\mathrm{CNN}}\bigl([M_\tau;F_\tau]\bigr).
\]

代码中空间编码器由三层\(3\times3\)卷积、两次最大池化以及全局平均池化组成。卷积后使用BatchNorm与SiLU，最终通过线性投影得到帧特征。

| 操作 | 默认输出形状，省略批量与时间维 | 具体设置 |
| --- | --- | --- |
| 输入拼接 | \(18\times35\times20\) | mesh_motion与force沿通道拼接 |
| 卷积块1 | \(32\times35\times20\) | Conv 3×3，padding=1，BatchNorm，SiLU |
| 最大池化1 | \(32\times17\times10\) | 2×2池化 |
| 卷积块2 | \(64\times17\times10\) | Conv 3×3，padding=1，BatchNorm，SiLU |
| 最大池化2 | \(64\times8\times5\) | 2×2池化 |
| 卷积块3 | \(128\times8\times5\) | Conv 3×3，padding=1，BatchNorm，SiLU |
| 自适应平均池化与展平 | \(128\) | 池化到1×1 |
| 帧特征投影 | \(128\) | Linear，LayerNorm，SiLU，Dropout |

#### 1.3.3 时序聚合与字段分类

帧特征首先由线性层映射到192维，经过LayerNorm与SiLU后进入时序卷积网络。TCN包含四个残差时序块，dilation依次为1、2、4、8。每个块包含两层核大小为3的一维卷积，各层后接GroupNorm、SiLU与Dropout，再与块输入进行残差相加。GroupNorm使用一个分组。

令时序聚合结果为

\[
U_t=E_{\mathrm{TCN}}(e_{t-29},\ldots,e_t).
\]

取末时间位置的192维特征\(u_t=U_t[-1]\)，经过LayerNorm与Dropout后送入多个独立线性分类头：

\[
\hat y_t^{(j)}=
\arg\max\operatorname{softmax}(W_ju_t+b_j).
\]

各字段共享空间与时序编码，仅最后的分类层独立。分类结果经固定模板拼接为Touch文本，不使用大语言模型逐token生成触觉描述。连续触觉信号在此被转换为可直接进入语言模型的离散语义接口。

网络使用截至当前时刻的窗口完成在线预测，输入不包含未来帧。当前TCN内部采用对称padding，因此正文描述为“窗口内时序聚合”，不将其标注为严格的因果卷积网络。

**已确认C2：描述器具体配置。**论文沿用窗口30、帧特征128维、时序隐藏192维、四个残差时序块的dilation为1／2／4／8、dropout=0.1。当前部署配置记录的权重路径不存在，这套结构参数由代码与用户确认，未从该权重重新读取。参数量、准确率及推理开销分别以实际统计和测量为准。

### 1.4 触觉元描述的语义接口

依据已确定的论文设计，Touch包含五个字段。各字段描述接触状态，不直接给出机器人恢复方向。

| 字段 | 类别 | 含义及参考 |
| --- | --- | --- |
| area | none／small／medium／full | 接触面积状态 |
| Fx | negative／near_zero／positive／pos_neg／neg_pos | 左指传感器局部X方向上的双指切向受力组合 |
| Fy | negative／near_zero／positive／pos_neg／neg_pos | 左指传感器局部Y方向上的双指切向受力组合 |
| Fz | near_zero／negative／left_stronger／right_stronger | 法向受力状态及双指负向受力幅值差异 |
| rotation | none／clockwise／counterclockwise | 以左指传感器为参考的平面内旋转 |

Fx和Fy中，positive／negative表示双指分别同为正向／负向，near_zero表示双指均近零；pos_neg表示左正右负，neg_pos表示左负右正。这里的切向正负按标签约定的左指传感器参考方向解释，不是直接比较左右原始通道的数值符号。左右顺序是固定传感器身份，不随图像中两指的位置变化。Fz的left_stronger和right_stronger同样按传感器身份区分，表示对应手指的负向受力幅值明显更大。

论文接口对应五个分类头，其输出类别数依次为4、5、5、4、3。文本形式例如：

```text
Touch[area=medium; Fx=pos_neg; Fy=near_zero; Fz=left_stronger; rotation=none]
```

rotation的顺、逆时针以左指传感器局部参考视角判定，传感器的固定观察方向及Fx／Fy／Fz符号约定需在数据标注协议中配图明确。这里的rotation与failure中按场景轴定义的姿态偏差，以及plan中按腕部关节定义的旋转动作，分别承担不同语义。

### 1.5 共享视觉语言主干

主体基于π0.5，由SigLIP视觉编码器、PaliGemma语言主干与动作专家构成。当前保存配置使用gemma_2b_lora与gemma_300m_lora，分别对应2048维的语言主干与1024维的动作专家，二者均为18层。LoRA的训练范围和阶段安排在第二节展开。

共享SigLIP采用So400m/14配置，对两路图像分别提取空间token，并映射到语言主干宽度。任务指令、离散状态、阶段条件及任务所需的其他文本通过同一语言词表编码。视觉token与文本token共同构成多模态前缀：

\[
Z_t^{(r)}=
G_\theta\left(
E_V(I_t^{\mathrm{front}}),
E_V(I_t^{\mathrm{wrist}}),
E_L\bigl(\operatorname{Prompt}_r(q_t,\ell,\ldots)\bigr)
\right),
\]

其中\(r\)表示具体任务，\(\operatorname{Prompt}_r\)选择该任务需要的条件。不同任务共用主干参数，但组织各自的输入前缀；不存在一条将所有历史、Touch与plan无差别发送给全部分支的固定输入通路。

输入前缀中的有效视觉与文本token相互可见；语言生成目标使用自回归注意力，逐token读取完整前缀及此前生成内容。动作专家的动作token读取多模态前缀，在流匹配时间条件下预测连续动作。具体注意力掩码随分类、文本生成和动作生成任务选择。

### 1.6 运动历史的文本化表示

当前qpos仅给出当前状态，不能单独反映此前执行了多少运动。为判断故障与恢复进度，额外构造最近100帧qpos历史，按时间均匀采样11个点，并保留当前帧作为最后一点：

\[
H_t=\left[q_{t-99+s_i}\right]_{i=0}^{10},
\qquad
s_i=\left\lfloor\frac{99i}{10}\right\rfloor.
\]

在标称30 Hz下，该窗口覆盖约3.3秒。采样不跨任务回合，不读取未来状态；任务开始时不足100帧，使用本回合首帧左侧填充。历史允许连续跨越回合内的执行／恢复阶段，不能在每次阶段切换时将其当成新的完整轨迹。

每个关节或夹爪状态先按训练集的q01与q99分位数归一化：

\[
\tilde q_{t,j}=
2\frac{q_{t,j}-q_j^{01}}
{q_j^{99}-q_j^{01}+\varepsilon}-1.
\]

然后沿用π0.5的256档阈值离散化，形成\(11\times7\)的整数文本。代码使用digitize并且不额外截断，因此不将所有越界状态强制宣称为0—255范围。历史文本直接进入语言前缀；当前配置关闭连续state-history分支，不额外使用GRU编码qpos历史。

H100仅输入need、failure与adjustment_end。动作专家和plan不直接读取该运动历史。这里的历史描述机器人已经发生的运动，与下一节全局failure记忆分别组织。

### 1.7 恢复触发与恢复结束判断

恢复触发need_recovery与恢复结束adjustment_end共享多模态主干，使用两个参数独立的MLP分类头。两个头的输入组织方式相同：通用输入\(C_t\)、当前Touch\(c_t\)、运动历史\(H_t\)及用于评估的当前恢复计划；无计划时使用none。两者的监督语义与有效调用阶段不同。

#### 1.7.1 掩码池化与分类头

得到前缀输出\(Z=(z_1,\ldots,z_N)\)后，对有效token做掩码均值池化：

\[
\bar z=
\frac{\sum_i m_i z_i}{\max(\sum_i m_i,1)},
\]

其中\(m_i\)同时屏蔽无效相机槽及填充token。池化沿token维进行，将\(B\times N\times2048\)变为\(B\times2048\)，不改变特征维度。平均对象是经过多模态注意力融合后的特征；当前池化本身不学习权重，也不按模态分别加权。每个头依次使用LayerNorm、Linear、GELU、Dropout和输出Linear，结构为2048→512→2，dropout为0.1；推理时关闭Dropout。

\[
P(n_t=1)=\operatorname{softmax}(h_{\mathrm{need}}(\bar z))_1,
\qquad
P(e_t=1)=\operatorname{softmax}(h_{\mathrm{end}}(\bar z))_1.
\]

need判断正常执行中是否需要恢复，end判断当前adjustment阶段是否应结束。结束当前恢复阶段不等价于判定原任务完成；后续正常执行仍可能发现残余故障。分类阈值、判断频率以及结果如何触发阶段切换，在第三节说明。

#### 1.7.2 与专用判断token的取舍

另一种可行方式是在输入前缀加入可学习的判断token，并取其输出特征送入分类头。该token通过注意力学习汇总任务相关信息；由于本项目的观测前缀采用双向注意力，位于前缀内的判断token可以读取完整有效前缀。

当前保留掩码均值池化，以复用已有输入和训练接口。专用token可能提供更有针对性的信息汇总，但是否优于均值池化需在相同训练条件下验证，不能仅据结构宣称优势。[ViT附录D.3](https://arxiv.org/pdf/2010.11929)也表明，两类读取方式的比较受优化设置影响。若增加专用token，须将新增embedding或独立向量纳入训练；当前Stage B冻结原词嵌入，仅增加词表项而不调整训练范围并不足够。

### 1.8 结合视觉的全局故障诊断

触觉元描述基于左指传感器局部方向，不能脱离夹爪和物体姿态直接解释成场景中的运动方向。故障诊断将当前视觉、qpos、Touch、运动历史和当前评估计划共同输入共享主干，由语言解码器生成全局failure：

\[
f_k\sim p_\theta\bigl(f\mid C_t,c_t,H_t,p_t^{\mathrm{ctx}}\bigr).
\]

诊断同时考虑接触证据与视觉中的夹爪—物体关系，输出主摄参考固定场景坐标系下的故障状态。该过程通过视觉与触觉联合推理实现，不增加未经实现的几何坐标变换网络，也不将局部触觉标签直接按固定字典替换为全局方向。

论文版failure包含五个字段：

| 字段 | 类别 | 诊断内容 |
| --- | --- | --- |
| grasp_position | too left／too right／too front／too back／too high／too low／appropriate | 夹取位置相对目标接触位置的偏差 |
| shear | front／back／left／right／up／down／none | 同向切向受力相关故障的场景方向 |
| grasp_force | too small／too large／appropriate | 整体夹取力度是否合适 |
| grasp_balance | front／back／left／right／up／down／appropriate | 受力偏大的接触位于夹取中心的哪一侧 |
| rotate | yaw_pos／yaw_neg／pitch_pos／pitch_neg／roll_pos／roll_neg／none | 围绕场景固定轴的姿态偏差 |

area为夹取位置判断提供接触证据；Fx／Fy同向分量为shear提供证据；Fz为整体夹取力度与受力失衡提供证据；Fx／Fy异向分量及rotation共同支持旋转相关诊断。这些是信息依据，不是单字段到单故障的确定性映射，最终解释仍依赖视觉和任务条件。

grasp_balance描述受力偏大的接触所在侧，不描述力矢量指向哪一侧。grasp_force与grasp_balance可以同时异常。rotate表达相对目标姿态的偏差，其正负不是下一步恢复的方向标签。示例：

```text
failure[grasp_position=too left; shear=none; grasp_force=appropriate; grasp_balance=appropriate; rotate=none]
```

failure经共享PaliGemma自回归生成，采用字段与类别限定的文本语法。它不是五个独立分类头的拼接；分类头仅用于Touch与两个状态判断任务。

### 1.9 基于全局故障历史的恢复规划

**记忆内容与容量。**用户已确认：仅保留当前failure之前最近4条全局failure，不保留历史plan，不保留历史Touch。令

\[
\mathcal M_k=(f_{\max(0,k-4)},\ldots,f_{k-1}),
\qquad
\mathcal S_k=(\mathcal M_k,f_k).
\]

\(\mathcal S_k\)按由旧到新的顺序输入，末项为当前failure。首次规划时没有过去failure，仅输入当前项；历史饱和后，输入最多包含4条过去failure及1条当前failure，共5条。容量不因保留首次记录而额外增长。

**规划条件。**规划器使用通用观测及上述全局故障序列：

\[
p_k\sim p_\theta\bigl(p\mid C_t,\mathcal S_k\bigr).
\]

plan不直接输入Touch，不输入H100，也不输入已执行plan或initial plan占位项。局部触觉通过当前failure和历史failure间接影响恢复决策。

统一到固定场景参考后，模型可以比较多次恢复后的故障方向是否持续、是否发生反向等变化，据此选择下一步恢复的方向与程度。由于输入不包含历史plan及实际位移，方法中的“利用恢复历史”具体指利用历史全局故障反馈，不将其扩展表述为显式读取每次已执行动作量或辨识精确物理响应。

**计划字段。**论文版plan包含四个动作字段：

| 字段 | 方向 | 程度 | 含义 |
| --- | --- | --- | --- |
| horizontal | front／back／left／right／none | slightly／moderately／significantly | 场景参考下的水平恢复移动 |
| vertical | up／down／none | slightly／moderately／significantly | 竖直恢复移动 |
| rotate | clockwise／counterclockwise／none | slightly／moderately／significantly | 围绕末端腕部旋转关节轴的恢复运动 |
| gripper_distance | open／close／none | slightly／moderately／significantly | 增大或减小夹爪开口距离 |

活动字段采用“方向 程度”的形式，不活动字段只写none。例如：

```text
plan[horizontal=right slightly; vertical=none; rotate=none; gripper_distance=none]
```

水平与垂直方向使用failure相同的场景参考；rotate使用腕部关节轴的固定观察方向。gripper_distance改变开口距离，不直接设定夹取力。不同字段的程度对应移动距离、腕部转角及开口距离变化量，均采用人工定性标注，不预设固定数值或范围；其示教含义在第四节展开。

故障诊断和恢复规划共用同一个语言主干与词表解码器，分别使用自己的条件前缀和输出语法。文本语法约束限定字段及可接续标签，使输出符合统一接口；结构化格式在此承担接口组织作用，不作为独立创新点。

#### 1.9.1 共享语言输出与词表投影

failure和plan均复用PaliGemma的Transformer、词嵌入及词表输出投影，不各自新增解码网络或字段分类层。相对原有主干，任务适配使用共享LoRA；工程上补充结构化回答的训练、解码和语法接口。语言生成路径不经过Action Expert。

词表投影采用输入输出权重共享。设原词嵌入矩阵为\(E\in\mathbb R^{V\times2048}\)，某预测位置的特征为\(h\)，则：

\[
z=hE^\top.
\]

这等价于使用共享权重、不带bias的2048→词表维度的线性层。当前实现仅提取合法输出涉及的原词表token集合\(\mathcal V_c\)，计算\(z_c=hE[\mathcal V_c]^\top\)，减少无关词表logits的计算。compact词表是原矩阵的行子集，不是另行初始化的小词表输出头。Stage B冻结该投影权重，通过共享LoRA改变\(h\)。

#### 1.9.2 前缀与回答的attention mask

文本训练序列由观测前缀\(P\)和回答后缀\(Y\)组成。前缀包含图像、任务所需文本、状态及Answer提示，内部双向可见；回答采用因果注意力。其分块掩码为：

\[
A=
\begin{bmatrix}
\mathbf 1_{P\times P} & \mathbf 0_{P\times Y}\\
\mathbf 1_{Y\times P} & \operatorname{tril}(\mathbf 1_{Y\times Y})
\end{bmatrix}.
\]

行表示读取者、列表示被读取位置，并额外屏蔽无效相机和padding。前缀不能读取回答；回答位置可以读取完整前缀及截至自身的回答token，而该位置的输出用于预测下一token。因此，答案对角线可见不造成未来答案泄漏。该Prefix-LM模式继承[PaliGemma原有设计](https://arxiv.org/html/2407.07726v2#S3.SS1)，不是新增的注意力模块。need／end只编码观测前缀，不附加目标回答，再按1.7节池化。

#### 1.9.3 token前缀树与生成约束

failure与plan分别将合法完整输出编码为SentencePiece token序列，含EOS，并建立token前缀树。生成时，根据已生成序列查询允许的下一token集合，在其中选择logit最高项，再继续解码，直至得到合法完整文本。

约束分为两层：compact集合限定本任务涉及的词表行；前缀树进一步限定当前位置允许的token。约束作用于token而非按空格切分的单词，保证字段、类别和文本顺序合法，但不保证故障或计划语义正确。新字段与类别必须同步进入输出语法，否则模型无法生成这些标签。

### 1.10 正常执行与恢复共享的动作专家

同一个动作专家在两个阶段生成连续关节动作，区别在于语言条件：

\[
A_t\sim
\begin{cases}
\pi_\theta(A\mid C_t,\texttt{execution}),&s_t=\texttt{execution},\\
\pi_\theta(A\mid C_t,\texttt{adjustment},p_k),&s_t=\texttt{adjustment}.
\end{cases}
\]

正常执行不输入plan，恢复动作以当前plan为条件。动作分支不直接输入Touch、H100或完整failure历史；触觉主要经故障理解、恢复计划以及阶段切换影响动作。

动作专家采用条件流匹配。它将带噪动作块与流时间编码作为动作侧输入，在各层注意力中读取视觉语言前缀，预测动作流场。π0.5通过时间MLP和AdaRMSNorm注入流时间条件。采样从噪声动作块出发，逐步积分得到连续动作，不将关节控制离散为语言token生成。

当前输出动作块为30步，内部适配π0.5的32维动作空间；机器人取有效的7维控制量，即六个臂关节及夹爪开合。训练使用相对当前qpos的动作目标，推理输出经过反归一化并恢复为绝对关节目标：

\[
a_{t+h}=q_t+\Delta a_{t+h},\qquad h=0,\ldots,29.
\]

这里每个预测目标均相对块起点的当前状态，不按相邻步增量累计。plan文本是学习到的动作条件，不通过硬编码解析器直接换算成固定关节增量。恢复幅度由动作生成与恢复结束判断共同落实。

### 1.11 各任务输入与输出的统一对照

所有任务均读取通用输入\(C_t\)，额外条件按下表选择。该表描述论文主方法；规划消融中的qpos屏蔽与运动历史长度对照均属于实验设置，不改变主方法定义。

| 任务 | 当前Touch | qpos运动历史H100 | 当前评估plan | 当前全局failure | 历史全局failure | 其他阶段条件 | 输出 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 正常动作 | 无 | 无 | 无 | 无 | 无 | execution | 30步动作块 |
| 恢复动作 | 无 | 无 | 有 | 无 | 无 | adjustment | 30步动作块 |
| need_recovery | 有 | 有 | 有，无计划时none | 无 | 无 | 评估上下文 | 二分类概率 |
| failure | 有 | 有 | 有，无计划时none | 无 | 无 | 与need共享的评估上下文 | 全局failure文本 |
| adjustment_end | 有 | 有 | 有 | 无 | 无 | 恢复评估上下文 | 二分类概率 |
| plan | 无 | 无 | 无 | 有 | 最近最多4条 | reasoning | 四字段plan文本 |

五项任务对应动作生成、need、failure、plan与end，其中动作生成按阶段分为两种条件。共享不要求不同任务使用完全相同的输入，也不要求所有输出在一次前向计算中同时生成。

need与failure使用相同前缀时，可以共享一次前缀编码及KV缓存：池化特征供need分类，缓存供后续failure解码。plan使用包含全局故障序列的新前缀，单独完成一次语言推理；end使用其分类头，正常与恢复动作使用共享动作专家。动作执行与后台判断的异步部署，以及缓存复用的具体时序，在第三节详细说明。

### 1.12 本节确认状态、实现差异与代码依据

**已由既有讨论及本轮确认的论文设定：**五字段Touch、五字段failure、四字段plan；Touch以左指传感器局部坐标为参考；plan不读取Touch或历史plan；历史只保留最近4条过去全局failure，另加当前failure；动作、诊断、规划及两个状态头共享同一套VLA主干。

**本节补充确认：**

| 编号 | 内容 | 当前写法 |
| --- | --- | --- |
| C1 | 主摄参考的场景坐标轴 | 已确认：+X前且靠近主摄、+Y右、+Z上 |
| C2 | 描述器最终结构参数 | 已确认：W30、128维CNN、192维TCN、dilation 1/2/4/8、dropout 0.1 |

**论文接口与原型实现的差异。**当前共享网络结构已能由代码核对，但保存的部署协议仍包含旧版接口。论文定稿及后续实验需要明确使用哪个接口版本，不将以下差异隐藏在共同网络名称下。

| 模块 | 已核对代码／配置 | 本节采用的论文接口 |
| --- | --- | --- |
| Touch字段 | V4六字段，含独立Fz_bias，Fx／Fy类别较少 | 五字段，Fx／Fy加入双指异向类别，Fz合并双指强弱差异 |
| Touch分类头 | 标签映射仅接受既有V3／V4类别集合 | 五个头，类别数4／5／5／4／3；新类别映射需对应实现 |
| failure | 部署版本存在任务相关的旧字段与类别 | 统一grasp_position／shear／grasp_force／grasp_balance／rotate |
| plan | 部分部署语法为horizontal／vertical／rotate三字段 | 增加gripper_distance，共四字段 |
| 恢复记忆 | 部分运行代码仍维护plan–failure配对 | 最近4条历史failure，另加当前failure；v13_4已有failure-only规划实验接口 |

**工程核对项：首token预测的索引（不作为论文方法表述）。**当前`assessment_prefill()`以`sum(prefix_mask)-1`选取前缀末位置；该写法只在有效token连续、无效位置全部在末尾时成立。现有输入包含位于语言token之前的无效相机槽，应改为取`prefix_mask`最后一个True的实际下标。用于生成的位置编号仍按有效token数计算，不能与数组物理下标混淆。本轮只记录该问题，尚未修改或验证运行代码；第三节描述的是正确索引下的预期数据流。

当前章节的网络结构依据如下，训练参数更新、数据采样和异步状态机的细节留待后续各节核对：

| 内容 | 主要代码或配置 |
| --- | --- |
| 触觉逐帧CNN、TCN和字段头 | [captioner/model.py](../src/tactile_vla/captioner/model.py) |
| 双指网格组织与W30缓冲 | [runtime/tactile_buffer.py](../src/tactile_vla/runtime/tactile_buffer.py) |
| 标准化与字段预测文本化 | [captioner/predictor.py](../src/tactile_vla/captioner/predictor.py) |
| V4旧字段定义 | [common/labels_v4.py](../src/tactile_vla/common/labels_v4.py) |
| π0.5视觉、语言及动作专家 | [pi0.py](../openpi/src/openpi/models/pi0.py)、[gemma.py](../openpi/src/openpi/models/gemma.py) |
| 状态离散化与语言前缀 | [tokenizer.py](../openpi/src/openpi/models/tokenizer.py) |
| 两个独立判断头与共享主干 | [v7_7_multitask_model.py](../src/tactile_vla/vla/v7_7_multitask_model.py)、[stage_b_v3_model.py](../src/tactile_vla/vla/stage_b_v3_model.py) |
| 掩码池化、前缀缓存与语言解码 | [stage_b_v3_jax.py](../src/tactile_vla/vla/stage_b_v3_jax.py) |
| 共享词表投影与输入输出权重绑定 | [gemma.py](../openpi/src/openpi/models/gemma.py)中的Embedder及decode |
| 合法输出前缀树与约束生成 | [structured_text.py](../src/tactile_vla/vla/structured_text.py)、[structured_generation.py](../src/tactile_vla/vla/structured_generation.py) |
| H100采样和文本表示 | [v7_7_phase_prompt.py](../src/tactile_vla/vla/v7_7_phase_prompt.py)、[v5_3_phase_change.py](../src/tactile_vla/vla/v5_3_phase_change.py) |
| 图像变换、状态与动作适配 | [openpi_bridge.py](../src/tactile_vla/vla/openpi_bridge.py) |
| 无历史plan的failure序列接口 | [blackbox_ablation_v13_4_data.py](../src/tactile_vla/vla/blackbox_ablation_v13_4_data.py) |
| 已确定的论文版字段 | [论文要点总结.md](论文要点总结.md)4.1节；本轮确认的历史容量优先于旧文档中的配对记忆描述 |

保存的Bottle V11.2及Black-box V7.7.2配置均核实为H30、action_dim=32、need_hidden_dim=512、need_dropout=0.1、use_state_history=false；语言与动作专家分别为gemma_2b_lora与gemma_300m_lora。上述代码检查未重新执行训练或真机推理。

## 二、训练方式

### 2.1 训练组织与参数共享范围

训练由触觉描述器的独立监督训练，以及VLA的两阶段训练组成。首先学习触觉信号到元描述的映射；随后通过动作示教学会正常执行与恢复动作；最后在保留动作能力的基础上，训练恢复判断、故障诊断和历史恢复规划。

**已确认的操作任务组织。**Black-box、Book、Sponge和Bottle分别训练四套VLA模型。四个模型使用相同的总体结构与训练流程，各自在对应操作任务的数据上训练。本文的“五任务联合训练”指同一操作任务模型内部的action、adjustment_end、need_recovery、failure与plan，不表示将四个操作任务合并训练成一个模型。触觉描述器作为独立模块训练，不参与VLA的联合反向传播。

| 阶段 | 初始化与输入 | 监督信号 | 训练产物 |
| --- | --- | --- | --- |
| 触觉描述器 | 从头初始化CNN、TCN及五个字段头；输入双指触觉窗口 | 有效字段的离散标签 | 固定的触觉描述器 |
| Stage A：动作学习 | π0.5预训练动作模型或已有动作checkpoint；输入图像、qpos、任务及阶段条件 | 正常执行与恢复示教的H30动作块 | 具备任务执行和按plan恢复能力的动作模型 |
| Stage B：五任务训练 | 从该任务的Stage A初始化主干与动作专家，增加两个判断头 | 动作、两个二分类标签、failure及plan文本 | 共享VLA与恢复判断、诊断、规划能力 |

Stage B使用离线生成的Touch文本。描述器参数在这一阶段固定，VLA的梯度不通过离散标签或文本接口回传到描述器；两阶段的优化器和参数更新分别管理。

### 2.2 触觉描述器的独立训练

#### 2.2.1 从头训练与有效字段监督

用户已确认，论文版五字段描述器从头训练CNN、TCN及全部五个分类头，采用第一节确定的网络结构。每个窗口输出五组分类logits，监督仅施加在有明确标注的字段上。

对窗口\(i\)和字段\(j\)，令\(m_i^{(j)}\in\{0,1\}\)表示是否具有有效标签。未标注字段不以none、near_zero或其他默认类别代替，也不作为负例加入损失。现有masked数据格式每个窗口只监督一个明确字段；监督掩码使共享编码器可以从不同字段的数据共同学习接触表征。论文版数据的采集与标注组织在第四节确认。

设\(p_{i,c}^{(j)}\)为字段\(j\)预测类别\(c\)的概率，\(K_j\)为类别数，则描述器损失为

\[
\mathcal L_{\mathrm{touch}}=
-\frac{1}{N_{\mathrm{valid}}}
\sum_{i,j}m_i^{(j)}
\sum_{c=1}^{K_j}w_c^{(j)}\tilde y_{i,c}^{(j)}
\log p_{i,c}^{(j)},
\qquad
N_{\mathrm{valid}}=\sum_{i,j}m_i^{(j)}.
\]

其中\(\tilde y\)为标签平滑后的目标分布。代码将所有有效字段目标的损失拼接后取均值，因此每个有效标注贡献一次，不先对每个字段各求均值再对字段等权平均。分类头没有有效标签时不产生该字段的分类损失。

#### 2.2.2 类别分布与优化

现有训练代码使用训练集类别计数计算权重。默认权重与类别频次的平方根倒数成比例，再归一化为均值1：

\[
w_c^{(j)}\propto\frac{1}{\sqrt{n_c^{(j)}}},
\qquad
\frac{1}{K_j}\sum_c w_c^{(j)}=1.
\]

该权重仅根据具有该字段有效监督的训练样本计算。训练可使用预构建的平衡索引；验证和测试使用各自原始索引。平衡索引的生成及数据划分属于第四节的数据处理内容。

编码器与字段头统一使用AdamW更新。代码默认先进行总更新步数5%的学习率warmup，再采用余弦衰减；每步对全部可训练参数做全局梯度裁剪。验证时关闭dropout，不使用训练时的类别加权与标签平滑。

**已确认T3：描述器优化设置。**用户已确认沿用下列代码默认值，作为论文版五字段描述器的训练方案；该确认不表示新五字段模型已经完成训练。

| 项目 | 已确认设置 |
| --- | --- |
| batch size | 128 |
| 最大训练轮数 | 50 epochs |
| 优化器 | AdamW，默认betas为0.9／0.999 |
| 初始学习率 | 3e-4 |
| weight decay | 1e-4 |
| 学习率调度 | 前5%更新warmup，随后余弦衰减 |
| label smoothing | 0.03 |
| 类别加权 | 频次倒平方根，按字段归一化 |
| 梯度裁剪阈值 | 全局范数1 |
| 随机种子 | 42 |
| 选点与早停 | 按验证集有效字段Macro-F1的均值选best；连续10轮无提升早停 |

训练结束后固定描述器，用于机器人轨迹的离线Touch标注以及在线Touch预测。离线与在线使用相同信号通道、标准化统计量、窗口长度和文本接口。描述器的识别性能及开销在实验部分单独评价。

### 2.3 Stage A：学习正常执行与恢复动作

Stage A使用机器人示教中的正常执行与恢复片段进行动作监督。正常执行输入任务与execution条件，恢复输入任务、adjustment条件及示教plan。两种阶段共用动作模型和连续动作空间，通过阶段与plan文本区分行为。

每个动作训练样本包含当前两路图像、当前qpos、任务条件及未来30步示教动作。该阶段的动作输入不包含Touch、H100或完整故障历史。恢复示教中的plan作为动作条件，使动作专家学习根据恢复语义生成关节运动。

动作训练使用相对块起点当前qpos的delta目标。设真实动作块为\(A\in\mathbb R^{30\times32}\)，其中有效机器人动作占7维，其余维度用于适配预训练动作空间。采样高斯噪声\(\epsilon\)与流时间\(\tau\)，构造

\[
A_\tau=(1-\tau)A+\tau\epsilon,
\qquad
u_\tau=\epsilon-A.
\]

动作专家在当前视觉语言条件\(C^{\mathrm{act}}\)下预测流场，采用均方误差：

\[
\mathcal L_{\mathrm{act}}=
\mathbb E_{A,\epsilon,\tau}
\left[
\frac{1}{30\cdot32}
\left\|v_\theta(A_\tau,\tau,C^{\mathrm{act}})-u_\tau\right\|_F^2
\right].
\]

当前实现中\(\tau=0.999\,\mathrm{Beta}(1.5,1)+0.001\)。流时间为1对应噪声，为0对应目标动作；该约定与第一节动作采样的积分方向一致。训练损失在内部32维动作及30个时间位置上取均值，部署时提取7维有效控制量。

动作块的起止位置需满足相应数据版本的阶段与故障边界条件，避免将跨越阶段或故障之后的错误动作当作期望执行策略。具体筛选在第四节说明，不将所有任务的数据处理规则混为一套。

### 2.4 两阶段的参数更新范围

#### 2.4.1 Stage A的LoRA与其他可训练参数

语言主干采用rank=16、alpha=16的LoRA配置，动作专家采用rank=32、alpha=32的LoRA配置；LoRA注入注意力与前馈模块。Stage A从预训练权重或已训练动作模型初始化，在正常和恢复动作数据上适配。

需要按实际冻结过滤器描述更新范围：当前train_lora_only设置冻结LLM内部的非LoRA参数，但不冻结整个模型的其余部分。视觉编码器、动作输入／输出投影以及时间MLP等LLM之外的参数仍属于Stage A可训练集合。因此，本阶段可表述为“基于LoRA的任务适配，并更新视觉与动作适配参数”，不能简写为全模型仅训练LoRA。

#### 2.4.2 Stage B固定动作专家，训练共享语义表征

Stage B加载对应Stage A的完整权重，增加独立初始化的need与end分类头。该阶段仅训练共享PaliGemma的LoRA及两个分类头，冻结视觉编码器、语言主干非LoRA参数、动作专家的全部参数及其他动作适配参数。

| 参数模块 | 触觉描述器训练 | Stage A | Stage B |
| --- | --- | --- | --- |
| 触觉CNN／TCN／五字段头 | 更新 | 固定，动作任务不调用 | 固定，仅提供离线文本 |
| SigLIP视觉编码器 | 不参与 | 更新 | 冻结 |
| PaliGemma非LoRA参数 | 不参与 | 冻结 | 冻结 |
| PaliGemma LoRA | 不参与 | 更新 | 更新 |
| 动作专家非LoRA参数 | 不参与 | 冻结 | 冻结 |
| 动作专家LoRA | 不参与 | 更新 | 冻结 |
| 动作投影与时间MLP | 不参与 | 更新 | 冻结 |
| need与end分类头 | 不参与 | 尚未使用 | 更新 |

动作专家冻结后仍保留action任务。动作损失可以通过固定动作专家的可微计算回传到共享前缀的PaliGemma LoRA，从而约束语义训练对原有动作表征的影响。冻结权重不等于切断该模块的反向传播。

### 2.5 Stage B：五任务交替监督

同一操作任务的Stage B包含五个训练流：

| 训练任务 | 样本监督 | 损失 | 主要梯度路径 |
| --- | --- | --- | --- |
| action | 正常执行或恢复的H30动作块 | 流匹配MSE | 经固定动作专家到共享PaliGemma LoRA |
| adjustment_end | 当前恢复阶段是否应结束 | 二分类交叉熵 | 共享LoRA与end头 |
| need_recovery | 当前执行是否需要恢复 | 二分类交叉熵 | 共享LoRA与need头 |
| failure | 当前全局故障文本 | 受语法约束的自回归token交叉熵 | 共享PaliGemma LoRA |
| plan | 根据当前及历史全局failure生成的恢复计划 | 受语法约束的自回归token交叉熵 | 共享PaliGemma LoRA |

训练按固定顺序循环：

```text
action → adjustment_end → need_recovery → failure → plan → 下一轮
```

每次更新只从当前任务的数据流取一个batch，计算该任务损失并进行一次优化器更新。五项任务不是对每条样本同时计算五种损失，也不是先分别训练五套模型后再拼接。固定周期使各任务获得相同的更新次数；总训练步数按完整五任务周期设置。

在任务级均匀采样的意义下，优化目标可写为

\[
\mathcal L_{\mathrm B}=
\frac{1}{5}\sum_{r\in\mathcal R}
\lambda_r\,\mathbb E_{D_r}[\mathcal L_r],
\qquad
\mathcal R=\{\mathrm{act},\mathrm{end},\mathrm{need},\mathrm{failure},\mathrm{plan}\}.
\]

四个主实验版本的任务损失权重均为1。上式用于概括训练目标；实际实现按上述顺序分步优化，共享一套训练状态和优化器。

### 2.6 分类与语言生成的训练目标

#### 2.6.1 两个状态判断头

need与end均以布尔标签监督。对任一分类任务，采用两类softmax交叉熵：

\[
\mathcal L_{\mathrm{cls}}=-\frac1B
\sum_{i=1}^{B}\log P_\theta(y_i\mid X_i),
\qquad y_i\in\{0,1\}.
\]

两项损失分别计算，共享主干但更新各自的分类头。当前训练函数使用普通二分类交叉熵，类别分布由数据构造与采样策略控制，不额外假设使用了focal loss或类别加权分类损失。故障、恢复开始与恢复结束边界如何转换为布尔标签，在第四节逐任务说明。

#### 2.6.2 failure与plan的自回归监督

两类文本均采用teacher forcing。输入前缀组织当前任务所需的图像、状态与上下文，目标答案作为自回归后缀。目标位置只能读取前缀及更早的答案token，不能读取未来答案。只对有效答案token计算损失，前缀、填充位置不参与语言目标监督。

设答案为\(Y=(y_1,\ldots,y_L)\)，输出语法在位置\(u\)给出合法token集合\(\mathcal A_u(Y_{<u})\)。在该集合上归一化概率：

\[
P_\theta(y_u\mid X,Y_{<u})=
\frac{\exp z_{u,y_u}}
{\sum_{v\in\mathcal A_u(Y_{<u})}\exp z_{u,v}},
\]

\[
\mathcal L_{\mathrm{text}}=-
\frac{1}{N_{\mathrm{ans}}}
\sum_{i,u}a_{i,u}
\log P_\theta(y_{i,u}\mid X_i,Y_{i,<u}),
\]

其中\(a_{i,u}\)为答案有效掩码。failure和plan使用同一个语言解码器及词表，各自的字段语法限定合法输出。实现中的compact token集合是已有SentencePiece词表中允许参与该任务的token子集，不为每个字段新增独立语言模型。

训练时一次前向并行预测所有回答位置：最后一个前缀位置预测\(y_1\)，\(y_1\)所在位置预测\(y_2\)，依次移位对齐。语法约束参与训练，非法候选logits被屏蔽后再计算交叉熵；若某位置只有一个合法候选，其直接选择损失为0，分支选择监督主要来自方向、程度等存在多个合法候选的位置。

failure监督目标为全局故障标注。plan监督目标为对应恢复计划，输入包含当前及历史全局failure，但不包含目标plan或历史已执行plan。训练前缀中的failure来自离线标注与历史样本构造，在线部署时则由模型诊断产生；训练历史的具体来源与构造规则在第四节确认。

两任务按2.5节交替更新共享LoRA，训练时不要求先生成failure再将离散输出接给plan反向传播。推理时才串联诊断与规划；验证时移除正确回答，从无答案前缀自回归生成后评价，见2.10节。

### 2.7 动作回放与样本平衡

Stage B保留正常执行与恢复示教的action训练流，使共享LoRA在学习诊断与规划时继续接受动作约束。Book V9.5、Sponge V10.1及Bottle V11.2使用逐batch阶段平衡：batch size为8时，每个action batch包含4条execution与4条adjustment样本。

两个阶段的数据池分别打乱，在各自一轮遍历内不重复；较短的数据池耗尽后独立重新打乱。每个batch内保持两阶段数量一致，避免数据量较大的正常执行片段长期主导动作回放。

Black-box V7.7.2沿用其原始action loader，在当前代码中未启用上述逐batch 4＋4采样器；不将后三个任务的阶段平衡实现套写到所有主实验。各流的分类正负比例、故障边界及文本样本选点以对应训练清单为准，第四节进一步记录。

### 2.8 恢复程度的反事实监督

在具有slightly与moderately程度标注的Black-box恢复数据上，为end训练补充反事实样本：保持图像、Touch、qpos与运动历史相同，仅替换当前plan的程度词，并根据替换后的目标程度重新标注是否应该结束。由此，同一物理观测可以在不同计划条件下获得不同结束标签，监督结束判断利用当前plan的程度。

例如，将一段moderately示教的plan改为slightly后，轨迹较早位置可以被标为应结束；将slightly示教改为moderately时，原本足够的小幅运动可以被标为仍需继续。这些反事实用于结束判断，不生成并不存在的后续动作轨迹，也不替代Stage A的真实动作示教。

当前Black-box反事实脚本以示教恢复开始帧\(A\)与停止帧\(S\)的中点\(\lfloor(A+S)/2\rfloor\)构造moderately→slightly的代理结束位置；slightly→moderately在已有短轨迹区间内标为未结束。帧中点不等于真实位移的一半，正文不将该规则改写为精确的距离标定。完整的样本区间、事实标签与反事实配对规则在第四节明确。

### 2.9 优化器与各任务实际训练配置

VLA两阶段均使用AdamW，betas为0.9／0.95、eps为1e-8、梯度全局范数裁剪阈值为1，随机种子为42。Stage A与Stage B分别建立优化器；Stage B加载Stage A模型参数，不将Stage A优化器状态直接视为五任务训练的延续。

**Stage A配置。**以下来自用户已确认版本对应的保存配置，保留任务之间的实际差异：

| 操作任务 | 动作训练版本 | 初始化 | 更新步数 | global batch | 学习率调度 |
| --- | --- | --- | --- | --- | --- |
| Black-box | V7.4.2 | 原V7.4动作模型第15000步 | 新环境继续训练10000步 | 8 | 1e-5余弦衰减至1e-6，跨度10000步 |
| Book | Stage A V9.5 | pi05_base | 15000 | 8 | 前7000步5e-5，此后5e-7 |
| Sponge | Stage A V10.1 | pi05_base | 15000 | 8 | 前7000步5e-5，此后5e-7 |
| Bottle | Stage A V11.2 | pi05_base | 7500 | 16 | 前3500步5e-5，此后5e-7 |

四组Stage A的weight decay均为1e-10。Black-box采用已有动作模型的新环境适配，其10000步不应被表述为从原始pi05_base开始的全部动作训练量。其余三组保存配置直接从pi05_base初始化。Book、Sponge和Bottle采用分段恒定学习率，不统一写为余弦退火。

**Stage B配置。**四组均使用学习率1e-4、weight decay 1e-4、global batch 8，每1000步验证并保存。表中的每任务更新数按五任务周期计算，best步数是产物中的验证选点记录。

| 操作任务 | 完整模型版本 | Stage A初始化步数 | Stage B总更新 | 每任务更新 | FSDP设备数 | 验证best步数 |
| --- | --- | --- | --- | --- | --- | --- |
| Black-box | V7.7.2 | 10000 | 5000 | 1000 | 2 | 4000 |
| Book | V9.5 | 15000 | 2000 | 400 | 2 | 2000 |
| Sponge | V10.1 | 15000 | 2000 | 400 | 2 | 1000 |
| Bottle | V11.2 | 7500 | 2000 | 400 | 1 | 2000 |

上述总步数已与各运行的最后训练记录核对，best步数与各自best/metrics.json核对。FSDP表示训练并行配置，不改变“每步仅更新一个任务”的组织方式。这里记录的是训练产物，正式真机运行所选checkpoint在第三节单独核对，不仅凭best文件或服务端默认路径推断。

### 2.10 验证选点与动作能力保持

Stage B的选点同时考虑语义任务与动作保持。验证分数由need Macro-F1、end Macro-F1、failure整条文本正确率与plan整条文本正确率的算术平均得到：

\[
S_{\mathrm{val}}=
\frac14\left(
F1_{\mathrm{need}}+F1_{\mathrm{end}}+
EM_{\mathrm{failure}}+EM_{\mathrm{plan}}\right).
\]

语言评价从无答案前缀生成完整文本，不将训练时teacher forcing得到的token正确率直接当作整条生成正确率。模型选择使用验证集，测试集不参与训练过程中的选点。

同时，以Stage A初始化模型在相同验证数据上的动作损失为基准，计算

\[
\rho_{\mathrm{act}}=
\frac{\mathcal L_{\mathrm{act,val}}^{\mathrm B}}
{\mathcal L_{\mathrm{act,val}}^{\mathrm A}}-1.
\]

当前配置要求动作验证损失的相对增加不超过10%，并在满足该条件的候选模型中选择验证语义分数更高的checkpoint。使用阶段平衡动作评价的版本还分别检查execution与adjustment，确保总损失未上升时某一阶段也未明显退化。验证帧和随机噪声设置与Stage A基准保持一致。

这里的failure与plan正确率用于离线模型选择，与任务成功率实验最终选择报告哪些过程指标分别处理；不因主实验不报告failure准确率而省略该训练验证依据。

### 2.11 无触觉对照与专项消融的训练关系

**无触觉对照。**用户已确认，完整方法与无触觉版本复用同一个Stage A动作模型；无触觉组在移除Touch后独立进行Stage B五任务训练。图像、当前qpos、运动历史、failure历史及恢复框架继续保留。由于action分支本身不直接输入Touch，Stage A无需因这个消融另行训练。

这一对照比较在有无触觉条件下学习到的判断、诊断与恢复能力，不采用“只在测试时遮挡完整模型Touch”的方式。两组具体Stage B配置、训练样本与选点在实验设置中逐项对应。

**历史表示消融。**已经确认的v13_4采用独立的plan-only训练协议：从同一Stage A初始化，仅训练plan任务的共享LoRA，不训练两个判断头或动作专家；训练、验证与测试统一屏蔽qpos，比较无反馈、局部Touch序列、标注全局failure序列。其400步预算与4个测试episode属于专项消融，不代替主方法的完整五任务训练，也不将oracle failure作为在线诊断的默认来源。

**运动历史与反事实对照。**运动历史实验改变H100窗口长度，恢复程度实验改变end的反事实监督。它们的具体训练版本、控制条件及数据复用关系在实验汇总中记录；不将旧参考模型与新增对照的训练差异隐去。

### 2.12 本节确认状态、版本差异与依据

**本轮已确认：**

- 四个操作任务分别训练VLA；五任务联合仅指每套模型内部的五项功能。
- 五字段触觉描述器从头训练全部CNN、TCN和分类头。
- 四项完整方法采用Black-box V7.7.2、Book V9.5、Sponge V10.1、Bottle V11.2的训练配置。
- 无触觉组复用对应Stage A，在去除Touch后独立训练Stage B。

- 触觉描述器沿用2.2节所列优化设置，包括batch 128、最多50 epochs、AdamW学习率3e-4及验证早停。

本节训练方案已确认。

**与第一节新接口的对应。**现有四个主实验产物仍包含部分旧Touch、failure、plan及配对记忆协议。上文保存配置表描述这些已完成运行的优化过程，不能据此宣称它们已经按论文版五字段Touch、统一failure、四字段plan及failure-only记忆全部重训。描述器的五字段从头训练方式已由用户确定，新标签映射、监督文本与记忆构造在第四节按论文接口确认，并保留与已有运行记录的对应。

| 内容 | 主要依据 |
| --- | --- |
| 描述器掩码损失、类别权重、优化器与早停 | [captioner/training.py](../src/tactile_vla/captioner/training.py) |
| 有效字段与训练数据索引 | [data/tactile_captioner_dataset.py](../src/tactile_vla/data/tactile_captioner_dataset.py) |
| Stage A优化器、初始化与更新 | [train_vla_stage_a_openpi.py](../scripts/train_vla_stage_a_openpi.py) |
| Stage A冻结规则 | [pi0_config.py](../openpi/src/openpi/models/pi0_config.py) |
| 流匹配动作损失 | [pi0.py](../openpi/src/openpi/models/pi0.py) |
| Stage B参数过滤器 | [v7_7_multitask_model.py](../src/tactile_vla/vla/v7_7_multitask_model.py) |
| 五任务轮换、训练损失、验证与best选择 | [train_vla_multitask_v7_7.py](../scripts/train_vla_multitask_v7_7.py)、[train_vla_stage_b_v3.py](../scripts/train_vla_stage_b_v3.py) |
| 语法限制的token交叉熵 | [stage_b_v3_jax.py](../src/tactile_vla/vla/stage_b_v3_jax.py) |
| 阶段平衡动作回放与验证 | [book_v9_4_5_action_replay.py](../src/tactile_vla/vla/book_v9_4_5_action_replay.py) |
| Black-box恢复程度反事实构造 | [v7_6_adjustment_end_data.py](../src/tactile_vla/vla/v7_6_adjustment_end_data.py) |
| 主实验配置 | 各任务对应运行目录的config.json、metrics.jsonl及best/metrics.json |

四项Stage B配置目录分别为：

```text
/data1/qxh/tac_vla_new/tac_data/demon_data/black_box/outputs/multitask_v7_7_2/pi05_rotation_v7_7_2_five_task_h100_no_history/
/data1/qxh/tac_vla_new/tac_data/demon_data/book/outputs/multitask_v9_5/pi05_book_v9_5_five_task_h100_no_history/
/data1/qxh/tac_vla_new/tac_data/demon_data/sponge/outputs/multitask_v10_1/pi05_sponge_v10_1_five_task_h100_no_history/
/data1/qxh/tac_vla_new/tac_data/demon_data/bottle/outputs/multitask_v11_2/pi05_bottle_v11_2_five_task_h100_no_history/
```

## 三、推理方式

### 3.1 在线状态与推理接口

在线系统维护execution和adjustment两个行为阶段。execution调用原任务动作策略，并运行need判断是否需要恢复；adjustment按照当前plan生成恢复动作，并运行end判断何时结束当前恢复。暂停与保持由控制程序管理，是阶段转换期间的控制状态，不增加第三套学习策略。

每次任务开始时初始化：

\[
s=\mathrm{execution},\qquad
\mathcal M=\varnothing,\qquad
p^{\mathrm{ctx}}=\mathrm{none},\qquad
r=0,
\]

其中\(\mathcal M\)为过去failure记忆，\(p^{\mathrm{ctx}}\)为评估上下文中的最近恢复计划，\(r\)为已完成的adjustment阶段数。控制器另维护当前动作块、阶段版本号、动作版本号及在途评估请求标识。

| 当前阶段或过程 | 模型调用 | 主要结果 | 控制作用 |
| --- | --- | --- | --- |
| execution动作生成 | action，条件为原任务与execution | H30动作块 | 推进原任务 |
| execution后台评估 | need；触发后继续failure解码 | need概率与当前failure | 中断动作并启动恢复规划 |
| 暂停后的规划 | plan，输入当前及历史failure | 当前恢复计划 | 为adjustment提供动作条件 |
| adjustment动作生成 | 共享action，条件为当前plan | H30恢复动作块 | 执行恢复 |
| adjustment后台评估 | end | 恢复结束概率 | 中断恢复并返回execution |

failure与plan由模型生成。无需实验员在每次故障时指定恢复方向或程度。现有部署接口由实验员按任务标准确认成功或终止试验；它与模型自主完成故障恢复阶段切换分别记录。

### 3.2 初始化与在线观测构造

#### 3.2.1 模型与接口初始化

服务端加载对应操作任务的完整Stage B权重，包括共享主干、动作专家与两个判断头，并读取相匹配的归一化统计、输出语法和分类阈值。客户端分别建立动作／规划请求连接与流式评估连接。各版本检查训练配置、模型产物及captioner身份的一致性，避免混用不同标签或归一化协议。

启动时对动作、need／failure、end和plan的推理路径进行预热。Black-box V7.7.2服务还针对逐token增长的缓存长度进行编译预热。预热期间不开始任务运动；正式试验的在线耗时与启动编译时间分别处理。

#### 3.2.2 Touch窗口与当前图像

机器人端持续订阅双指三维网格位置、网格运动和力场。六路网格均更新后组合一个触觉帧，按固定传感器身份和通道顺序加入W30缓冲。触觉描述器在需要状态评估时读取最新完整窗口，生成当前Touch。

正式任务开始前等待真实触觉窗口就绪。代码为未就绪窗口定义了中性描述，但完整触觉部署在进入正常运行前等待W30，不把中性占位文本当作真实接触证据。

动作请求只采集当前图像与qpos，不等待触觉描述器。后台评估单独获取新的front、wrist、qpos和Touch，以减少动作请求对触觉采集与文本生成的等待。

评估采集检查图像、机器人状态与触觉末帧的时间戳均已更新，并将该次图像与qpos复制为固定快照，供本次评估使用。该检查保证观测新鲜性；现有代码未提供所有模态严格硬件同步的证据，不将其写成同一曝光时刻或毫秒级同步采集。

#### 3.2.3 H100在线历史

机器人状态回调持续维护带时间戳的真实qpos历史。以当前评估快照的qpos时间为末端，在标称30 Hz时间网格上读取最近100个历史位置，再取11点离散文本。历史只使用不晚于当前观测的反馈，不使用未来状态。

任务开始前的缺失历史用本次任务首个qpos左填；回合内的execution／adjustment切换不重置该历史。当前实现还检查采样点与最近ROS反馈的时间差，默认最大允许间隔为0.02秒，具体运行配置以启动参数为准。

### 3.3 动作执行与后台评估的异步组织

机器人端的主控制循环统一按30 Hz发布关节目标，并处理状态切换。execution阶段的need与adjustment阶段的end实际评估频率均按3 Hz描述。客户端采用最多两个工作线程的线程池及两条WebSocket连接：一条用于动作／plan，另一条用于阶段评估。Black-box共享主循环直接同步请求动作；Book、Sponge和Bottle将动作请求提交线程池，再等待Future并处理控制事件。两种实现均在新动作块返回后才继续发布，不将线程池大小解释为两套同时运行的模型。

服务端只有一个共享模型worker，串行处理模型计算。need与failure、plan、action及end没有在多份独立模型上同时前向。当前框架的异步主要体现为：机器人正在执行已经获得的动作块时，后台根据更新后的观测完成状态评估；判断触发后，控制器可以在动作块内部停止发布剩余目标。

同一时刻最多保留一个在途状态评估请求。前一评估尚未完成时不继续堆积新的评估请求；完成后再根据当前阶段、时间间隔和新反馈发起下一次判断。3 Hz表示本项目确认的实际运行评估频率，不沿用旧脚本7 Hz的目标请求参数，也不表示每次评估均能严格等间隔完成。实际完成间隔仍受观测更新、captioner、模型排队、计算与通信耗时影响。

动作块之间仍存在等待新动作推理结果的间隙。当前实现没有提前生成多块未来动作、持续填满动作队列的流水线，因此不将异步部署写成完全无等待的动作生成。

#### 3.3.1 请求协议与服务端分流

任务运行中的模型调用由机器人端发起，服务端依据请求的mode、phase及prompt选择计算路径。execution／adjustment是行为阶段，action／need／end／plan是模型任务，两者分别表达。

| 调用目的 | 请求mode | 独立phase字段 | 动作条件或服务端路径 | 返回内容 |
| --- | --- | --- | --- | --- |
| 正常动作 | execution | 无 | prompt为execution，调用共享Action Expert | H30动作数组 |
| 恢复动作 | execution | 无 | prompt为adjustment并附加plan，调用同一Action Expert | H30动作数组 |
| 恢复触发 | phase | execution | need头；触发后续写failure | 判断事件，必要时再返回failure事件 |
| 恢复结束 | phase | adjustment | end头 | 判断事件 |
| 恢复规划 | reasoning | 无 | 共享语言解码器生成plan | 完整plan文本 |

当前mode=execution实际表示动作生成接口，不只表示正常执行。服务端由外层程序显式选择分类头或生成函数，不依赖模型自行猜测请求类型；机器人端维护当前阶段，并在每次请求中提供相应条件。need=true后的failure由服务端在同一次请求内继续生成，不需要客户端再发送独立诊断请求。

动作与plan均为一次请求、一次完整响应，复用`action_policy.infer_with_timeout()`及同一连接，靠mode区分；流程上先取得plan，再请求恢复动作，不在该连接同时混发两种请求。评估连接使用`StreamingPhaseClient.events()`，支持先返回判断、再条件性返回failure。增加连接本身不改变单模型worker串行计算的事实。

#### 3.3.2 机器人端发送逻辑与请求时序

启动时建立两条连接并接收服务端metadata；`get_server_metadata()`读取连接时缓存的信息，不额外发起推理。实际业务发送入口如下：

| 入口 | 触发时机 | 发送前准备与返回后处理 |
| --- | --- | --- |
| `_request_action_chunk()` | 首次动作、块结束或阶段切换后 | 采当前图像／qpos，组织阶段prompt与动作噪声；收到动作后核对形状、数值及本地动作版本 |
| `maybe_submit_phase()`→`_run_phase_assessment()`→`client.events()` | 已收到动作反馈、达到间隔且无在途评估、未停止 | 后台等待新观测，计算Touch和H100；发送phase请求并逐事件处理 |
| `finish_phase_result()`→`plan_payload_after_failure()`→`infer_with_timeout()` | need触发且完整failure返回后 | 组织故障观测快照与记忆；保持期间等待plan，成功后进入adjustment |

`executor.submit()`只安排后台任务；阶段请求直到新观测准备完成并进入`events()`时才真正send。当前评估提交点位于逐步动作执行、获取反馈之后，不是服务端独立运行的定时任务。评估未处理完时不发新评估，停止期间也不重复提交；动作／plan等待期间不会通过该调用点持续产生新请求。

底层动作／plan通信采用MessagePack编码后send一次、recv一次；评估通信send一次，先recv判断，need=true时再recv failure。两次接收属于同一次请求。`phase_future`在完整事件序列接收后才完成，但后台收到第一个true事件时就调用控制门停止旧动作，无需等Future完成。主循环随后在`finish_phase_result()`中消费完整结果并发起plan。

```text
正常动作请求 → 执行动作并发送need评估
    need=false：继续执行，后续再评估
    need=true：先停止 → 同一请求接收failure
               → 单独请求plan → 请求恢复动作
恢复动作执行期间发送end评估
    end=false：继续当前plan
    end=true：停止并切回execution → 请求新的正常动作
```

end=true不直接触发新plan；返回execution后再次检出故障，才进入下一轮诊断与规划。推理请求超时或返回非法数据时进入现有异常停止路径，发送函数本身不自动重发同一推理请求。

### 3.4 动作块生成与执行

#### 3.4.1 通过共享动作专家采样

主控制循环根据当前阶段组织动作条件。execution只包含原任务及当前观测；adjustment额外包含当前plan。每个动作请求都从当前机器人反馈构造新的输入，模型预测30步动作。

动作流匹配采样统一使用10个离散积分步。令\(S=10\)为采样步数、\(\tau_j=1-j/S\)，从高斯噪声块\(A^{(0)}\)出发进行Euler更新：

\[
A^{(j+1)}=A^{(j)}-
\frac1S v_\theta\left(A^{(j)},\tau_j,C^{\mathrm{act}}\right),
\qquad j=0,\ldots,S-1.
\]

采样中复用当前动作前缀的KV缓存。最终输出经反归一化、恢复绝对关节目标并取7维有效动作，得到\(30\times7\)控制序列。

客户端为每次试验指定随机种子，再依据阶段与该阶段动作块序号派生噪声，便于重复运行和记录。不同动作块使用不同噪声；同一组种子索引可以复现相应噪声输入，不据此保证闭环轨迹完全相同。

#### 3.4.2 按反馈推进动作块

控制器统一按30 Hz逐步发送关节目标，并等待比该条指令之前更新的真实反馈，再推进控制记录。每块最多执行30步；若need或end触发，可以在任意已发布步之后丢弃剩余部分。

动作发布前检查当前动作版本与停止标志，夹爪目标按机器人接口的标定偏移和最小开口约束处理。这里的约束属于控制接口，不将其解释为模型输出的夹取力控制。

没有阶段切换时，动作块执行到设定长度后重新采集当前观测，再生成下一块。发生阶段切换时，不继续执行原动作块的余下关节目标。

### 3.5 execution中的恢复触发

正常执行期间，后台以新的当前图像、qpos、Touch、H100及最近恢复计划上下文构造评估输入。首次尚无恢复计划时使用none。

服务端执行一次共享前缀编码，得到用于need分类的池化特征，以及failure语言解码可复用的KV缓存。令need的true概率为\(P_n\)，按阈值\(\eta_n\)形成判断：

\[
\hat n=\mathbb I[P_n\geq\eta_n].
\]

当need=false时，仅返回本次判断，继续正常执行；后台随后基于更新观测再次评估。当need=true时，服务端先发送恢复触发事件，机器人端立即设置停止标志、清除待执行的动作余量，并以当前反馈位置发送保持目标。

触发事件先于完整failure与plan返回，停止发布旧动作不需要等待两段文本全部生成。这里的“立即中断”指控制器收到有效触发事件后停止继续发布旧目标，不表示机器人具有零制动时间。

### 3.6 停止后的故障诊断与恢复规划

#### 3.6.1 复用need的观测与缓存生成failure

服务端在判断need之前完成一次共享prefill，同时准备分类特征、failure首token的logits和前缀KV缓存。模型封装层的返回值及用途为：

| 返回量 | 含义 |
| --- | --- |
| logits | 池化特征经need头得到的二分类logits |
| first | 预期由前缀最后一个有效位置的hidden state投影得到的failure首token logits |
| cache | 各Transformer层前缀位置的K／V，供语言续写使用 |
| mask | 前缀有效位置掩码 |
| position | 第一个生成token的语义位置编号 |

first是词表投影后的预测分数，不是2048维hidden state；它也不同于need头的二分类logits。need=true时复用的是主干KV缓存，不是分类头输出。服务端先yield判断事件，外层发送后再继续生成器；need=false直接结束，need=true调用`constrained_greedy_generate_from_prefill()`续写。

```text
观测前缀 → 一次prefill → 池化特征 → need头
                     ├→ first
                     └→ cache
need=true并发送判断后：
    first选择y1
    输入y1＋cache → 更新cache，预测y2
    输入y2＋cache → 更新cache，预测y3
    ……直到完整合法输出
```

每步`generation_step()`只嵌入当前选中的一个token，计算该位置的新Q／K／V，并读取缓存中的前缀与已生成token的K／V，不重新编码完整图像及prompt。单步attention允许读取有效前缀、此前生成位置和当前自身，屏蔽前缀无效槽；因每步仅新增一个位置，不存在未来答案可读。

failure与need共享触发时的同一份图像、qpos、Touch和H100快照；停止后新到达的观测不会自动写入旧缓存。因此，failure解释本次触发证据，而不是重新诊断停止后的新状态。首token位置的当前代码索引问题单列于1.12节。

failure采用语法约束下的贪心解码，每步仅在当前字段允许的token中选择概率最高项，直到生成合法完整文本。一次触发仅接受与该请求标识匹配的一条failure。诊断时客户端保持停止状态，不继续发布被中断动作块。

#### 3.6.2 根据当前及历史failure生成plan

取得当前全局failure \(f_k\)后，将其与此前最多4条历史failure组成按时间排序的序列：

\[
\mathcal S_k=(\mathcal M_k,f_k).
\]

该序列最多5条，当前项位于末尾。plan使用本次故障评估快照中的图像、qpos、任务指令及\(\mathcal S_k\)，不重新加入Touch、H100或历史plan。当前运行代码在诊断到规划之间复用触发时的视觉与状态快照，进入恢复动作前再采集新的执行观测。

由于plan前缀与评估前缀不同，规划单独执行一次前缀编码，不将need／failure的语言缓存直接用于新的规划上下文。plan同样按自身语法进行贪心解码，并在生成完成后校验字段与类别。

取得新plan后，将当前failure提交为下一轮的历史：

\[
\mathcal M_{k+1}=\operatorname{Tail}_4(\mathcal S_k).
\]

这一顺序使本次规划能读取4条过去反馈与1条当前反馈，下一轮记忆仍至多4条，不重复加入当前failure，也不保存历史plan作为规划输入。原型中的配对记忆仍需按第一节所列论文接口调整，此处明确采用已确认的failure-only协议。

### 3.7 启动恢复与恢复结束判断

plan就绪后，将阶段切换到adjustment。控制器先保持当前位置，采集新的两路图像与qpos，以adjustment和新plan为条件请求动作块。只有属于新阶段的有效动作块返回后才解除保持，避免将上一阶段的迟到动作用于恢复。

恢复期间共享动作专家持续按当前plan生成动作，后台评估改为end头。其输入为当前视觉、qpos、Touch、H100和当前plan：

\[
\hat e=\mathbb I[P_e\geq\eta_e].
\]

end=false时继续执行当前plan，并在后续动作块重新读取观测。当前实现不会在adjustment期间直接调用need、failure和plan进行另一次规划。

end=true时，后台事件使控制器停止发布当前恢复动作块的剩余目标并保持当前位置，随后切换到execution。一次adjustment阶段至此完成，恢复计数加1。

因此，端到端的多轮恢复路径为：

```text
execution → need触发 → hold → failure → plan
          → adjustment → end触发 → hold → execution
          → 若再次检出故障，进入下一轮诊断与恢复
```

end判断的是当前恢复操作是否完成，不保证故障已被永久消除。若返回execution后仍存在故障，下一次need会基于新接触反馈触发后续恢复。

### 3.8 返回原任务与历史的持续维护

恢复结束后保留全局failure历史，不在execution／adjustment切换时清空。最近一次plan也保留为状态评估上下文，供后续need及failure理解此前恢复目标；它不因此进入plan任务的历史输入。

原任务恢复执行时使用原始任务指令与最新当前观测重新生成动作。继续原任务不是回放被中断动作块，也不是从旧动作序列的中断索引机械续播，而是由正常执行策略根据恢复后的状态继续完成任务。

新任务回合开始时清空failure历史、当前恢复计划、动作队列与回合计数，并重新定义本回合历史起点。由此，H100在同一任务中连续，跨任务的恢复记录则不互相污染。

### 3.9 动作中断与过期结果处理

由于动作、评估及文本生成具有不同耗时，系统为动作请求和阶段评估维护独立版本号，并为每个评估请求分配唯一标识。

| 检查项 | 接受条件或处理规则 |
| --- | --- |
| 动作块返回 | 动作版本号仍有效，且未被阶段触发失效 |
| 分类事件返回 | 请求标识与当前在途请求一致，事件阶段与当前阶段一致 |
| failure返回 | 只接受当前execution触发所对应的故障事件 |
| 阶段切换 | 增加阶段及动作版本号，清空旧动作余量 |
| 新阶段启动 | 保持至新版本动作块就绪后再解除 |

每条控制指令发布前，在同一控制锁内检查停止标志和动作版本，再发送该条指令。这样，收到触发事件后的旧动作不会因为主线程尚未处理完整failure文本而继续发布。阶段已改变时，迟到的旧动作块或旧判断被丢弃，不用于当前控制。

对应状态为`stop_latched`、`action_generation`、`phase_generation`及当前有效`request_id`；`try_publish_action()`在锁内检查并发布。触发事件使旧动作版本失效，阶段切换再更新相应版本，新阶段仅通过`release_hold_with_fresh_actions()`解除保持。已在运行的模型前向不因此被强制取消，过期结果在客户端拒收。

保持目标由最新有效机器人反馈确定。协议错误、推理超时或反馈无效时，现有部署停止模型动作并维持停止状态，不将通信失败伪装成一个成功恢复结果。具体超时配置与任务成功／失败判据分别记录。

### 3.10 异步闭环算法概述

以下伪代码概括已确定的论文接口与当前状态机行为；failure-only记忆和10次试验预算按论文约定表达，不表示旧版运行脚本已自动实现全部接口修改。

```text
初始化：phase=execution，M=[]，plan=none，completed_recoveries=0
等待W30就绪，开始本回合qpos历史

主控制循环：
    取得当前图像与qpos
    按phase生成动作块；adjustment时附加当前plan
    新阶段动作有效后解除hold
    逐条发布有效动作，持续接收真实反馈
    无在途评估且满足请求间隔时，后台采集新观测并提交当前阶段判断

后台execution评估：
    共享前缀编码，计算need
    need=false：返回，等待下一份新观测
    need=true：先返回触发事件，立即作废旧动作并hold
               复用缓存生成当前failure f

主循环收到完整failure后：
    以故障观测、M+[f]单独请求plan，等待期间保持
    M=最近4条(M+[f])
    切换adjustment，等待新阶段动作块

后台adjustment评估：
    计算end
    end=false：继续当前plan
    end=true：作废旧动作并hold，completed_recoveries += 1
              切换execution，等待新的正常执行动作块

任务结束：外部按成功标准确认完成，或达到试验终止条件
```

当已完成10次恢复后仍需要启动下一轮恢复时，按任务试验预算终止并记录未成功；完成第10次恢复后若正常执行即可完成任务，仍按任务目标判定。10次上限属于试验协议，不是模型只能生成10条plan的结构限制。

### 3.11 推理时序与异步作用

记一次评估观测形成时刻为\(t_o\)，触发事件抵达客户端的时刻为\(t_d\)，故障文本和计划就绪时刻为\(t_f\)与\(t_p\)。旧动作从\(t_d\)起停止继续发布；\(t_d\)至\(t_p\)期间保持，plan及新阶段动作块就绪后进入恢复。

因此，恢复触发的停止等待不包含完整failure与plan文本生成时间。停止延迟仍包括观测更新、描述器计算、共享模型排队、need前缀计算、通信及控制器响应；实际末端停止还包含运动与制动过程。

同步对照在动作块结束后才保持并判断；异步方法允许物理动作执行期间评估，并在块内停止。两者差异来自判断与执行的时序组织，不来自两套网络、不同故障语义或额外并行模型。实际效果由同步／异步位移实验评价，不仅凭时序设计宣称固定的加速倍数。

### 3.12 任务预算、终止与运行记录

四个任务的试验均最多完成10个adjustment阶段，每完成一个阶段计一次恢复。failure事件数、plan输出数与已完成恢复阶段数分别记录，不互相代替。任务成功按第一节与实验部分的操作目标判断；end=true不作为任务成功标志。

当前客户端同时记录动作预算、推理超时、操作员成功确认及退出。旧版Black-box默认max_attempts与其他任务不设次数上限的代码，不能直接解释为论文统一的10次恢复协议；最终试验按已确认的计数定义执行。

运行日志保留：

- 试验标识、模型版本、归一化与captioner身份、随机种子及启动参数。
- 动作块条件、原始输出、恢复后的关节目标、实际发布指令与机器人反馈。
- 状态评估采集时间、请求时间、need／end概率、决策接收时间与处理阶段。
- failure、plan、规划输入的failure序列，以及阶段切换与被丢弃的旧结果。
- 恢复开始、end触发及实际停止位置，用于后续计算恢复位移；原任务后续动作不混入该位移。

服务端推理耗时、客户端请求往返耗时、分类触发耗时及末端实际停止耗时分别统计。名义发布频率与目标评估频率不替代上述实测时延。

### 3.13 推理参数与结束阈值校准

四个操作任务采用以下统一推理配置。除end阈值外，表中参数已由本轮讨论确认。

| 参数 | 论文采用设置 |
| --- | --- |
| Touch窗口 | W30 |
| qpos历史 | 30 Hz时间网格、H100→11点、约3.3秒 |
| 动作预测长度 | H30，内部32维、输出有效7维 |
| 每块最多发布步数 | 30，可由need／end提前中断 |
| 动作发布频率 | 四个任务统一30 Hz |
| need实际评估频率 | execution阶段约3 Hz |
| end实际评估频率 | adjustment阶段约3 Hz |
| need阈值 | 0.5 |
| end阈值 | 待验证集校准，不预设为0.5；下述校准方案待确认 |
| 动作采样积分步数 | 10 |
| 文本解码 | 按字段语法约束的贪心解码 |
| failure历史 | 最近4条过去failure，另加当前failure |
| 恢复试验上限 | 每任务最多10次已完成adjustment |
| 客户端推理线程／连接 | 线程池上限2、两条连接；各版本动作请求的等待方式见3.3节 |
| 服务端模型计算 | 单个共享模型worker串行处理 |

**真机部署权重。**四个任务均统一使用Stage B第2000步权重。各运行目录均已核实存在对应2000步的full_params与delta_params元数据；服务脚本的旧默认值不作为本文部署选点。

| 任务 | 已确认模型版本 | 验证best步数 | 服务脚本默认步数 | 论文实际部署步数 |
| --- | --- | --- | --- | --- |
| Black-box | V7.7.2 | 4000 | 由启动参数指定 | 2000 |
| Book | V9.5 | 2000 | 2000 | 2000 |
| Sponge | V10.1 | 1000 | 1000 | 2000 |
| Bottle | V11.2 | 2000 | 1000 | 2000 |

第二节保留各版本实际完成的训练总步数及历史best记录；统一2000步指本轮确定的部署选点，不改写已有训练记录。此前若有其他checkpoint的离线指标，仍保留其原选点身份，不能直接归为2000步的结果。夹爪标定偏移、最小开口及机器人控制器参数在第四节的平台与标定记录中补充。

#### 3.13.1 adjustment_end阈值的确定方法（建议方案，待确认）

end过早触发会在恢复尚未完成时切回原任务，而阈值过高也可能导致恢复迟迟不结束。因此，建议采用**在验证集上限制提前误判率、再最大化结束召回率**的规则，不直接按逐帧准确率最高或默认0.5确定阈值。四个任务分别训练模型，故分别校准一个end阈值，但使用相同的选择规则；同一任务内不同恢复方向与程度共用该阈值。

**校准数据与模型。**冻结所选2000步权重及归一化、Touch、H100和plan输入协议，使用与训练回合、最终测试回合分离的验证恢复轨迹。按3 Hz提取有时间顺序的评估观测，以人工标注的当前plan完成状态为依据，记录每个观测的标签\(y_i\)与end概率\(p_i\)。标签表示当前恢复操作是否可以结束，不表示原任务是否成功。校准使用真实恢复轨迹的自然时序，不采用训练时重采样后的类别比例，也不将重复的反事实样本当作独立真实片段。

**阈值选择。**对候选阈值\(\eta\)，以\(p_i\geq\eta\)为结束预测，计算：

\[
\operatorname{FPR}_{\mathrm{early}}(\eta)
=\frac{\sum_i\mathbb I[y_i=0,\ p_i\geq\eta]}
{\sum_i\mathbb I[y_i=0]},
\qquad
\operatorname{Recall}_{\mathrm{end}}(\eta)
=\frac{\sum_i\mathbb I[y_i=1,\ p_i\geq\eta]}
{\sum_i\mathbb I[y_i=1]}.
\]

建议沿用现有评估代码的1%提前误判率上限，选择：

\[
\eta_e^*\in\underset{\eta:\ \operatorname{FPR}_{\mathrm{early}}(\eta)\leq0.01}
{\arg\max}\ \operatorname{Recall}_{\mathrm{end}}(\eta).
\]

候选阈值由验证概率的不同取值及0、1组成；若多个阈值召回率相同，取较高阈值，与现有代码一致。1%为待确认的经验约束，具体阈值需实际计算后填写。若没有满足约束的工作点，或满足约束时召回率过低，记录该结果并重新检查模型或数据，不将“几乎不触发”视为理想工作点。

**连续恢复片段核查。**逐帧误判率低不等于整段恢复很少提前结束，因为在线系统在第一次end=true时即切换阶段。确定候选阈值后，按每段轨迹的时间顺序回放3 Hz判断，取首次触发时刻，与人工标注的可结束区间比较，另行记录提前结束片段数／总片段数、未触发片段数／总片段数，以及未提前结束且成功触发片段的结束延迟。离线回放用于核查判断时序，实际停止位置仍由真机验证，不将回放等同于机器人制动过程。

**固定与复现。**阈值选择和上述核查均在验证集完成，随后固定阈值用于真机测试，不根据最终20次任务结果回调。保存每任务阈值、验证混淆矩阵、片段统计、模型步数及权重身份；更换权重或输入协议时重新校准。该方案目前为方法草案，尚未生成2000步权重的校准结果。

已有逐帧选择逻辑见[select_max_recall_under_early_fpr](../src/tactile_vla/vla/v5_3_adjustment_end_evaluation.py)；验证集选阈值及固定阈值评估的实现可参考[calibrate_book_v9_3_thresholds.py](../scripts/calibrate_book_v9_3_thresholds.py)。连续片段首次触发核查是此处建议补充的评估，不将其表述为上述脚本已有功能。

### 3.14 实现对应与版本差异

本节采用已确认的论文版failure-only记忆与新字段接口，保留代码中已经核实的异步控制逻辑。需要同步实现的差异包括：旧运行脚本的plan–failure配对记忆、plan直接输入Touch、旧字段输出语法，以及各版本不同的恢复次数限制。这里不修改运行代码，也不将旧协议的运行记录直接改称为新协议测试结果。

| 推理内容 | 主要依据 |
| --- | --- |
| need先返回、failure后解码、动作流匹配调用 | [serve_tactile_vla_v7_7.py](../scripts/serve_tactile_vla_v7_7.py) |
| Black-box单worker与编译预热 | [serve_tactile_vla_v7_7_2.py](../scripts/serve_tactile_vla_v7_7_2.py) |
| Book单worker与运行配置 | [serve_tactile_vla_book_v9_5.py](../scripts/serve_tactile_vla_book_v9_5.py) |
| Sponge单worker与运行配置 | [serve_tactile_vla_sponge_v10_1.py](../scripts/serve_tactile_vla_sponge_v10_1.py) |
| Bottle流式服务与V11.2入口 | [serve_tactile_vla_bottle_v11_1.py](../scripts/serve_tactile_vla_bottle_v11_1.py)、[serve_tactile_vla_bottle_v11_2.py](../scripts/serve_tactile_vla_bottle_v11_2.py) |
| Black-box机器人端状态机 | [agilex_inference_v7_7_asyn.py](../openpi/inference/agilex/inference/agilex_inference_v7_7_asyn.py)、[V7.7.2入口](../openpi/inference/agilex/inference/agilex_inference_v7_7_2_asyn.py) |
| Book／Sponge机器人端控制 | [Book V9.5](../openpi/inference/agilex/inference/agilex_inference_book_v9_5_asyn.py)、[Sponge V10.1](../openpi/inference/agilex/inference/agilex_inference_sponge_v10_1_asyn.py) |
| Bottle观测、阶段切换与动作发布 | [Bottle共享主循环](../openpi/inference/agilex/inference/agilex_inference_bottle_v11_1_asyn.py)、[V11.2入口](../openpi/inference/agilex/inference/agilex_inference_bottle_v11_2_asyn.py) |
| 状态版本与旧结果失效 | [v7_7_async_state.py](../src/tactile_vla/vla/v7_7_async_state.py) |
| 评估观测新鲜性与反馈读取 | [agilex_inference_forced_phase_anlation_5_3.py](../openpi/inference/agilex/inference/agilex_inference_forced_phase_anlation_5_3.py) |
| W30与H100缓冲 | [tactile_buffer.py](../src/tactile_vla/runtime/tactile_buffer.py)、[state_history.py](../src/tactile_vla/runtime/state_history.py) |
| 缓存复用与语法约束解码 | [stage_b_v3_jax.py](../src/tactile_vla/vla/stage_b_v3_jax.py)、[structured_generation.py](../src/tactile_vla/vla/structured_generation.py) |
| 动作／plan单次发送接收 | [TimeoutWebsocketPolicy](../openpi/inference/agilex/inference/agilex_inference_forced_phase_anlation_5_3.py)、[WebsocketClientPolicy](../openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py) |

## 四、数据采集以及处理

### 4.1 数据体系与采集单位

训练数据分为两个部分：触觉描述器数据用于学习动态接触状态到元描述的映射；机器人操作数据用于学习动作生成、阶段判断、故障诊断与恢复规划。二者的标注目标不同，分别建立数据索引与训练／验证／测试划分。

**触觉描述器数据。**按area、Fx、Fy、Fz和rotation五个字段分别采集触觉片段。每个片段只为一个目标字段提供人工标签，其余字段不参与该片段的监督。模型由不同字段的片段共同训练，不要求每个触觉窗口都具备全部五字段标注。

**机器人操作数据。**通过主从臂遥操作，记录正常执行、出现故障以及恢复后继续原任务的轨迹。同步保存两路图像、从臂状态、主臂动作目标、双指触觉信号，以及任务指令、结果和故障／恢复元数据。冻结的触觉描述器随后离线生成Touch，再从同一批轨迹派生五项训练任务的数据。

为统一数据量的含义，区分三个单位：

| 单位 | 定义 | 统计方式 |
| --- | --- | --- |
| 原始回合episode | 一次采集组织下的任务实例，可包含失败尝试和后续恢复尝试 | 按原始episode_id去重 |
| 片段attempt | 回合内单独保存的一段采集；现有主实验数据主要包含attempt1与attempt2 | 按episode_id、attempt_id二元组去重 |
| 派生样本sample | 从某帧或窗口组织出的某项任务输入与监督 | 按任务索引计数，可包含多个文本变体 |

现有数据通常将一次失败的attempt1与随后恢复成功的attempt2保存在同一episode内。attempt2可以同时包含恢复动作与返回原任务后的动作，因此attempt编号不能直接替代execution／adjustment阶段标签。多种历史长度或反事实条件下，同一物理帧可形成多个训练样本，这些样本不增加实际采集回合数。

### 4.2 采集平台与操作任务

**已确认的平台。**实验采用Piper单臂及夹爪，两个手指安装千觉Xense触觉传感器。通过主从臂遥操作进行示教，从臂执行主臂输入的运动。模型使用六个机械臂关节加夹爪开合组成的七维状态与动作。

视觉系统包括固定主摄front和腕部相机wrist。相机具体型号按用户要求暂留空。当前LeRobot数据中的`observation.images.front`对应主摄，`observation.images.left`对应实际使用的腕部相机；left是保存键名，不表示额外增加一台相机。采集和保存频率统一按30 Hz组织，在线need／end约3 Hz的判断频率与原始轨迹采样频率分别表述。

四类任务分别组织采集数据并训练对应VLA：

| 任务 | 操作目标与接触异常 | 恢复相关信息 |
| --- | --- | --- |
| Black-box | 稳定夹持未知重心的长条并转移到指定区域 | 依据接触故障及历史反馈改变夹取位置，当前数据包含moderately与slightly两种程度 |
| Book | 将书插入书架，在插入受阻时恢复 | 根据接触偏差选择位置恢复方向，恢复后继续插入 |
| Sponge | 将物块移至指定区域，运动中可能遇到阻碍 | 实验设置包含两路相机均无法观察障碍物的遮挡条件，利用触觉判断受阻并恢复 |
| Bottle | 转开所需旋转角度未知的瓶盖，例如90°或180° | 检出打开受阻后重新组织夹取与旋转，再继续原任务 |

上述表格描述任务目标与实验情形，不表示每一种测试扰动都已经对应同等数量的示教。实际采集规模及已有标签分布在4.6、4.12节单独记录。

### 4.3 原始记录、时间戳与数据格式

#### 4.3.1 逐帧保存的内容

每段attempt保存独立数据文件，并保留原始回合与片段标识。现有HDF5数据已核实包含以下信号：

| 数据项 | 保存内容 | 后续用途 |
| --- | --- | --- |
| 图像 | front、wrist图像及其文件引用 | 当前视觉观测 |
| 从臂关节 | 七维关节／夹爪位置，另可保存速度与effort | 当前qpos、H100及执行反馈 |
| 主臂关节 | 七维关节／夹爪目标 | 动作模仿学习监督 |
| 双指网格 | 左右各自的mesh_3d、mesh_3d_flow | 描述器的网格位置及运动输入 |
| 双指力场 | 左右各自的三轴force | 描述器的受力输入 |
| 时间信息 | 同步帧时间戳和事件时间戳 | 事件对齐、窗口构造与日志追溯 |
| 任务元数据 | 指令、结果、故障标签、恢复方向／程度、有效性 | 派生五任务标签与筛选样本 |

当前文件中还存在合力／力矩等附加记录，但本文描述器使用网格、网格运动和力场，不将附加记录计为独立输入分支。

qpos使用真实从臂反馈，action使用主臂示教目标；两者不能互相代替。主从系统的夹爪量纲、开口映射及部署偏移需要与机器人接口一致，具体标定数值暂未确认，不在此编造。

#### 4.3.2 事件对齐与同步含义

已有数据profile将shift、move_start和rexecution等事件的时间戳，对齐到**不早于该事件的第一个同步帧**。因此，事件对应帧由实际时间戳确定，而不是按片段固定比例猜测。主实验的30 Hz用于定义保存帧率与时间尺度，不代替各模态原始时间戳。

原始记录经过时间对齐后转为HDF5，再构建LeRobot视频／表格数据。转换时保持episode_id、attempt_id、原始帧索引和ROS时间戳，使训练样本可以追溯到具体采集时刻。LeRobot中的episode_index属于存储编号，不能在划分数据时替代原始episode_id。

当前工作区包含通用采集、同步与转换工具，也保留实际数据处理产物；部分项目专用原始处理脚本不在当前工作区。离线多模态对齐采用的确切容差及全部话题的匹配规则尚待核对，不将当前数据称为经过严格硬件同步的观测。

### 4.4 触觉描述器的采集与人工标注

#### 4.4.1 按字段组织片段

用户已确认，论文版沿用按字段采集、每段只标注一个字段的方式。设片段的目标字段为\(j\)，类别为\(y^{(j)}\)，则该片段派生窗口的监督掩码为：

\[
m^{(j)}=1,\qquad m^{(k)}=0\quad(k\ne j).
\]

未监督字段使用无效标签并在损失中mask，不默认填为none、near_zero或appropriate。现有masked数据加载器也要求每个窗口恰有一个有效监督字段。不同字段的片段共同训练共享CNN与TCN，以及五个独立分类头。

标签依照第一节的论文版五字段定义：area描述接触面积类别；Fx与Fy描述双指切向受力的同向或异向组合；Fz描述法向接触及双指强弱差异；rotation描述左指传感器参考下的平面内旋转。

#### 4.4.2 类别边界与坐标参考

**已确认采用人工判读，不设固定数值阈值。**area各档位、near_zero与显著受力的界限，以及left_stronger／right_stronger中的明显强弱差异，均由采集与标注人员结合触觉信号判定。因此，方法中不将这些类别解释为固定接触面积百分比、确定的牛顿阈值或统一的左右受力比值。

标注协议固定左右传感器身份、通道顺序及参考方向。**已确认左右原始信号未经坐标统一，各自保留本传感器的局部坐标系；触觉标签统一以左指传感器局部坐标系为参考。**输入处理直接拼接双指信号，不增加坐标旋转或符号翻转步骤。人工标注根据左指参考方向解释双指接触状态，描述器学习从原始双指信号到统一语义标签的映射。

在该标签参考下，pos_neg和neg_pos分别表示左正右负、左负右正，不能直接用左右原始通道的符号比较替代人工标签。rotation的顺逆时针同样以左指传感器视角定义，具体观察方向在标注示意图中固定。

按片段赋标签时，片段内部需符合该目标类别的含义。类别发生改变的片段应按标注边界拆分后生成有效窗口，不能将跨类别过渡的窗口直接当作稳定类别样本。此处是论文版数据组织要求；尚未据此核实五字段数据的实际片段数量。

#### 4.4.3 W30窗口、归一化与划分

从每个标注片段内部提取30帧完整窗口，窗口末帧\(t\)对应输入\([t-29,t]\)，不跨不同片段或采集回合拼接。按30 Hz计，W30约对应1秒触觉观测。保存原始序列及窗口末帧索引，训练时按需读取，避免重复存储大量重叠窗口。

每帧的mesh_motion与force分别组织为\(12\times35\times20\)和\(6\times35\times20\)，按通道进行均值／标准差归一化；统计量由训练部分计算，并固定用于验证、测试与在线部署。

论文实验沿用要点总结中的按完整采集轨迹或回合划分原则：先分配原始片段或回合，再在各集合内部生成窗口，使高度重叠的窗口不会被随机分入不同集合。旧训练脚本存在window_random数据目录默认值，但这不能证明论文版数据已经按回合完成重建。五字段数据的最终训练／验证／测试片段数、字段类别数量与有效窗口数暂留空，待可读取相应数据索引后统计。

单帧重复输入实验使用同一数据划分与窗口标签，将窗口末帧重复30次作为输入，其他训练条件保持一致；这是对时序证据的消融，不新增一套按测试结果筛选的数据。

### 4.5 冻结描述器的离线Touch生成

描述器训练完成后冻结参数，对机器人示教中的双指网格与力场进行滑窗推理。每个有效时刻只输入截至当前的30帧触觉，不读取未来帧。预测类别按固定字段顺序序列化为Touch文本，并随原始帧标识保存。

生成流程为：

```text
检查时间戳、帧数、网格形状及有限数值
    → 取当前帧及其前29帧
    → 使用描述器训练统计量归一化
    → CNN—TCN与五字段分类
    → 按论文语法生成Touch
    → 保存帧标识、描述器身份与标签版本
```

现有离线脚本对每个attempt的前29帧写入中性占位描述，从第30帧开始才得到完整窗口预测；这些占位不是真实接触判定。在线部署则在任务开始前等待真实W30就绪。原始Touch滑窗在attempt起点预热，H100运动历史可跨同一episode的attempt连续维护，二者的边界处理不同。

已有四项训练产物使用的是旧版六字段Touch，其中Bottle还使用不同的描述器checkpoint。论文版五字段标签新增了异向切向力与Fz强弱类别，不能通过简单删除Fz_bias字段就获得等价监督。采用新描述器后，需重新生成对应Touch，并重建依赖Touch的标签和样本索引；本文不将旧版缓存直接称为五字段模型预测。

自动生成的Touch是VLA输入。触觉描述器字段准确性实验则使用独立人工标签作为评价依据，不把描述器自己的输出重新当作触觉真值。

### 4.6 操作数据的规模与回合划分

当前四组完整方法数据的profile和LeRobot元数据已逐项核对。原始episode按约80%／10%／10%划分训练、验证与测试，随机种子42；同一episode内的失败attempt与恢复attempt必须属于同一集合。划分按采集条件分组组织；Book的已有划分还保留历史分配，再按组补充新增回合，而不是每次重建时重新随机全部回合。

| 任务与版本 | 原始回合数 | 训练／验证／测试回合 | 一次成功回合 | 失败＋成功恢复回合 | attempt总数 | 保存帧数 |
| --- | --- | --- | --- | --- | --- | --- |
| Black-box V7.7.2 | 90 | 72／9／9 | 10 | 80 | 170 | 102,923 |
| Book V9.5 | 90 | 72／9／9 | 30 | 60 | 150 | 121,010 |
| Sponge V10.1 | 40 | 32／4／4 | 20 | 20 | 60 | 48,188 |
| Bottle V11.2 | 40 | 32／4／4 | 20 | 20 | 60 | 53,442 |

表中“失败＋成功恢复回合”指一个原始回合对应失败attempt1及成功attempt2，共计两段记录，不是两次独立采集回合。保存帧数包括随后可能被某项任务筛选排除的帧，不等于动作训练样本数。

四组数据分别训练模型，不将四个任务的回合混合后再次划分。历史文本变体、反事实样本与对应原始观测继承同一split。离线验证／测试回合与最终每任务20次真机任务试验分别记录，不能将两种测试次数混用。

### 4.7 图像、状态与动作的预处理

**图像。**读取当前front和wrist图像，按现有OpenPI输入变换调整为224×224，组成两路有效视觉输入。第三路占位图像由输入mask关闭，不视为额外观测。不默认增加历史图像、深度或点云分支。

**状态与动作。**从臂七维位置形成当前状态\(q_t\)，主臂连续30帧位置目标形成动作块\(A_t=(a_t,\ldots,a_{t+29})\)。动作训练前先转为相对当前状态的目标：

\[
\Delta a_{t,i}=a_{t+i}-q_t,\qquad i=0,\ldots,29.
\]

这里每个目标都减同一个当前\(q_t\)，不是相邻动作逐步求差。七个有效维度均参与相对目标处理，随后按数据统计量归一化，并适配模型内部32维空间；推理时使用相同统计量反归一化并恢复绝对目标。

VLA采用state与actions的分位数统计，H100使用与当前state一致的q01／q99。现有各版本复用来源V4训练动作索引计算的统计文件，后续动作过滤未必重算统计；因此，不将统计样本数误写成最新Stage A的有效动作起点数。训练统计、数据索引与部署统计通过身份记录保持对应。

**H100。**离线将同一episode的qpos按时间顺序组织，包含当前时刻及此前最多99帧，不跨episode；不足100帧时用回合首帧左填，再均匀取11点，最后一点始终为当前状态。历史点保留来源attempt和原始帧索引。

### 4.8 动作监督的阶段标注与筛选

#### 4.8.1 从示教事件确定阶段

attempt1属于execution。attempt2开始时通常先执行恢复，再继续原任务，但阶段边界依任务版本确定：

| 任务 | adjustment区间 | 返回execution的起点 |
| --- | --- | --- |
| Black-box V7.7.2 | 从恢复开始至标定恢复停止帧S，包含S | S＋1 |
| Book V9.5 | attempt2起点至原始rexecution事件R之前 | R |
| Sponge V10.1 | attempt2起点至R之前 | R |
| Bottle V11.2 | attempt2起点至R之前 | R |

Black-box的恢复运动开始／停止边界来自已有离线运动检测与审计记录；Book、Sponge与Bottle使用原始rexecution事件。它们不是同一个事件的不同名字，不在所有任务中统一套用S±10的标签规则。

#### 4.8.2 H30有效性与故障后动作排除

训练起点必须具有完整30步目标，并满足当前版本的阶段限制。在adjustment中，若30步目标会越过返回execution的边界，则排除该起点；不将一个plan条件下的恢复动作块与后续原任务动作拼为同一监督。

Sponge与Bottle还对失败attempt设置故障安全边界C：仅当\(t+29<C\)时保留该起点的动作监督。跨越C或从C之后开始的动作被排除，避免把持续受阻后的失败操作当作期望策略。被排除的原始帧仍保留，用于H100、need、failure和plan等其他任务。Black-box与Book当前版本没有据此统一加入相同的故障后过滤，按各自真实索引记录。

正常阶段只使用任务与当前观测作为动作条件；恢复阶段附加对应人工示教plan。动作标签来自真实遥操作，不由文本历史扩增或end反事实样本生成。

### 4.9 need_recovery监督的构造

need用于判断是否应从execution进入恢复，而不是预测整个任务最终是否成功。正样本来自故障已发生的有效观测；负样本包含故障发生前、一次成功执行，以及恢复后已返回正常执行的观测。恢复动作本身的早期观测不直接当作正常执行负样本。

现有主实验版本按约1:3构造正负样本，负样本尽量在三种来源间平衡。部分版本优先保留故障边界前的困难负样本，再补充其他负样本，以避免模型只学会区分明显接触与完全无接触。

| 任务 | 现有故障边界与need标签规则 |
| --- | --- |
| Black-box V7.7.2 | 沿用V4已标定的故障窗口；负样本包含故障前帧、一次成功轨迹和S之后的正常执行帧 |
| Book V9.5 | F之前为负；F及之后若Touch旋转与标注故障方向一致则为正，rotation=none为负，旋转方向不符的帧排除 |
| Sponge V10.1 | B为描述器首次输出指定受力失衡状态的帧，C=B＋3；B之前为负，[B,C)排除，C及之后为正；恢复后R及之后为负 |
| Bottle V11.2 | F为W30预热后首段连续至少18帧Fy positive的起点，C=F＋3；F之前为负，[F,C)排除，C及之后为正；R及之后为负 |

Book另在训练集构造配对rotation-none负样本：保持原始图像、qpos与历史等输入不变，仅将指定过渡区间当前Touch的rotation改为none，并设need=false。该文本干预只用于训练，不将其当作真实采集到的接触观测，也不与Black-box的end程度反事实混为同一项实验。

Sponge与Bottle的上述边界部分依赖冻结描述器的输出，属于规则派生监督；人工提供故障含义，不等于每一帧need标签都经过独立人工判断。换为论文版五字段Touch后，需要重新检查对应状态与边界，不机械保留旧字段名fz_bias或假定旧时间戳必然不变。

### 4.10 adjustment_end监督与程度反事实样本

#### 4.10.1 事实恢复轨迹的结束标签

end描述当前恢复操作是否可以结束，需同时依赖当前plan和已发生的运动。它不是“所有接触故障已消失”的标签，也不是任务完成标签。

**Black-box。**设恢复运动开始帧为A、停止帧为S，现有事实样本覆盖\([A,S+10]\)，其中\([S-10,S+10]\)为true，更早位置为false。停止点前后10帧的范围是现有训练标注容差，不等于部署时强制多执行10帧。

**Book、Sponge与Bottle。**以attempt2起点为A，以返回原任务事件R为参考，样本覆盖\([A,R]\)；\([R-10,R]\)为true，\([A,R-11]\)为false。训练集每段保留11个正样本、紧邻边界的10个困难负样本、中间段8个负样本和较早段4个负样本，即正负比1:2；验证与测试保留该区间全部事实帧。

上述边界来自训练数据协议。第三节的end概率阈值是在模型输出上选择的推理参数，两者分别确定，不通过修改测试标签来获得更好的阈值结果。阈值校准需使用自然时序事实片段，而不是上述经过平衡采样的训练分布。

#### 4.10.2 Black-box恢复程度反事实

Black-box为同一物理状态组织事实与反事实plan条件，图像、Touch、qpos、H100及当前帧身份保持一致，仅改变plan中的程度词：

| 来源示教 | 替换后的plan程度 | 反事实end标签 |
| --- | --- | --- |
| moderately | slightly | 对\(p\in[A,S]\)，当\(p\geq\lfloor(A+S)/2\rfloor\)时为true，否则为false |
| slightly | moderately | 在已观测的\([A,S]\)区间均为false |

同一观测由此可因计划目标不同而得到不同监督，促使end利用plan程度与运动历史，而非只凭接触标签判断。中点是帧位置的代理标注，不代表已测得恢复距离的一半；短轨迹改成moderately时，也不假设存在未采集的后续运动或已知结束位置。

现有Black-box训练采样保留正例、moderately→slightly配对样本及停止附近困难负例，并对部分较早负例下采样以达到1:2。验证与测试保留完整定义区间，事实与反事实分开统计。相同物理帧的文本变体不增加独立试验数量，也不用于替代真机中slightly／moderately实际位移的验证。

### 4.11 failure、plan与历史文本的标注

#### 4.11.1 人工故障与恢复计划

用户已确认，failure以人工结合图像与触觉的标注为依据，plan依据人工标注的恢复示教方向与程度生成。failure描述当前全局故障状态，plan描述针对该状态的恢复动作；不能直接将failure方向词复制为plan方向而省略二者语义的区别。

论文版failure按grasp_position、shear、grasp_force、grasp_balance和rotate五字段序列化，遵守第一节的主摄参考固定场景坐标系。故障可同时涉及多个字段；每字段使用相应的正常或异常类别。旋转偏差按固定场景Z／Y／X轴的yaw、pitch、roll正负定义，不能把旧版rotate left等标签直接改名为yaw_pos。

论文版plan按horizontal、vertical、rotate和gripper_distance四字段组织。不活动字段写none，活动字段写方向及程度。水平／垂直使用场景坐标参考；旋转使用已确认的末端腕部关节轴；gripper_distance描述开合及其程度。

**已确认程度词仅采用人工定性标注，不规定固定数值范围。**slightly、moderately等词表示示教中对应动作维度的相对程度，不绑定固定毫米数、角度或夹爪开口变化量。动作模型根据当前观测与plan生成运动，end结合计划条件和运动历史判断结束时机；不同程度实际产生的位移在真机实验中测量，不将其解释为预先指定的数值控制目标。

#### 4.11.2 故障观测窗口与监督来源

在人工或规则确定的故障观测窗口内，用对应真实图像、qpos、Touch与H100构造failure输入，用人工failure作为目标。现有四组数据在训练时使用15帧故障窗口，验证／测试选窗口末端一帧，以减少高度相邻观测对离线评估数量的重复贡献。

Book、Sponge与Bottle当前推理样本窗口为\([C,C+14]\)。Book以原故障帧F加方向相关偏移确定C：left为F＋27，right为F＋8；该C用于稳定诊断窗口，不等于所有方向need首次应为true的位置。Black-box沿用既有V4故障窗口及其人工标签。

plan目标来自恢复示教或历史构造规则所确定的人工语义。对Book、Sponge与Bottle，当前目标保留相邻真实成功attempt2的计划；Black-box的扩增历史还按方向反馈规则改变程度。后续成功轨迹可提供训练目标，但其未来图像、未来状态或未来Touch不进入当前推理输入。

#### 4.11.3 failure-only规则历史

**已确认继续使用规则构造的训练历史，并改为failure-only文本。**设当前全局故障为\(f_k\)，构造最多4条过去failure的有序序列\(\mathcal M_k\)，与当前failure区分保存：

\[
\mathcal M_k=(f_{\max(1,k-4)},\ldots,f_{k-1}),
\qquad |\mathcal M_k|\leq4.
\]

plan输入为当前通用观测、\(\mathcal M_k\)及\(f_k\)，不包含Touch、H100或历史plan。历史按从早到晚排列；无过去故障时为空序列，当前故障单独提供。规则变体保持同一真实当前观测，只改变历史全局故障文本及与该历史语义相符的目标计划，不伪造历史图像、qpos或机器人执行记录。

Black-box的既有历史程度规则可概括为：尚未出现方向反转时采用moderately，首次出现相反方向的故障反馈后采用slightly，之后不因方向再次变化而恢复为moderately。该规则用于构造依赖反馈历史的程度监督，具体方向关系须按论文新failure语义重新核对；不能机械使用旧rotate标签替代新旋转轴标签。

已有Book构造不同左右故障前缀，但保持当前真实故障与相邻恢复示教plan不变；Sponge与Bottle主要重复同一故障和对应计划以扩展记忆长度。删除历史plan后，这些样本仍主要用于学习历史接口与对当前反馈的响应，不能仅凭重复前缀证明模型已学会在这些任务中随历史改变恢复程度。

规则历史保存来源类型、规则版本、随机种子、原始观测标识与变体标识，并继承其原始episode的split。训练历史是标注／规则文本，在线历史则是模型在本回合此前实际输出的failure。两者来源不同，方法不将规则文本前缀描述成完整的真实长程恢复采集。

### 4.12 当前五任务索引的样本规模

以下统计直接读取已确认版本的训练索引。action列为有效H30起点数；其余列为相应任务实际选用的manifest行数。end包含该版本保留的事实／反事实变体，plan可能包含不同历史长度的文本变体，因此各列不能相加解释为独立采集量。

| 任务 | 集合 | action | end | need | failure | plan |
| --- | --- | --- | --- | --- | --- | --- |
| Black-box V7.7.2 | train | 66,397 | 9,660 | 27,432 | 960 | 1,920 |
| Black-box V7.7.2 | val | 8,344 | 1,696 | 3,300 | 8 | 16 |
| Black-box V7.7.2 | test | 8,341 | 1,872 | 3,504 | 8 | 16 |
| Book V9.5 | train | 91,880 | 1,584 | 17,572 | 720 | 2,880 |
| Book V9.5 | val | 11,096 | 1,303 | 2,392 | 6 | 24 |
| Book V9.5 | test | 11,944 | 1,243 | 2,384 | 6 | 24 |
| Sponge V10.1 | train | 35,342 | 528 | 4,368 | 240 | 960 |
| Sponge V10.1 | val | 4,554 | 403 | 632 | 2 | 8 |
| Sponge V10.1 | test | 4,575 | 481 | 588 | 2 | 8 |
| Bottle V11.2 | train | 39,642 | 528 | 5,512 | 240 | 960 |
| Bottle V11.2 | val | 4,764 | 448 | 692 | 2 | 8 |
| Bottle V11.2 | test | 4,670 | 407 | 656 | 2 | 8 |

这里统计的是当前数据版本，不是论文新五字段Touch、统一failure、四字段plan及failure-only历史完成重建后的数量。重建时如果仅去除历史plan且保持样本一一对应，plan行数可能不变；若重新标注边界、改变历史规则或删去无效样本，数量需重新计算。表中不预先写入尚未生成的新数据量。

### 4.13 数据检查、追溯与处理流程

数据处理按以下顺序组织：

```text
字段级触觉采集与人工标注
    → 按原始轨迹／回合划分
    → W30索引及训练归一化统计
    → 训练并冻结五字段描述器

主从臂机器人示教＋图像＋双指触觉
    → 时间对齐与HDF5／LeRobot保存
    → 冻结描述器离线生成Touch
    → 原始episode级训练／验证／测试划分
    → 阶段及故障边界、人工failure与恢复plan
    → action／need／end／failure／plan索引
    → failure-only规则历史与end程度反事实
    → 固定版本的数据产物供Stage A、Stage B及离线评价读取
```

上述是论文版目标流程，部分步骤可在工程上调整先后，但必须满足以下对应关系：先确定原始回合的集合归属，再让所有窗口与变体继承；归一化统计只使用训练数据；当前输入不能包含未来观测；用于最终评价的人工标签与输入Touch的自动预测来源分别保存。

现有索引已实现或检查的项目包括：时间戳有效及顺序一致，数组形状与有限数值，事件帧位于有效片段，H30及诊断窗口完整，原始回合不跨集合，文本字段／token属于合法语法，以及数据、描述器和归一化文件的哈希对应。部分新版索引还绑定动作筛选与captioner身份，避免数据变化后继续使用旧缓存。

标签版本迁移后重新执行这些检查。尤其是旧版旋转语义、新Fz强弱类别和gripper_distance不能只靠字符串替换获得可靠监督；需要以人工标注和真实示教为依据，保留旧字段到论文字段的可追溯关系。

### 4.14 本节确认状态与实现依据

**本轮已确认：**

- Piper单臂、双指千觉Xense传感器，通过主从臂遥操作采集；相机型号暂留空。
- 五字段触觉数据按字段采集，每段仅监督一个字段，其余字段mask。
- 触觉类别由人工判读，暂不设置固定数值阈值。
- 双指原始信号保留各自传感器坐标系，直接拼接；Touch标签统一以左指传感器局部坐标系为参考。
- failure由人工结合图像与触觉标注；plan依据人工标注的恢复示教方向及程度生成。
- plan程度仅作人工定性标注，不预设位移、转角或夹爪开口的固定数值范围。
- 训练沿用规则构造的历史，转换为failure-only；在线保存实际预测的最近4条过去failure。

**待补充或确认：**

| 编号 | 待定内容 | 当前处理 |
| --- | --- | --- |
| D1 | 五字段描述器数据的片段数、各类别数量及最终划分索引 | 未找到可读取的新版数据产物，数量留空；按完整轨迹／回合划分 |
| D2 | 左指旋转参考的具体观察方向及坐标示意图 | 标签参考已确认；示意图需标出固定观察方向 |
| D5 | 相机型号、离线同步容差、主从夹爪映射及标定数值 | 暂留空，后续按平台记录补充 |

第三节的end阈值校准方案仍为建议，进入本节不代表其1%约束已经确认或阈值已经算出。

| 内容 | 主要依据 |
| --- | --- |
| 通用采集、同步与格式转换工具 | [data_tools/README.md](../data_tools/README.md)、[dataSync.cpp](../data_tools/src/dataSync.cpp)、[data_to_hdf5.py](../data_tools/scripts/data_to_hdf5.py)、[hdf5_to_lerobot.py](../data_tools/scripts/hdf5_to_lerobot.py) |
| masked触觉窗口与单字段监督检查 | [tactile_captioner_dataset.py](../src/tactile_vla/data/tactile_captioner_dataset.py) |
| 双指通道组织与传感器身份 | [tactile_buffer.py](../src/tactile_vla/runtime/tactile_buffer.py) |
| 离线Touch滑窗、预热与文件检查 | [annotate_demo_tactile_captions.py](../scripts/annotate_demo_tactile_captions.py) |
| 状态／动作来源、图像键与输入变换 | [openpi_bridge.py](../src/tactile_vla/vla/openpi_bridge.py)、实际HDF5与LeRobot元数据 |
| VLA训练统计与回合索引 | [compute_vla_norm_stats.py](../scripts/compute_vla_norm_stats.py)、[v4_data.py](../src/tactile_vla/vla/v4_data.py) |
| Black-box五任务数据、事实／反事实与采样 | [build_v7_7_2_multitask_data.py](../scripts/build_v7_7_2_multitask_data.py)、[v7_7_2_multitask_data.py](../src/tactile_vla/vla/v7_7_2_multitask_data.py) |
| Book阶段、need与诊断窗口 | [book_stage_a_data.py](../src/tactile_vla/vla/book_stage_a_data.py)、[book_v9_5_need_data.py](../src/tactile_vla/vla/book_v9_5_need_data.py)、[book_v9_5_multitask_data.py](../src/tactile_vla/vla/book_v9_5_multitask_data.py) |
| Book／Sponge／Bottle的end区间采样 | [book_v9_2_adjustment_end_data.py](../src/tactile_vla/vla/book_v9_2_adjustment_end_data.py) |
| Sponge故障过滤与规则历史 | [sponge_v10_1_stage_a_data.py](../src/tactile_vla/vla/sponge_v10_1_stage_a_data.py)、[sponge_v10_1_multitask_data.py](../src/tactile_vla/vla/sponge_v10_1_multitask_data.py) |
| Bottle训练数据与动作阶段 | [bottle_v11_1_multitask_data.py](../src/tactile_vla/vla/bottle_v11_1_multitask_data.py)、[bottle_v11_2_stage_a_data.py](../src/tactile_vla/vla/bottle_v11_2_stage_a_data.py)、[bottle_v11_2_multitask_data.py](../src/tactile_vla/vla/bottle_v11_2_multitask_data.py) |
| 文本历史扩增来源 | [book_v9_4_memory.py](../src/tactile_vla/vla/book_v9_4_memory.py)、各任务当前plan manifest |
| 数据规模与split统计 | 四组运行所引用的profile.json、splits.json、meta/info.json及五任务训练索引 |
