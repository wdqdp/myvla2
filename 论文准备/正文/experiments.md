# 正文

## 4.1 实验设置

**平台与任务。**实验采用Piper单臂与双指夹爪，两指安装千觉Xense触觉传感器，视觉观测来自固定主摄和腕部相机。平台布置与配图见附录A.1。四项任务的道具与结构设计如图2所示，故障发生、恢复及原任务续执行过程见图3。

![四项任务的道具与结构设计](tools.png)

*图2．四项任务的道具与故障设计。从左至右为Black-box、Book、Bottle和Sponge：Black-box展示外观与内部可移动配重；Book通过改变两侧书籍数量形成不同插入约束；Bottle的剩余旋开角度设为0°—180°；Sponge在海绵上加装纸盘，右侧给出受阻时的腕部与主摄图像，两路相机均无法直接观察到障碍物。*

**Black-box：未知重心物体转移。**任务指令为“稳定提起并转移物体”。我们通过移动盒内铁块改变重心，同时保持盒体外观不变（图2）。初始夹取位置不合适时，物体提起后可能倾斜。系统需要根据接触故障改变夹取位置并重新抓取，必要时结合全局故障历史减小纠正幅度，直至稳定夹持后继续转移。图3上方额外展示左右触觉传感器的Tz扭矩曲线，辅助观察故障与恢复引起的接触变化。

**Book：不同约束下的书籍插入。**任务指令为“拿起书并将其水平放置在书架上”。我们改变插入位置两侧的书籍数量，形成不同的侧向约束，使书在插入过程中面临不同的卡滞情形。发生受阻后，系统需要根据接触反馈判断偏差方向，改变书的横向插入位置，解除卡滞后继续插入。

**Bottle：未知旋开角度的瓶盖开启。**任务指令为“打开瓶子”。单次旋拧最多允许90°，瓶盖剩余旋开角度在0°—180°范围内随机设置，且不提供给机器人。当瓶盖尚未完全旋开时，向上提起会受阻，这一内部约束无法由图像直接判断。系统需要识别上提受阻，回落并继续旋拧，再次尝试提起，直至完成开盖。

**Sponge：遮挡条件下的海绵移动。**任务指令为“将黑色海绵放入黑色框内”。我们在被夹持的海绵上加装纸盘，遮挡移动过程中的障碍物。图2右侧给出了受阻时的两路相机图像，腕部与主摄均无法直接观察到障碍物。系统需利用触觉识别受阻，先回退解除接触，再抬高海绵绕过障碍，随后继续向目标区域移动。

![四项任务的执行、故障与恢复过程](tasks.png)

*图3．四项真机任务的故障恢复示例，自上而下为Black-box、Book、Bottle与Sponge。灰色、红色和绿色阶段条分别表示正常执行、故障与恢复；Need与End标记恢复的启动和当前恢复操作的完成。故障帧中展示相关的诊断与计划字段。上方曲线对应Black-box示例中左右触觉传感器的Tz扭矩变化。*

**评价指标。**任务成功率（Task Success Rate，TSR）为完成指令目标的测试回合比例。Need Accuracy（NA）按Need头正确检出的真实故障次数/实际故障次数计算；实际未发生故障却触发Need时，记为一次误检，即一次检测失败，不计入成功检出的真实故障。由于部分测试不发生故障，而部分测试会出现多个故障，我们合并所有测试中的真实故障事件，计算恢复成功率（Recovery Success Rate，RSR），即成功纠正并能继续原任务的故障次数/实际故障次数。故障总数包含未检出或未启动恢复的故障。计划正确率（Plan Accuracy，PA）为人工判定正确的计划数/全部计划数，包含最终失败回合中的输出。表1的NA按真实故障事件计算，误触发现象另作定性分析。

## 4.2 真机任务表现

**评估设置。**每种方法在每项任务上测试20次，成功条件是在最多10次恢复内完成任务。无触觉版本（w/o tactile）移除Need、End和Failure输入中的触觉描述后重新训练，其他输入、训练与部署设置均与有触觉版本保持一致。

**任务完成表现。**如表1所示，TARO在四项任务上均优于无触觉版本，平均TSR由38.75%提高至85.00%，提升46.25个百分点。Black-box由3/20提高至18/20，提升75个百分点；Book、Sponge和Bottle分别提升35、40和35个百分点。

**表1．四项真机任务的分条件结果与恢复表现。C1—C5为各任务独立编号的测试条件，具体设置见附录A.2；条件单元格为成功数/失败数（S/F），“—”表示不适用。TSR、NA、RSR与PA以百分比报告，括号内为原始分子/分母。总体行的TSR为四任务均值，NA与RSR按真实故障事件汇总，PA按计划计数汇总。**

| 任务 | 方法 | C1 | C2 | C3 | C4 | C5 | TSR ↑ | NA ↑ | RSR ↑ | PA ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Black-box | w/o tactile | 2/2 | 0/4 | 1/3 | 0/4 | 0/4 | 15.00（3/20） | 81.25（13/16） | 18.75（3/16） | 46.15（6/13） |
| Black-box | TARO | 3/1 | 4/0 | 3/1 | 4/0 | 4/0 | **90.00（18/20）** | **97.50（39/40）** | **95.00（38/40）** | **97.44（38/39）** |
| Book | w/o tactile | 2/3 | 3/2 | 1/4 | 3/2 | — | 45.00（9/20） | 47.37（9/19） | 42.11（8/19） | 50.00（8/16） |
| Book | TARO | 4/1 | 4/1 | 5/0 | 3/2 | — | **80.00（16/20）** | **93.33（14/15）** | **73.33（11/15）** | **93.33（14/15）** |
| Sponge | w/o tactile | 1/4 | 4/1 | 2/3 | 1/4 | — | 40.00（8/20） | 40.00（8/20） | 40.00（8/20） | 53.33（8/15） |
| Sponge | TARO | 3/2 | 5/0 | 4/1 | 4/1 | — | **80.00（16/20）** | **95.00（19/20）** | **80.00（16/20）** | **100.00（19/19）** |
| Bottle | w/o tactile | 6/4 | 5/5 | — | — | — | 55.00（11/20） | 47.06（8/17） | 47.06（8/17） | 53.33（8/15） |
| Bottle | TARO | 10/0 | 8/2 | — | — | — | **90.00（18/20）** | **80.00（8/10）** | **80.00（8/10）** | **100.00（8/8）** |
| 总体 | w/o tactile | — | — | — | — | — | 38.75（31/80） | 52.78（38/72） | 37.50（27/72） | 50.85（30/59） |
| 总体 | TARO | — | — | — | — | — | **85.00（68/80）** | **94.12（80/85）** | **85.88（73/85）** | **97.53（79/81）** |

<!-- 排版建议：使用table*跨双栏，并以tabular*和extracolsep fill将表格撑满textwidth。C1—C5组成“分条件成功/失败数”表头组，Task、Method与四项指标用multirow跨两行上下居中。英文版使用small字号与2.5pt基础列间距。条件计数为S/F，指标括号为分子/分母，两者不要混淆；不得将C1等编号解释为跨任务相同配置。 -->

**故障检测。**合并四项任务后，NA由52.78%提高至94.12%。收益在视觉难以判断真实接触状态的任务中尤其明显：Sponge由40.00%提高至95.00%，Bottle由47.06%提高至80.00%。前者的障碍物被遮挡，后者的瓶盖连接状态难以从图像直接辨别，触觉为是否需要恢复提供了实际接触依据。

测试观察进一步说明了缺少触觉时的判断偏差。在Bottle中，无触觉模型会在旋转量不足时上提夹爪，也会在已经达到开盖角度后继续旋转，导致任务失败。在Sponge中，无触觉模型表现出对固定恢复位置的依赖：即使没有障碍物，移动到该位置时仍会触发Need。这类误检说明，模型可能将场景位置作为恢复触发的依据。触觉反馈有助于将阶段判断与实际接触状态关联。

**故障理解与恢复规划。**触觉还通过Failure历史影响Plan。去除Failure的触觉输入后，历史中用于描述接触状态的信息随之减少，故障理解的偏差也可能传递给后续规划。完整方法的PA由50.85%提高至97.53%；在Sponge和Bottle中均达到100%，而无触觉版本均为53.33%。这些结果说明，触觉不仅用于判断是否恢复，还能通过全局故障历史支持恢复计划的生成。

**故障恢复。**按所有实际故障事件汇总，RSR由37.50%提高至85.88%。值得注意的是，Black-box无触觉版本的NA已达81.25%，但RSR仅为18.75%；TARO的RSR则达到95.00%。这表明，发现异常之后仍需要可靠的故障理解、恢复规划与结束判断，才能将故障检测落实为有效恢复。

## 4.3 全局故障历史对恢复规划的作用

本实验检验将局部触觉经VLM理解为全局Failure的价值，以及历史信息对恢复程度推理的作用。同一局部触觉描述在不同夹爪朝向下可能对应不同的全局故障，仅保存局部触觉会使各轮反馈的空间含义产生歧义。此外，恢复幅度还取决于此前纠正的反馈。例如，未知重心物体在重新夹取后向相反方向偏转，可能意味着前一次纠正过量，此时需要减小下一轮幅度。为此，我们在Black-box任务上构造面向Plan的历史数据，检验全局故障历史能否帮助模型推理正确的恢复计划。

我们比较三种输入设置：**w/o history**（A）仅根据通用输入直接生成Plan；**tac-history**（B）在通用输入外加入局部触觉描述历史；**TARO**（C）使用通用输入与全局故障历史。各组通用输入一致，为了关闭模型通过机械臂姿态判断plan的捷径，本消融统一屏蔽机器人状态输入。三种测试从相同Stage A权重初始化，仅训练Plan任务的共享PaliGemma LoRA。我们额外采集了blackbox任务的40个旋转失败片段，左右旋转故障分别20个。每个片段采样30个样本，plan人工标定，历史通过规则离线构建。为了模拟同一个触觉描述对应不同全局故障情况，我们在构造历史时随机设定正反抓取历史。详细描述在附录中。

**评价指标。**方向准确率（DA）与程度准确率（MA）分别衡量水平恢复方向和程度是否正确；联合准确率（JA）要求两者同时正确。多轮联合准确率（MJA）在具有历史反馈的240条样本上计算JA；反向相关联合准确率（RJA）在全局故障方向发生过反转的120条样本上计算JA。

**表2．历史表示对离线Plan推理的影响。总体为360条测试样本，多轮子集为240条，反向相关子集为120条。TARO组使用标注的全局Failure。**

| 方法 | DA ↑ | MA ↑ | JA ↑ | MJA ↑ | RJA ↑ |
| --- | --- | --- | --- | --- | --- |
| w/o history | 80.83% | 66.67% | 53.89% | 40.42% | 0.00% |
| tac-history | 100.00% | 66.67% | 66.67% | 50.00% | 0.00% |
| TARO | 100.00% | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

**全局方向判断。**仅依赖当前视觉的w/o history在部分观测上判断错误，DA为80.83%；加入局部触觉历史的tac-history与使用全局故障历史的TARO均达到100%。这表明，引入触觉相关反馈有助于模型结合视觉理解接触故障的全局方向，并选择恢复方向。

**历史对恢复程度的作用。**w/o history缺少此前的故障反馈，对所有样本均输出moderately，在需要根据方向反转减小幅度的样本上RJA为0%。对于相同的当前观测，是否发生过纠正过量取决于此前反馈，因此恢复程度的选择需要历史信息。

**局部与全局历史的作用。**tac-history虽然保留了局部触觉反馈，仍始终输出moderately，RJA为0%。局部描述以传感器坐标系为参考，在历史抓取朝向不同时，模型难以仅凭这些描述判断全局故障方向是否反转。TARO使用统一到全局坐标系的Failure历史，能够根据方向反转选择slightly，MJA与RJA均达到100%。本对照中TARO使用人工标注的全局Failure，评估的是全局故障表示及其历史对Plan推理的作用。

## 4.4 触觉描述器的准确性、时序作用与开销

| 元字段 | 类别 | 含义 |
| --- | --- | --- |
| area | `none / small / medium / full` | 接触面积状态 |
| Fx | `negative / near_zero / positive / pos_neg / neg_pos` | 以左指传感器局部 X 轴为参考的双指受力方向组合 |
| Fy | `negative / near_zero / positive / pos_neg / neg_pos` | 以左指传感器局部 Y 轴为参考的双指受力方向组合 |
| Fz | `near_zero / negative / left_stronger / right_stronger` | 法向受力状态及双指负向受力强弱差异 |
| rotation | `none / clockwise / counterclockwise` | 以左指传感器为参考的平面内旋转状态 |

1．属性预测准确性
分别评估area、Fx、Fy、Fz和rotation：
- Accuracy：该属性预测正确的样本数／该属性有效标注样本数。
- Macro-F1：先计算该属性各类别的F1，再等权平均，避免数量较多的类别掩盖少数类别的错误。
- 五属性平均值按属性等权计算，避免采集数量不同影响总体结果

2．推理开销
- 参数量（M）：各模型生成触觉描述所需的总参数量。
- 推理时延（ms）：输入准备完成后，到完整触觉文本生成结束；报告中位数及P95。
- 峰值显存（GB）：模型加载并执行推理时的峰值占用。
统一硬件、batch size=1，预热后重复测量，并注明精度、输入规模和生成文本长度。Octopi1.5采用其原生输入与输出形式，因此这组比较用于说明生成触觉描述的资源开销，不直接代表相同接触识别能力。

待做事项：
- 补全之前没有采集的数据，重新训练一版本触觉captioner
- 测试评估指标
- 测试推理开销以及octopi1.5的推理开销


## 4.5 异步判断对恢复执行的影响

为了更好地利用触觉信号高频的特点我们设计了异步need/end判断。高频的判断让我们的模型能够更快识别到故障并且做出更精确的恢复动作。为了验证异步部署的有效，我们在Black-box的五种重心条件下，分别对同步与异步版本进行20次任务测试。同步版本在动作块执行结束后等待状态判断；异步版本在物理动作执行期间评估接触状态，判断触发后中断当前动作。need与end判断的快慢在部署中表现出来的是故障动作与恢复动作的运动距离。在Black-box任务中对应抬起高度与水平恢复位移，我们比较两种调度方式下的实际运动距离。

如表3所示，异步版本在四类水平恢复中的平均位移均更小。例如，异步版本的平均抬起高度比同步版本低1.9 mm，左移moderately和右移moderately的平均位移分别减少43.79%和20.65%，运动距离与示教数据更接近。此外，同步版本可能因移动距离过长而错过最佳夹持点，导致成功率低于异步版本。

**表3．同步与异步状态判断下的实际位移（mm，均值±标准差）。每组在Black-box上测试20次任务，位移按任务中的恢复片段分组统计。**

| 判断方式 | 抬起高度 | 左 moderately | 左 slightly | 右 moderately | 右 slightly |
| --- | --- | --- | --- | --- | --- |
| 同步 | 41.2±38.5 | 86.1±29.3 | 36.9±15.6 | 67.3±4.8 | 39.1±11.1 |
| 异步 | 39.3±20.9 | 48.4±4.3 | 17.1±2.5 | 53.4±8.6 | 22.6±3.1 |


## 4.6 运动历史对恢复进度判断的作用

当前机器人状态难以单独反映恢复过程中已经执行的运动，而判断计划中的程度是否完成需要此前运动信息。为确定运动历史的作用及适用分支，我们进行输入设计消融，比较H0、H30、H60与H100四种窗口。在该消融版本中，Need、Failure与End同时读取各组对应的运动历史。H0将当前机器人状态重复11次；H30、H60分别从包含当前状态的最近31、61帧采样11点；H100从最近100帧采样11点。各组均保留当前机器人状态，其他输入相同。

H0、H30和H60模型沿用与H100相同的数据划分、监督与初始化，分别进行训练。表4报告验证集结果，包括Need宏平均F1（NF1）、End准确率（EA）和End宏平均F1（EF1）。宏平均F1为正、负两类F1的算术平均，EA为全部验证样本的判断正确率。此外，报告原始真实样本上的误报结束率（FPR），即其中不应结束的样本被误判为结束的比例。

**表4．运动历史的验证结果。所有指标均以百分比（%）报告；↑表示越高越好，↓表示越低越好。H100作为既有模型参考。**

| 历史窗口 | NF1 ↑ | EA ↑ | EF1 ↑ | FPR ↓ |
| --- | --- | --- | --- | --- |
| H0 | 99.40 | 81.25 | 76.51 | 22.22 |
| H30 | 99.52 | 88.80 | 84.72 | 14.44 |
| H60 | 99.52 | 90.51 | 86.68 | 9.86 |
| H100参考 | 99.27 | 93.46 | 90.90 | 6.81 |

从H0到H100，EA由81.25%提高至93.46%，EF1由76.51%提高至90.90%，分别增加12.21和14.39个百分点；事实样本上的FPR由22.22%降至6.81%。H30与H60的结果介于两者之间，支持运动历史用于恢复进度判断。相比之下，各组NF1均超过99%，运动历史对Need判断影响很小。因此，最终方法仅在End中保留H100，用于结合计划程度判断恢复进度；Need与Failure共同采用不含运动历史的输入。

<!-- 接口与结果核对：本节表4、表5及附录表A5、表A6来自Need、Failure、End统一读取历史的既有版本，用于输入设计分析。其结果不表示已完成Need、Failure移除历史后的重训或真机复测。主任务、异步和反事实实验的已有统计也不因论文接口更新而自动变为新接口结果；定稿时核对最终评估版本。 -->

我们进一步在Black-box上对每种设置进行20次任务试验，按恢复方向与程度统计水平位移。如图所示，虚线分别表示moderately和slightly的示教平均位移50 mm和20 mm。H0在两类左移计划中均未能正常停止，反映出缺少运动历史时恢复结束判断的不稳定性；H30的slightly位移明显小于示教均值，而H60的moderately位移偏大，右移达到174.4±22.1 mm。本质是因为窗口太小无法涵盖整个移动过程，从而导致模型无法收敛。相比之下，H100在四种计划下的平均位移均最接近对应的示教均值。这些结果支持利用运动历史评估恢复进度，使计划程度对应的实际位移更接近示教行为。

**表5．不同运动历史下的水平恢复位移（mm）。有标准差的记录按均值±标准差报告；“未停下”表示运动过长而未能正常结束，不表示实际位移为无穷大。H100复用表3异步组。**

| 恢复计划 | H0 | H30 | H60 | H100参考 |
| --- | --- | --- | --- | --- |
| 左 moderately | 未停下 | 93.1±11.4 | 108.5±18.9 | 48.4±4.3 |
| 右 moderately | 19.2±5.3 | 61.4±9.3 | 174.4±22.1 | 53.4±8.6 |
| 左 slightly | 未停下 | 2.1±1.2 | 11.9±3.1 | 17.1±2.5 |
| 右 slightly | 15.1±13.7 | 7.4±3.3 | 25.7±17.1 | 22.6±3.1 |

## 4.7 反事实监督对恢复程度执行的作用

本文的方法通过仅改变end训练数据的输入plan程度以及对应end label来让模型学会不同程度的运动距离差别。为检验程度反事实监督的作用，我们比较有、无该监督的训练版本。反事实样本保持图像、触觉、机器人状态与运动历史不变，仅替换计划中的程度词并调整完成标签，使同一状态在slightly下可以已完成、在moderately下仍需继续。两组分别在Black-box上进行20次任务试验，按恢复方向与程度统计实际位移。

如图所示，虚线分别表示slightly与moderately的示教平均位移20 mm和50 mm，两种程度相差30 mm。无反事实监督时，左、右方向的程度间距仅为10.0 mm和7.4 mm，moderately的平均位移分别为30.2 mm和36.8 mm，未能充分体现示教中的幅度差异。加入反事实监督后，程度间距分别扩大至31.3 mm和30.8 mm，接近示教中的30 mm间距；moderately的平均位移达到48.4 mm和53.4 mm，slightly则分别为17.1 mm和22.6 mm。该结果表明，在相同物理状态下提供随计划程度变化的结束标签，有助于End建立程度词与停止时机之间的联系，使两种语言程度在实际执行中得到更明确的区分。

![反事实监督对恢复程度执行的影响](试验结果/反事实监督_方案二_程度间距图.png)

*图．反事实监督对恢复程度执行的影响。CF表示反事实监督，圆点与误差棒表示位移均值及标准差，连线连接同一方向、同一训练设置下两种程度的均值；右侧Gap为两组均值之差。虚线表示示教平均位移20 mm与50 mm，浅绿色行对应有反事实监督的版本。*

**表6．反事实监督与恢复位移（mm，均值±标准差）。程度间距为两组均值之差，不是逐次配对距离差。有反事实组复用表3异步组，即表5的H100参考。**

| 训练设置 | 方向 | slightly | moderately | 程度均值间距 |
| --- | --- | --- | --- | --- |
| 无反事实监督 | 左 | 20.2±0.2 | 30.2±5.1 | 10.0 |
| 有反事实监督 | 左 | 17.1±2.5 | 48.4±4.3 | **31.3** |
| 无反事实监督 | 右 | 29.4±1.9 | 36.8±7.3 | 7.4 |
| 有反事实监督 | 右 | 22.6±3.1 | 53.4±8.6 | **30.8** |

图中的20 mm和50 mm是示教数据的统计均值，用于比较语言程度对应的执行行为；训练标签仍采用人工程度标注。表3异步组、表5的H100参考与本实验有反事实组来自同一组20次Black-box试验，在三个对照中共享参考数据。


## 英文 LaTeX：4.1 实验设置

```latex
\section{Experiments}
\label{sec:experiments}

\subsection{Experimental Setup}
\label{sec:experimental_setup}

\noindent\textbf{Platform and tasks.}
We use a Piper arm with a two-finger gripper and Xense tactile sensors on both fingertips. Visual observations come from a fixed front camera and a wrist-mounted camera. Platform details and an overview image are provided in the appendix. Figure~\ref{fig:task_objects} shows the task objects and configurations, while Fig.~\ref{fig:task_recovery} illustrates failure recovery and task resumption.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{tools.png}
    \caption{Task objects and configurations. From left to right: Black-box uses an internal movable weight to vary its center of mass; Book varies the number of neighboring books to create different insertion constraints; Bottle varies the remaining opening angle from $0^\circ$ to $180^\circ$; Sponge uses a paper disk to occlude obstacles. The wrist and front views on the right show that the obstacle is not directly visible when motion is blocked.}
    \label{fig:task_objects}
\end{figure*}

\noindent\textbf{Black-box: transfer with an unknown center of mass.}
The instruction is \emph{Pick up and transfer the object stably.}
We relocate an internal iron weight to vary the center of mass without changing the box's appearance. An unsuitable grasp may cause tilting during lifting. Recovery requires changing the grasp location and regrasping, using global failure history to reduce the correction magnitude when necessary, before resuming transfer. The torque traces at the top of Fig.~\ref{fig:task_recovery} show the left and right sensors' $T_z$ responses during failure and recovery.

\noindent\textbf{Book: insertion under varying constraints.}
The instruction is \emph{Pick up the book and place it horizontally on the bookshelf.}
We vary the number of books on either side of the insertion location to create different lateral constraints and jamming conditions. Upon detecting resistance, the robot must infer the direction of the contact error, reposition the book laterally to release the jam, and continue insertion.

\noindent\textbf{Bottle: opening with an unknown remaining angle.}
The instruction is \emph{Open the bottle}.
Each twist is limited to $90^\circ$, while the remaining opening angle is randomly set within $[0^\circ,180^\circ]$ and is unknown to the robot. A partially attached cap resists upward lifting, although this attachment state is not directly observable from images. The robot must detect this resistance, lower the gripper, continue unscrewing, and retry lifting until the cap is removed.

\noindent\textbf{Sponge: object transfer under visual occlusion.}
The instruction is \emph{Put the black sponge into the black frame.}
We attach a paper disk to the grasped sponge to occlude obstacles along its path. The two camera views in Fig.~\ref{fig:task_objects} show that neither camera directly observes the obstacle when motion is blocked. The robot must use tactile feedback to detect resistance, retreat to release contact, raise the sponge to clear the obstacle, and resume transfer.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{tasks.png}
    \caption{Representative recovery sequences for Black-box, Book, Bottle, and Sponge, from top to bottom. Gray, red, and green indicate execution, failure, and recovery, respectively. Need and End mark recovery initiation and completion of the current recovery operation. Relevant failure and plan fields are shown at failure detection. The torque traces above correspond to the Black-box example.}
    \label{fig:task_recovery}
\end{figure*}

\noindent\textbf{Evaluation metrics.}
Task Success Rate (TSR) is the fraction of trials completing the instructed goal. Need Accuracy (NA) is defined as the number of actual failure events correctly detected by Need divided by the total number of actual failures. A Need activation without an actual failure is counted as a false positive, i.e., a detection error, and not as a correctly detected failure. Trials may contain no failures or multiple failures; we therefore pool actual failure events across trials to compute Recovery Success Rate (RSR), the fraction resolved sufficiently to resume the task. Its denominator includes undetected failures and those that do not trigger recovery. Plan Accuracy (PA) is the fraction of generated plans judged correct by a human, including plans from unsuccessful trials. NA in Table~\ref{tab:real_world_results} measures detection over actual failures; false activations are analyzed qualitatively.
```

## 英文 LaTeX：4.2 真机任务表现

```latex
\subsection{Real-World Task Performance}
\label{sec:real_world_performance}

\noindent\textbf{Evaluation protocol.}
Each method is evaluated in 20 trials per task. A trial succeeds if the instructed goal is completed within a budget of ten recovery operations. The baseline, denoted \emph{w/o tactile}, is retrained after removing tactile descriptions from the inputs to Need, End, and Failure. All other inputs, training settings, and deployment settings are kept identical to TARO.

\noindent\textbf{Task completion.}
As shown in Table~\ref{tab:real_world_results}, TARO outperforms the baseline on all four tasks, increasing mean TSR from 38.75\% to 85.00\%, a gain of 46.25 percentage points. Black-box improves from 3/20 to 18/20 successful trials, while Book, Sponge, and Bottle improve by 35, 40, and 35 percentage points, respectively.

% Add \usepackage{booktabs,multirow} to the preamble.
\begin{table*}[t]
    \centering
    \small
    \setlength{\tabcolsep}{2.5pt}
    \renewcommand{\arraystretch}{1.1}
    \caption{Task outcomes by condition and recovery performance. C1--C5 denote task-specific conditions defined in the appendix. Condition entries report successful/failed trial counts (S/F); dashes indicate inapplicable conditions. TSR, NA, RSR, and PA are percentages, with numerator/denominator counts in parentheses. Overall TSR is averaged across tasks; NA and RSR pool actual failure events, while PA pools plan counts.}
    \label{tab:real_world_results}
    \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccccccccc@{}}
        \toprule
        \multirow[c]{2}{*}{Task} &
        \multirow[c]{2}{*}{Method} &
        \multicolumn{5}{c}{Condition outcomes (S/F)} &
        \multirow[c]{2}{*}{TSR $\uparrow$} &
        \multirow[c]{2}{*}{NA $\uparrow$} &
        \multirow[c]{2}{*}{RSR $\uparrow$} &
        \multirow[c]{2}{*}{PA $\uparrow$} \\
        \cmidrule(lr){3-7}
         & & C1 & C2 & C3 & C4 & C5 & & & & \\
        \midrule
        Black-box & w/o tactile & 2/2 & 0/4 & 1/3 & 0/4 & 0/4 & 15.00 (3/20) & 81.25 (13/16) & 18.75 (3/16) & 46.15 (6/13) \\
         & TARO & 3/1 & 4/0 & 3/1 & 4/0 & 4/0 & \textbf{90.00 (18/20)} & \textbf{97.50 (39/40)} & \textbf{95.00 (38/40)} & \textbf{97.44 (38/39)} \\
        \midrule
        Book & w/o tactile & 2/3 & 3/2 & 1/4 & 3/2 & -- & 45.00 (9/20) & 47.37 (9/19) & 42.11 (8/19) & 50.00 (8/16) \\
         & TARO & 4/1 & 4/1 & 5/0 & 3/2 & -- & \textbf{80.00 (16/20)} & \textbf{93.33 (14/15)} & \textbf{73.33 (11/15)} & \textbf{93.33 (14/15)} \\
        \midrule
        Sponge & w/o tactile & 1/4 & 4/1 & 2/3 & 1/4 & -- & 40.00 (8/20) & 40.00 (8/20) & 40.00 (8/20) & 53.33 (8/15) \\
         & TARO & 3/2 & 5/0 & 4/1 & 4/1 & -- & \textbf{80.00 (16/20)} & \textbf{95.00 (19/20)} & \textbf{80.00 (16/20)} & \textbf{100.00 (19/19)} \\
        \midrule
        Bottle & w/o tactile & 6/4 & 5/5 & -- & -- & -- & 55.00 (11/20) & 47.06 (8/17) & 47.06 (8/17) & 53.33 (8/15) \\
         & TARO & 10/0 & 8/2 & -- & -- & -- & \textbf{90.00 (18/20)} & \textbf{80.00 (8/10)} & \textbf{80.00 (8/10)} & \textbf{100.00 (8/8)} \\
        \midrule
        Overall & w/o tactile & -- & -- & -- & -- & -- & 38.75 (31/80) & 52.78 (38/72) & 37.50 (27/72) & 50.85 (30/59) \\
         & TARO & -- & -- & -- & -- & -- & \textbf{85.00 (68/80)} & \textbf{94.12 (80/85)} & \textbf{85.88 (73/85)} & \textbf{97.53 (79/81)} \\
        \bottomrule
    \end{tabular*}
\end{table*}

\noindent\textbf{Failure detection.}
Across tasks, NA improves from 52.78\% to 94.12\%. The gains are particularly pronounced when the contact state is difficult to determine visually: NA increases from 40.00\% to 95.00\% in Sponge and from 47.06\% to 80.00\% in Bottle. The obstacle is occluded in Sponge, while the cap's attachment state is not directly visible in Bottle.

Our trial observations further illustrate these ambiguities. In Bottle, the baseline lifted before sufficient unscrewing or continued rotating after the required opening angle had been reached, causing task failures. In Sponge, it triggered Need at a fixed location even when no obstacle was present, suggesting reliance on a learned recovery location. Tactile feedback helps ground phase assessment in the actual contact state.

\noindent\textbf{Failure interpretation and recovery planning.}
Tactile evidence also informs Plan through the global failure history. Removing tactile input from Failure reduces the contact information available in this history, and interpretation errors may propagate to subsequent planning. Overall PA increases from 50.85\% to 97.53\%; on both Sponge and Bottle, TARO reaches 100\%, compared with 53.33\% for the baseline. These results support the role of tactile feedback in recovery planning through the global failure history.

\noindent\textbf{Recovery success.}
Pooling all actual failure events, RSR increases from 37.50\% to 85.88\%. On Black-box, the baseline achieves 81.25\% NA but only 18.75\% RSR, whereas TARO reaches 95.00\% RSR. This gap highlights the importance of reliable failure interpretation, recovery planning, and completion assessment after an anomaly has been detected.
```

## 英文 LaTeX：4.3 全局故障历史对恢复规划的作用

```latex
\subsection{Effect of Global Failure History on Recovery Planning}
\label{sec:failure_history_ablation}

We investigate the value of interpreting local tactile descriptions as global failures with a VLM, and the role of history in selecting recovery magnitudes. The same local tactile description may correspond to different global failures under different gripper orientations, making the spatial meaning of past feedback ambiguous. Recovery magnitude also depends on earlier corrections: when an object with an unknown center of mass tilts in the opposite direction after regrasping, the previous correction may have overshot, calling for a smaller subsequent adjustment. We therefore construct history-conditioned planning data for Black-box to evaluate recovery plans based on global failure histories.

\noindent\textbf{Comparison.}
We compare three settings: \emph{w/o history} predicts a plan from the common inputs alone; \emph{tac-history} additionally receives a history of local tactile descriptions; and \emph{TARO} uses a global failure history. All groups share the same common inputs, with robot state masked to suppress posture cues. For this ablation, TARO receives manually annotated global failures. Each model is initialized from the same Stage A checkpoint, and only the shared PaliGemma LoRA parameters are trained for planning.

We collect 40 rotational failure clips, evenly split between left and right failures, and sample 30 consecutive observations per clip. Base plans are manually annotated, and histories are constructed offline using rules. Historical grasp orientations are randomized to simulate the same local tactile description corresponding to different global failures. The test set contains 360 samples from four held-out clips; additional details are provided in the appendix.

\noindent\textbf{Metrics.}
Direction Accuracy (DA) and Magnitude Accuracy (MA) measure the correctness of the horizontal recovery direction and magnitude, respectively. Joint Accuracy (JA) requires both to be correct. Multi-round Joint Accuracy (MJA) computes JA over 240 samples with prior feedback, while Reversal Joint Accuracy (RJA) computes JA over 120 samples whose global failure direction has reversed at least once.

% Requires \usepackage{booktabs}.
\begin{table}[t]
    \centering
    \small
    \setlength{\tabcolsep}{2pt}
    \caption{Effect of global failure history on offline recovery planning. Results are percentages on 360 test samples; MJA and RJA are evaluated on subsets of 240 and 120 samples, respectively. TARO uses annotated global failures.}
    \label{tab:history_planning}
    \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccccc@{}}
        \toprule
        Method & DA $\uparrow$ & MA $\uparrow$ & JA $\uparrow$ & MJA $\uparrow$ & RJA $\uparrow$ \\
        \midrule
        w/o history & 80.83 & 66.67 & 53.89 & 40.42 & 0.00 \\
        tac-history & 100.00 & 66.67 & 66.67 & 50.00 & 0.00 \\
        TARO & 100.00 & \textbf{100.00} & \textbf{100.00} & \textbf{100.00} & \textbf{100.00} \\
        \bottomrule
    \end{tabular*}
\end{table}

\noindent\textbf{Global direction.}
As shown in Table~\ref{tab:history_planning}, \emph{w/o history} achieves 80.83\% DA using the current visual observation, whereas \emph{tac-history} and TARO both reach 100\%. This suggests that tactile-related feedback helps the model combine visual and contact evidence to reason about the global direction of a contact failure and select a recovery direction.

\noindent\textbf{Role of history.}
Without past failure feedback, \emph{w/o history} always predicts \emph{moderately} and achieves 0\% RJA on samples that require a smaller correction after reversal. For the same current observation, whether an earlier correction overshot depends on preceding feedback; selecting the recovery magnitude therefore requires history.

\noindent\textbf{Local and global histories.}
Although \emph{tac-history} retains local tactile feedback, it also always predicts \emph{moderately} and obtains 0\% RJA. Because local descriptions are expressed in the sensor frame, differing historical grasp orientations make it difficult to infer whether the global failure direction has reversed. TARO uses global failure history in a consistent global frame and selects \emph{slightly} after reversal, achieving 100\% MJA and RJA. With annotated global failures, this comparison evaluates the value of the failure representation and its history for planning.
```

## 英文 LaTeX：4.5 异步判断对恢复执行的影响

```latex
\subsection{Effect of Asynchronous Assessment on Recovery Execution}
\label{sec:async_assessment}

To better exploit high-frequency tactile signals, we run the Need and End assessments asynchronously with physical execution, allowing the robot to detect failures and control recovery motion promptly. We evaluate synchronous and asynchronous assessment on Black-box under five center-of-mass configurations, with 20 task trials per method. The synchronous method waits for assessment after each action chunk, whereas the asynchronous method evaluates contact during physical execution and interrupts the current action when a phase transition is triggered. We compare lift height and horizontal recovery displacement to assess the effect of the two scheduling methods on robot motion.

\begin{table}[t]
    \centering
    \small
    \setlength{\tabcolsep}{3pt}
    \caption{Effect of asynchronous assessment on motion in Black-box. Values are mean $\pm$ standard deviation in mm, grouped by motion type over 20 task trials per method.}
    \label{tab:async_motion}
    \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcc@{}}
        \toprule
        Motion & Synchronous & Asynchronous \\
        \midrule
        Lift height & $41.2 \pm 38.5$ & $39.3 \pm 20.9$ \\
        Left, moderately & $86.1 \pm 29.3$ & $48.4 \pm 4.3$ \\
        Left, slightly & $36.9 \pm 15.6$ & $17.1 \pm 2.5$ \\
        Right, moderately & $67.3 \pm 4.8$ & $53.4 \pm 8.6$ \\
        Right, slightly & $39.1 \pm 11.1$ & $22.6 \pm 3.1$ \\
        \bottomrule
    \end{tabular*}
\end{table}

As shown in Table~\ref{tab:async_motion}, asynchronous assessment reduces the mean displacement in all four horizontal recovery conditions. The mean lift height decreases by 1.9 mm, while the mean displacements for moderately sized leftward and rightward movements decrease by 43.79\% and 20.65\%, respectively. The resulting motion distances are closer to those in the demonstrations. Excessive motion under synchronous assessment can also cause the gripper to miss the appropriate grasp location, leading to lower task success than asynchronous assessment.
```

<!-- 待核对：示教位移基准及同步、异步方法的任务成功率具体统计尚未在现有实验结果记录中找到。 -->

## 英文 LaTeX：4.6 运动历史对恢复进度判断的作用

```latex
\subsection{Effect of Motion History on Recovery Progress Assessment}
\label{sec:motion_history_ablation}

The current robot state alone does not capture how much recovery motion has been executed, yet this information is needed to determine whether the planned magnitude has been reached. To examine the role of motion history and identify which branches benefit from it, we compare four windows: H0, H30, H60, and H100. In this input-design ablation, Need, Failure, and End all receive the corresponding motion history. H0 repeats the current robot state 11 times; H30 and H60 sample 11 states from the latest 31 and 61 frames, including the current frame; H100 samples 11 states from the latest 100 frames. All settings retain the current robot state and share the remaining inputs.

The H0, H30, and H60 models are trained separately using the same data split, supervision, and initialization as H100. Table~\ref{tab:motion_history_validation} reports validation results using Need macro-F1 (NF1), End accuracy (EA), and End macro-F1 (EF1). Macro-F1 averages the F1 scores of the positive and negative classes, while EA measures accuracy over all validation samples. We also report the false-positive rate (FPR) on factual samples: the fraction of samples labeled as incomplete that are incorrectly predicted as complete.

\begin{table}[t]
    \centering
    \small
    \setlength{\tabcolsep}{2.5pt}
    \caption{Validation results with different motion history windows. All metrics are reported in \%. H100 is the reference model.}
    \label{tab:motion_history_validation}
    \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccc@{}}
        \toprule
        Window & NF1 $\uparrow$ & EA $\uparrow$ & EF1 $\uparrow$ & FPR $\downarrow$ \\
        \midrule
        H0 & 99.40 & 81.25 & 76.51 & 22.22 \\
        H30 & 99.52 & 88.80 & 84.72 & 14.44 \\
        H60 & 99.52 & 90.51 & 86.68 & 9.86 \\
        H100 (ref.) & 99.27 & 93.46 & 90.90 & 6.81 \\
        \bottomrule
    \end{tabular*}
\end{table}

From H0 to H100, EA increases from 81.25\% to 93.46\% and EF1 from 76.51\% to 90.90\%, gains of 12.21 and 14.39 percentage points, respectively; FPR decreases from 22.22\% to 6.81\%. H30 and H60 yield intermediate results, supporting the use of motion history for recovery progress assessment. In contrast, NF1 exceeds 99\% in every setting, showing little sensitivity to the history window. Accordingly, the final design supplies H100 only to End, while Need and Failure share an input without motion history.

We further evaluate each setting over 20 Black-box task trials, grouping horizontal recovery displacements by direction and magnitude. In Fig.~\ref{fig:motion_history_displacement}, dashed lines indicate mean demonstration displacements of 50 mm for \emph{moderately} and 20 mm for \emph{slightly}. H0 exhibits non-termination in leftward recovery for both magnitudes, revealing unreliable stopping without motion history. H30 produces \emph{slightly} displacements well below the demonstration mean, whereas H60 produces excessive \emph{moderately} displacements, reaching $174.4 \pm 22.1$ mm rightward. One possible explanation is that shorter windows omit part of the recovery motion, limiting progress assessment. H100 yields mean displacements closest to the corresponding demonstration means in all four plan conditions. These results support using motion history to assess recovery progress and better reproduce the demonstrated motion magnitudes.

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{试验结果/运动历史与恢复位移_单栏.png}
    \caption{Effect of motion history on horizontal recovery displacement in Black-box. Bars show means and error bars indicate standard deviations. Dashed lines denote the mean demonstration displacement for each magnitude. \emph{No stop} denotes a recovery operation that did not terminate normally. The panels use different y-axis scales. H100 reuses the asynchronous reference results.}
    \label{fig:motion_history_displacement}
\end{figure}
```

## 英文 LaTeX：4.7 反事实监督对恢复程度执行的作用

```latex
\subsection{Effect of Counterfactual Supervision on Recovery Magnitude Control}
\label{sec:counterfactual_magnitude}

We train End to distinguish recovery magnitudes through counterfactual examples that change only the magnitude word in the input plan and its corresponding completion label. Images, tactile descriptions, robot state, and motion history remain unchanged, so the same physical state can be complete under \emph{slightly} but still require further motion under \emph{moderately}. To evaluate this supervision, we compare models trained with and without it over 20 Black-box task trials per setting, grouping actual recovery displacements by direction and magnitude.

As shown in Fig.~\ref{fig:counterfactual_magnitude}, the dashed lines mark the mean demonstration displacements of 20 mm for \emph{slightly} and 50 mm for \emph{moderately}, a gap of 30 mm. Without counterfactual supervision, the mean displacement gaps are only 10.0 mm leftward and 7.4 mm rightward. The corresponding \emph{moderately} displacements are 30.2 mm and 36.8 mm, failing to reproduce the magnitude separation in the demonstrations. With counterfactual supervision, the gaps increase to 31.3 mm and 30.8 mm, close to the demonstrated gap of 30 mm. Mean \emph{moderately} displacements reach 48.4 mm and 53.4 mm, while \emph{slightly} displacements are 17.1 mm and 22.6 mm, respectively. These results suggest that assigning magnitude-dependent completion labels to the same physical state helps End associate magnitude words with stopping decisions, producing a clearer distinction between the two magnitudes during execution.

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{试验结果/反事实监督_方案二_程度间距图.png}
    \caption{Effect of counterfactual supervision (CF) on recovery magnitude control. Points and error bars indicate displacement means and standard deviations. Each line connects the means for the two magnitudes under the same direction and training setting; \emph{Gap} denotes their difference. Dashed lines mark demonstration means of 20 mm and 50 mm. Shaded rows indicate training with CF.}
    \label{fig:counterfactual_magnitude}
\end{figure}

The demonstration means provide empirical references for comparing magnitude-dependent behavior; training still uses manually annotated magnitude labels. The supervised setting reuses the same 20 Black-box task trials as the asynchronous reference in Table~\ref{tab:async_motion} and H100 in Fig.~\ref{fig:motion_history_displacement}.
```

# 附录

## A. 实验设置与复现细节

### A.1 平台、采集与推理配置

实验采用Piper单臂，夹爪两指安装千觉Xense触觉传感器，视觉输入来自固定主摄和腕部相机。示教通过主从臂遥操作采集，记录两路图像、六个机械臂关节与夹爪开合组成的七维状态，以及双指触觉信号。数据采集与动作执行频率均为30 Hz，在线Need/End状态判断约以3 Hz运行。动作块长度为30，流匹配推理使用10步积分。

<!-- 附录图A1待插入：平台整体照片，标示Piper机械臂、双指夹爪及Xense触觉传感器、固定主摄和腕部相机的位置。平台图单独放在本节；正文图2使用tools.png展示任务道具，不再与平台照片合并。 -->

### A.2 示教规模、数据划分与训练

四项操作任务分别组织示教并训练模型，数据规模与划分见表A1。划分以原始回合为单位，同一回合中的失败与恢复片段归入同一集合；派生历史与反事实样本继承对应原始回合的划分。表中的离线测试回合与每种方法、每项任务20次的最终真机测试分别统计。

**表A1．各任务的示教规模与回合划分。**

| 任务 | 原始回合数 | 训练回合 | 验证回合 | 离线测试回合 |
| --- | --- | --- | --- | --- |
| Black-box | 90 | 72 | 9 | 9 |
| Book | 90 | 72 | 9 | 9 |
| Sponge | 40 | 32 | 4 | 4 |
| Bottle | 40 | 32 | 4 | 4 |

各模型先学习正常执行与恢复动作，再联合训练阶段判断、故障诊断、恢复规划与动作保持。主实验部署采用Stage B第2000步权重；专项消融采用对应实验说明的训练与选点协议，保留与主实验部署选点的区别。训练方法与优化目标见方法部分。

完整方法与无触觉版本复用对应任务的同一个Stage A动作模型。无触觉版本移除Need、End与Failure输入中的触觉描述后，独立重新训练Stage B；其他输入、数据划分、训练与部署设置以及恢复流程均与完整方法保持一致。该对照评价触觉信息在恢复框架中的作用。

### A.3 真机任务与条件分配

<!-- 原始任务指令已逐字核对四项主实验数据的meta/tasks.jsonl，并检查对应Stage B的全部Failure/Plan清单。Black-box为 Pick up and transfer the object stably.；Book为 Pick up the book and place it horizontally on the bookshelf.；Sponge为 Put the black sponge into the black frame.；Bottle为 Open the bottle（原始指令不带句号）。正文中文是对应译文；英文LaTeX保留原始大小写、措辞和句内标点。 -->

正文表1的C1—C5在每个任务内独立编号，不表示不同任务采用相同物理条件。每种方法在每项任务上测试20次，两种方法共160次真机测试。两种方法采用相同的条件分配，编号映射与每个条件的重复次数见表A2。

**表A2．正文分条件结果的设置说明。“—”表示该任务不包含对应条件。**

| 任务 | C1 | C2 | C3 | C4 | C5 | 每个条件测试次数 |
| --- | --- | --- | --- | --- | --- | --- |
| Black-box | 重心位置1 | 重心位置2 | 重心位置3 | 重心位置4 | 重心位置5 | 4 |
| Book | 左侧3本 | 左侧2本 | 右侧3本 | 右侧2本 | — | 5 |
| Sponge | 物块位置1 | 物块位置2 | 物块位置3 | 物块位置4 | — | 5 |
| Bottle | 剩余旋开角度90° | 剩余旋开角度180° | — | — | — | 10 |

**Black-box。**通过移动黑盒内部铁块设置五种重心位置，保持盒体外观不变。C1—C5依次对应原始试验记录中的五种重心配置，每种配置重复4次，以评价未知重心条件下稳定抓取与转移的表现。

**Book。**通过改变目标插入位置两侧的相邻书籍数量形成不同的侧向约束。“左侧3本”和“左侧2本”分别对应左侧相邻书籍为3本与2本，“右侧3本”和“右侧2本”同理。四种条件分别测试5次，以比较不同卡滞条件下的位置恢复能力。

**Sponge。**C1—C4对应原始试验记录的四种物块位置，每种位置重复5次。使用正文图2所示的纸盘遮挡结构，使腕部与主摄均无法直接观察运动受阻时的障碍物。

**Bottle。**单次旋拧不超过90°，任务设计允许瓶盖剩余旋开角度在0°—180°范围内变化。现有成功率记录采用90°与180°两组条件，分别对应C1与C2，每组10次；正文表1按该原始分组报告结果。

<!-- 配置细节核对：Black-box五种配重位置和Sponge四种物块位置的具体坐标或逐条件照片尚未提供，不补写未测量的位置值；Book另一侧的具体书籍数及间距也不额外推定。Bottle正文任务描述为0°—180°随机设置，当前结果记录为90°/180°两组，二者的具体对应仍需在定稿前核对，不改写已有成功/失败计数。 -->

每次测试最多允许10次恢复，完成一个恢复阶段计为一次；同一故障下完成多个恢复阶段分别计次。任务成功以机器人在该预算内完成原任务目标为准，由人工判定。正文条件单元格的失败数为该条件测试次数减去成功数，因此S/F的两项相加分别为4、5或10。合并四项任务后，w/o tactile为31次成功、49次失败，TARO为68次成功、12次失败。

### A.4 指标定义与计数规则

**表A3．主实验指标及统计单位。**

| 指标 | 计算方式 | 统计单位 |
| --- | --- | --- |
| 任务成功率（TSR） | 完成原任务的测试次数/总测试次数 | 任务回合 |
| Need Accuracy（NA） | Need正确检出的真实故障次数/实际故障次数 | 真实故障事件 |
| 恢复成功率（RSR） | 被成功纠正并能继续原任务的故障次数/实际故障次数 | 故障事件 |
| 计划正确率（PA） | 人工判定正确的计划数/全部计划数 | 每次计划输出 |

实际故障总数包含未检出或未启动恢复的故障。同一故障的连续多轮恢复仅计一次故障，恢复后出现新故障则另计。恢复成功与最终任务成功分别评价：某次故障被纠正并能继续原任务，不要求该任务后续一定成功。计划逐条评定，包含最终失败回合中的全部输出。不同策略可能产生不同的后续交互，故障事件不视为两种方法之间逐一配对的相同事件。

实际未发生故障但Need判断需要恢复时，每次触发记为一次误检（检测失败），不计为一次真实故障或一次正确检出。NA按本文约定表示真实故障事件上的检出比例，数值口径对应召回率；误检次数单独记录。现有结果汇总未提供各组误检总数，因此正文对误触发现象作定性分析，表1的NA不用于衡量无故障状态上的判断准确性。

部分测试全程不发生故障，部分测试包含多个故障，因此故障后恢复成功率合并所有测试中的真实故障事件计算。正文总体行的TSR为四项任务TSR的算术平均；由于各任务测试次数相同，该值也等于合并任务成功次数后的比例。其余过程指标先合并各任务的分子与分母，再计算比例。

专项规划消融分别报告方向准确率（DA）、程度准确率（MA）及二者同时正确的联合准确率（JA）；多轮与反向相关样本上的JA分别记为MJA与RJA。状态判断采用准确率、Macro-F1及恢复结束误报率；误报结束率为不应结束的负例被预测为完成的比例。真机位移统一以mm报告，有标准差的记录采用均值±标准差。位移按任务中出现的方向与程度分组，每组20次指任务试验次数，不表示每一方向与程度各进行20次。


## B. 历史表示消融的训练与子集结果

原始实验记录中的A、B、C组分别对应正文的w/o history、tac-history与TARO。w/o history仅输入当前通用观测，tac-history使用局部触觉历史，TARO使用标注的全局Failure历史；三组均预测Plan。

三组学习率均为1e-4，weight decay为1e-4，batch size为8，随机种子为42，采用FSDP单设备配置。初始化为Black-box Stage A第10000步权重，训练400步，每200步验证。按验证集的回合宏平均方向与程度联合准确率选取checkpoint，持平保留较早模型。w/o history选择400步，tac-history与TARO选择200步，固定后在相同测试样本上评估。

w/o history的最新测试汇总中，三个episode的30个当前观测全部方向正确，另一个episode为7个正确、23个错误，共97/120个观测正确。三种历史变体对该组提供相同输入，因此方向正确数为291/360。程度始终为moderately，初始与未反向子集分别有97/120条方向、程度同时正确，反向子集全部程度错误；总体联合正确数为194/360，多轮联合正确数为97/240。

<!-- 更新来源：上述A组测试汇总由用户提供，取代旧test_best.metrics.json中的100%方向结果。当前未提供发生错误的episode编号或新逐样本评估文件；原始评估文件未修改，旧验证集记录保持原值。该重算沿用原测试划分、每观测三种历史及w/o history确定性输入不变的协议。 -->

**表A4．历史相关子集的方向与程度联合准确率（JA）。四行互斥，共360条样本。**

| 场景 | 样本数 | w/o history | tac-history | TARO |
| --- | --- | --- | --- | --- |
| 初始，无过去反馈 | 120 | 80.83% | 100.00% | 100.00% |
| 多轮，未发生全局反向 | 120 | 80.83% | 100.00% | 100.00% |
| 当前首次全局反向 | 76 | 0.00% | 0.00% | 100.00% |
| 此前已发生全局反向 | 44 | 0.00% | 0.00% | 100.00% |

## C. 运动历史的补充指标

本节报告Need、Failure与End统一读取运动历史的输入设计消融，作为正文4.6中精简输入的依据。H0、H30、H60组采用FSDP=1，H100参考采用FSDP=2。四列均为第4000步权重的验证指标，保留与主实验第2000步部署选点的区别。

**表A5．运动历史消融的事实样本判断及生成、动作指标。End正类表示当前恢复操作已完成。**

| 指标 | 0秒 | 约1秒 | 约2秒 | H100参考 |
| --- | --- | --- | --- | --- |
| 事实样本End正类精确率 | 37.74% | 49.27% | 59.66% | 74.87% |
| 事实样本End正类召回率 | 57.74% | 60.12% | 62.50% | 86.90% |
| Failure整条输出正确数 | 8/8 | 8/8 | 8/8 | 8/8 |
| Plan整条输出正确数 | 16/16 | 16/16 | 16/16 | 16/16 |
| 动作验证损失 | 0.007672 | 0.008250 | 0.007840 | 0.007644 |

各组生成结果在当前小规模评价上相同，主要差异来自End判断。事实样本中负例为720条、正例为168条，对应混淆矩阵如下。

**表A6．事实样本End判断的混淆计数。**

| 历史窗口 | TN | FP | FN | TP | 总数 |
| --- | --- | --- | --- | --- | --- |
| 0秒 | 560 | 160 | 71 | 97 | 888 |
| 约1秒 | 616 | 104 | 67 | 101 | 888 |
| 约2秒 | 649 | 71 | 63 | 105 | 888 |

补充部署日志覆盖0/1/2秒模型的2/3/2段记录。以下“完成恢复阶段”仅指End触发并返回正常执行，不表示任务最终成功，也不与正文每组20次任务试验的完整统计混用。

**表A7．局部部署日志中的恢复阶段记录。**

| 历史窗口 | 日志段数 | Need触发次数 | 完成/进入恢复阶段 | 已完成阶段中水平位移小于10 mm |
| --- | --- | --- | --- | --- |
| 0秒 | 2 | 7 | 5/7 | 2/5 |
| 约1秒 | 3 | 18 | 17/18 | 14/17 |
| 约2秒 | 2 | 13 | 12/12 | 1/12 |

<!-- 编辑核对：以下不属于论文正文，不将尚未核实的设置写成事实。
1. 数据依据：试验结果/实验章节数据核对.md 与任务成功率实验记录表.md；平台、示教规模及主实验2000步部署选点补充依据 method整理.md 的已确认记录。未重新运行训练或真机试验。
2. 相机型号、推理硬件、各组实际阈值与全部运行配置仍需补齐；End阈值校准方案尚未产生实测结果。
3. 已完成运行包含旧版Touch、Failure、Plan与配对记忆协议；论文版五字段Touch、统一Failure、四字段Plan及failure-only历史尚未核实已完整重训。本文不将已有表格称为新接口重训结果；投稿前需核对每张表的权重与文本协议，统一Method、Abstract和实验的版本身份。
4. 主实验只报告任务成功率、故障后恢复成功率、计划正确率与故障检出率；不恢复已取消的平均恢复轮数和主实验Failure准确率。附录Failure生成正确率属于运动历史验证协议。
5. 同步/异步只报告位移，不补写未执行的任务成功率对照。相同权重、阈值及控制参数的实际运行配置仍需对照日志核实。
6. 位移表不逐格标动作n。每组20次表示任务试验次数，不表示每一方向/程度各做20次。未确认的测量协议不统一改写为夹爪中心三维欧氏距离；0/1/2秒历史组已确认为水平净位移，其他表沿用原“恢复位移”名称，尚需确认轴向投影与停止位置采样规则。
7. 反事实真机结果来自任务中实际恢复片段，不写成直接给定Plan、逐次配对或仅替换End的已执行协议；完整训练与控制差异仍需补齐。因此正文采用“有助于”的描述，不将均值间距认定为统计显著或绝对位移精度。
8. 表3异步组、表5 H100和表6有反事实组复用同一参考数据，不累加为独立试验。H100训练配置与新增历史组不同，保留参考身份。
9. 2秒组局部日志Need触发13次、进入恢复12次的原因仍未核实；保留原计数，不由该表推导任务成功率。
10. Book左3/左2/右3/右2的具体摆放、Black-box重心位置和Sponge物块位置宜以实验场景图补充。触觉描述器与恢复动作泛化暂无结果，未写成完成实验。
-->
