# Introduction

## 中文

视觉–语言–动作（Vision–Language–Action, VLA）模型推动了通用机器人操作的发展，但物理交互的可靠性仍受视觉难以观测的接触状态制约。[Tactile-VLA](https://arxiv.org/abs/2507.09160) 将触觉融入VLA，结合位置–力混合控制与触觉反馈推理改善物理交互；[VLA-Touch](https://arxiv.org/abs/2507.17294) 则将触觉用于高层规划与动作精修。然而，真实环境中的物体偏移、夹取失衡或意外阻碍等OOD的故障可能使既定轨迹失效，仅精修原有动作未必能够完成任务。机器人还需要理解已经发生的接触故障，推理出新的恢复策略，并判断何时能够继续原任务。本文关注的核心问题是：**如何利用动态触觉理解接触故障，并根据全局故障历史实现在线多轮自主恢复？**

针对操作失败，[TACO](https://arxiv.org/abs/2607.02840) 从失败附近的状态生成视触觉纠正轨迹，用于策略后训练。其纠正过程由任务进度估计引导，尚未显式将故障原因转化为方向与程度明确的恢复目标。[PEARS](https://arxiv.org/abs/2610.08784) 进一步利用视觉语言模型分析失败，但是其推理内容仅限于接触力的大小。此外该框架在任务执行完毕后才推理，难以满足不可逆任务中的即时恢复需求。因此，机器人需要针对不同接触故障，推理涵盖位置、姿态与夹取状态的恢复策略，并在同一次任务执行中实现在线恢复。

为此，本文提出触觉驱动的在线故障恢复框架 **TARO**。我们设计轻量化时序触觉描述器，将连续双指触觉离散为接触面积、三轴受力与旋转五类元描述，标签统一以左指传感器局部坐标系为参考。我们利用视觉语言模型（VLM）的推理能力结合视觉与局部触觉描述，推理全局参考系下的接触故障；随后以固定窗口内的全局故障历史为依据，用自回归的方式生成下一步的恢复策略。此外，TARO 在 π0.5 的共享视觉语言主干上引入恢复触发与恢复结束两个独立的二分类头，分别判断是否启动恢复以及当前恢复操作是否完成。两个判断头在对应阶段与物理动作执行异步运行，驱动执行与恢复阶段的自主切换，形成任务执行与在线恢复的闭环

我们在未知重心物体转移、书籍插入、遮挡条件下的物块移动和未知旋开角度的瓶盖开启四项真机任务中评估 TARO。每种方法在每项任务上测试20次，相较于独立训练的无触觉恢复版本，平均任务成功率由38.75%提升至85.00%，提高46.25个百分点。受控离线消融进一步表明，标注的全局故障历史比局部触觉历史更有利于选择恢复程度；真机对照则检验了异步判断、运动历史与程度监督对实际恢复位移的影响。

在四项真机任务中，每种方法每项任务测试20次，TARO 相较于独立训练的无触觉恢复版本，将平均成功率由38.75%提升至85.00%。本文的主要贡献如下：

- **离散化的触觉元描述。** 提出轻量化时序触觉描述器，将连续双指触觉信号离散为具有物理含义的元描述文本，为故障诊断与恢复推理提供简洁的接触语义。

- **基于触觉的故障诊断和恢复推理方法。** 结合视觉与局部触觉理解全局空间中的接触故障，并根据全局故障历史推理后续恢复方向与程度。

- **执行与恢复的触觉闭环框架。** 通过恢复触发与恢复结束两个判断头，在动作执行期间异步评估恢复需求与进度，驱动执行与恢复阶段的自主切换，实现在线多轮恢复与原任务续执行。

## English

```latex
\section{Introduction}
\label{sec:introduction}

Vision-language-action (VLA) models have advanced general-purpose robotic manipulation, yet reliable physical interaction remains limited by contact states that are difficult to observe visually. Tactile-VLA~\cite{huang2025tactilevla} integrates tactile sensing with hybrid position--force control and tactile feedback reasoning, while VLA-Touch~\cite{bi2025vlatouch} uses touch for high-level planning and action refinement. However, under out-of-distribution (OOD) conditions, object displacement, unbalanced grasps, or unexpected obstructions can invalidate planned trajectories. Refining the original actions alone may then be insufficient: robots must understand contact failures, infer new recovery strategies, and determine when to resume the original task. This motivates our central question: \emph{How can robots use dynamic tactile feedback to understand contact failures and achieve online, multi-round recovery guided by global failure history?}

TACO~\cite{liu2026taco} addresses manipulation failures by generating visuo-tactile correction trajectories from failure-adjacent states for policy post-training. Its correction process relies on task progress estimates without explicitly grounding recovery directions and magnitudes in diagnosed failure causes. PEARS~\cite{song2026pears} further employs a vision-language model to analyze failures, but its failure reasoning targets contact-force magnitudes and is performed after each episode. This timing limits its support for immediate recovery in irreversible tasks. Robots therefore need recovery strategies that address position, orientation, and grasping state across different contact failures, together with the ability to recover online within the same task execution.

We present \textbf{TARO}, a tactile-driven framework for online failure recovery. We design a lightweight temporal tactile descriptor that discretizes continuous tactile sequences from both fingers into five fields describing contact area, three-axis forces, and rotation, with directional labels referenced to the left finger sensor's local coordinate frame. We leverage the fine-tuned vision-language backbone of $\pi_{0.5}$ to interpret contact failures in a global reference frame from visual observations and local tactile descriptions. Conditioned on a fixed-size window of global failure history, the model autoregressively generates the next recovery strategy. TARO further augments the shared vision-language backbone of $\pi_{0.5}$ with two independent binary classification heads that determine whether to initiate recovery and whether the current recovery operation is complete, respectively. In their respective phases, these heads operate asynchronously with physical action execution, enabling autonomous transitions between execution and recovery and closing the loop between task execution and online recovery.

We evaluate TARO on four real-world contact-rich manipulation tasks, with 20 trials per task per method. Compared with a separately trained recovery baseline without tactile input, TARO improves mean task success from 38.75\% to 85.00\%, a gain of 46.25 percentage points. Controlled offline ablations establish the utility of annotated global failure histories for selecting recovery magnitudes, while real-world comparisons examine how asynchronous assessment, motion history, and magnitude supervision affect recovery displacements.

Our main contributions are:
\begin{itemize}
    \item \textbf{Discrete tactile descriptions.} We introduce a lightweight temporal tactile descriptor that discretizes continuous tactile signals from both fingers into physically meaningful textual descriptions, providing concise contact semantics for failure diagnosis and recovery reasoning.
    \item \textbf{Tactile-based failure diagnosis and recovery reasoning.} We combine visual observations with local tactile descriptions to interpret contact failures in a global reference frame and infer subsequent recovery directions and magnitudes from the global failure history.
    \item \textbf{A tactile closed-loop framework for execution and recovery.} We introduce recovery-trigger and recovery-completion heads to asynchronously assess recovery needs and progress during action execution, enabling autonomous transitions between execution and recovery, online multi-round recovery, and resumption of the original task.
\end{itemize}
```

<!--
写作核对（不属于论文正文）：
1. 方法表述以 method.md 中已确认的论文版接口为准：plan不直接输入Touch、运动历史或历史plan；全局故障历史包括当前failure及最近最多4条过去failure。输入字段、分类头、文本格式和生成约束留在方法与实验设置中展开。
2. 主实验数值采用《试验结果/实验章节数据核对.md》：无触觉31/80，完整方法68/80；两组均保留恢复框架。不据此量化相对“不启用恢复”的收益。历史表示消融采用标注真值failure，仅支持受控离线规划结论；同步／异步实验只报告位移，不将位移缩短等同于绝对精度提升。
3. 五字段描述器的准确性和开销实验尚未补齐，正文不宣称已验证的覆盖率或速度优势。论文版接口与旧训练／部署产物的差异沿用 method.md 的记录，最终实验与接口版本需对应核实。
4. 相关工作链接已核对。TACO的比较针对进度引导的纠正轨迹生成；PEARS的比较针对PFR故障推理模块，不将其整个框架概括为只能调整力，也不声称其没有历史推理。排版时将链接替换为正式文献引用。
-->
