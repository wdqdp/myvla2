# 标题
TARO: Tactile Understanding and History-Guided Reasoning for Online Failure Recovery

# 模型名称
TARO，取自 **TA**ctile Understanding and History-Guided **R**easoning for **O**nline Failure Recovery。

# 中文
随着触觉在接触丰富操作中日益受到关注,很多工作利用触觉反馈精修既定动作。然而，真实环境中的突发扰动可能使既定轨迹失效。因此，机器人需要理解已发生的故障，并据此生成新的恢复策略。为此，本文提出触觉驱动的在线故障恢复框架TARO。我们设计轻量化触觉描述器，将连续时序触觉离散为以左指传感器局部坐标系为参考的元描述；在此基础上，我们利用微调后 π0.5 的视觉语言主干，结合视觉与局部触觉描述理解全局参考系下的接触故障，并根据全局故障历史生成恢复策略。此外，TARO通过 Need 和 End 两个判断头，在动作执行期间异步评估恢复需求与进度，实现执行与恢复阶段的自主切换和在线多轮恢复。在四项真实机器人接触操作任务中，本文方法相较于独立训练的无触觉恢复版本，将平均任务成功率由38.75%提升至85.00%，提高46.25个百分点。

# 英文

```latex
\begin{abstract}
Tactile feedback is increasingly used to refine intended actions in contact-rich manipulation. However, unexpected disturbances in real-world environments can invalidate planned trajectories, requiring robots to understand contact failures and devise new recovery strategies. We present TARO, a tactile-driven framework for online failure recovery. A lightweight temporal tactile descriptor discretizes continuous tactile sequences into textual contact descriptions expressed in the left finger sensor's local coordinate frame. We leverage the fine-tuned vision-language backbone of $\pi_{0.5}$ to interpret contact failures in a global reference frame by integrating visual observations with local tactile descriptions. It then reasons over the global failure history to infer recovery strategies. Two binary heads, Need and End, run asynchronously during physical action execution to determine whether to initiate recovery and whether the recovery is complete, respectively. This enables autonomous transitions between execution and recovery and supports online multi-round recovery. Across four real-world contact-rich manipulation tasks, TARO improves mean task success from 38.75\% to 85.00\% compared with a separately trained recovery baseline without tactile input, a gain of 46.25 percentage points.
\end{abstract}
```
