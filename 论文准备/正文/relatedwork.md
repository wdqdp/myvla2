# Related Work

**触觉融合与接触语义。** 触觉用于补充视觉难以观测的接触信息，并逐步从局部控制进入高层决策：[Tactile-VLA](https://arxiv.org/abs/2507.09160) 结合触觉推理与位置–力控制，[VLA-Touch](https://arxiv.org/abs/2507.17294) 则利用触觉语义支持规划与动作精修。这一过程依赖对传感信号的语义理解。[TVL](https://arxiv.org/abs/2402.13232) 通过视触觉–语言对齐描述物体触感，[CLTP](https://arxiv.org/abs/2505.08194) 进一步关注传感器视角下的接触几何与受力，[AnyTouch 2](https://arxiv.org/abs/2602.09617) 则扩展到时序形变与力感知动态。本文采用轻量化时序触觉描述器，将双指动态接触离散为具有物理含义的元描述文本，为故障理解与恢复推理提供简洁的语义输入。

**触觉驱动的故障恢复。** 恢复能力的学习需要有效的纠正反馈：[TacCoRL](https://arxiv.org/abs/2606.11743) 通过仿真强化学习，以任务奖励优化近失败状态下的触觉条件动作；为扩展纠正数据，[TACO](https://arxiv.org/html/2607.02840v2) 则从真实失败附近生成视触觉轨迹，并按预测进度筛选，用于策略后训练。两者主要通过任务结果学习纠正行为，没有根据故障原因确定明确的恢复目标。在部署阶段，[ViTaR](https://arxiv.org/abs/2608.15816) 根据候选纠正的效果比较，结合实时视触觉反馈选择并缩放残差，实现在线局部干预，但纠正范围由固定候选集限定。[PEARS](https://arxiv.org/html/2610.08784v1) 进一步利用视觉语言模型分析失败，但其推理输出主要限于接触力或力矩的范围，且在回合结束后更新，无法在线恢复。本文利用触觉反馈理解接触故障并推理恢复策略，实现在线故障恢复。

## English

```latex
\section{Related Work}
\label{sec:related_work}

\paragraph{Tactile integration and contact semantics.}
Tactile sensing complements contact information that is difficult to observe visually, extending its role from local control to high-level decision-making. Tactile-VLA~\cite{} combines tactile reasoning with hybrid position--force control, while VLA-Touch~\cite{} uses tactile semantics for planning and action refinement. Supporting such reasoning requires semantic interpretation of sensor signals. TVL~\cite{} describes tactile object properties through touch--vision--language alignment. CLTP~\cite{} further addresses contact geometry and forces from the sensor's perspective, while AnyTouch~2~\cite{} extends tactile representation learning to temporal deformation and force-aware dynamics. We adopt a lightweight temporal tactile descriptor that discretizes dynamic contact signals from both fingers into physically meaningful textual descriptions, providing concise semantic inputs for failure understanding and recovery reasoning.

\paragraph{Tactile-driven failure recovery.}
Learning to recover requires informative corrective feedback. TacCoRL~\cite{} uses task rewards in simulation-based reinforcement learning to improve tactile-conditioned actions in near-failure states. To scale corrective supervision, TACO~\cite{} generates visuo-tactile trajectories from states near real failures and selects them by predicted progress for policy post-training. Both learn corrective behaviors primarily from task outcomes, without explicitly deriving recovery objectives from diagnosed failure causes. At deployment, ViTaR~\cite{} uses comparisons of candidate correction effects and real-time visuo-tactile feedback to select and scale residual actions, enabling online local intervention within a fixed candidate set. PEARS~\cite{} further employs a vision-language model to analyze failures, but its reasoning outputs primarily specify contact-force or torque bounds, with failure-driven updates made after each episode for subsequent executions. We instead use tactile feedback to understand contact failures and infer recovery strategies during task execution, enabling online failure recovery.
```
