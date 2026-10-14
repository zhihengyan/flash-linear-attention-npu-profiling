# GDN 后续优化方向判断（2026-10-13）

当前建议：**把 H 的双 head work unit 跨 chunk 输入预取/流水扩展放到下一轮首位，Prepare Stage3 的诊断拆分作为备用定位方向。** 这是基于现有源码和 case 0 的候选选择，尚未证明硬件收益；本次未创建分支或修改 kernel。新实验从 codex/gdn-profiling 的固定基线 `809bce6887639e831caa856b8ac3a197a6a94294` 出发，不叠加已停止的 Stage6 候选。

## 5% 的预算与改动粒度

沿用已有口径：普通 msprof 中一次公开调用七项 kernel 的 Task Duration(us) 时间和。该指标不包含 host/launch 间隙；真实 API 墙钟时间需要单独核对，不将 kernel 时间和直接称为全部端到端时间。

| 路径 | A0 mean，us | 七项时间和占比 | 只优化该路径时，为整体减少 5% 所需的耗时下降 |
| --- | ---: | ---: | ---: |
| Prepare | 715.090 | 41.86% | 11.95% |
| H | 588.803 | 34.47% | 14.51% |
| O | 359.376 | 21.04% | 23.77% |

整体基线 1708.391 us，5% 为 85.420 us。若其它六项不变，H 应从 588.803 us 降到不高于 503.383 us；H 耗时减少 15% 对应整体约减少 5.17%。这是条件预算，不是候选预报。

改动粒度需要区分：调整一条 Copy 的位置通常只影响流水的一小部分；重排一段完整、反复执行的 chunk 流水则可以同时改变多个操作的重叠。后一项仍可作为一个主要策略的实验，内部多个代码位置共同实现它，整体继续按至少 5% 验收。若探索一个组合方案，也应逐项确认增量和相互影响，而非把多个 scope 的缩短相加。

## 实际 H 路由与源码证据

本 case 的 Prepare/H/O host 路由调用的是独立 `ChunkFwdH`，见 [GDN host 调用](../../chunk_gated_delta_rule_fwd/op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp#L770)。优化对象是 `chunk_fwd_h/op_kernel/arch35`，不直接修改另一个公开 `ChunkGatedDeltaRuleFwdH` 或融合内部 legacy H。

已核对 ChunkFwdH 设计、tiling、AIC/AIV 实现与调用方：

- [tiling](../../chunk_fwd_h/op_host/chunk_fwd_h_tiling_processor.h) 按 sequence/head 分配连续区间；case 0 的 32 个 value head 与已测 16 AIC 相符，为每个 AIC 两个 head。`FwdHBuildHeadRange` 将两 head 分给 AIV0/AIV1，各持有一个 local head；G2 的两个 head 共享同一 source K。上述映射与 H_WH/H_KV 每 AIC 实际 354 次、H_VNEW/H_STATE_UPDATE 每 AIV 实际 177 次相符，新的设备实验仍保留实际 tiling/分支命中证据。
- [Cube ProcessWorkUnit](../../chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h#L738) 仅在 FP32 state、scalar-g、activeHeadCount==1 时调用 `ProcessSingleHeadPipeline`。
- [Vector ProcessWorkUnit](../../chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h#L1162) 使用同样的 activeHeadCount==1 条件。case 0 为两 head work unit，不命中这条跨 chunk lookahead 路径。
- 单 head 的现有路径以 chunk 奇偶 slot 预取下一 chunk 的 W/K 与 U/g，保持当前 H 的真实递推就绪依赖；双 head 走每 chunk RunStage0/RunStage2 和 RunStage1/RunStage3 的通用循环。通用循环已有本 chunk 内的预取/重叠，不能称为完全串行，但没有该单 head 的跨 chunk lookahead 策略。
- FP32 rolling state 已在 AIV UB 跨 chunk 常驻，A5 的 P/D 已由 Fixpipe 直接写配对 AIV UB。不能把“消除这些已有 GM 往返”当成新候选。当前仍有历史 H 和 right 的 GM 写读；H 是公开后续 O/输出所需，right 的 GM 路径是现有设计约束，不在本轮直接删除。

当前 HEAD 与 codex/gdn-profiling 在 chunk_fwd_h 目录无源码差异，因此上述 H 策略存在于拟用基线中。B1 的内部 commit 未在本地对象库，不用当前源码替代其全部构建版本证明。

## 首选候选：双 head 跨 chunk lookahead

唯一主要变量：将单 head 已有的“预取下一 chunk 的独立输入”策略扩展到 case 0 的双 head work unit，协调 AIC/AIV 的下发时机和轮转 bank。先限定 FP32 state、scalar-g、两个 head 共用一个 K head 的场景，其他路径保留既有实现。

计划方向：

1. AIC 保持当前 chunk 的 H/right 真实就绪依赖，在合法的 slot 释放点提前加载下一 chunk 的两份 W 和一份共享 K；共享 K 仍只加载一次。
2. AIV0/AIV1 各处理一个 head，以两个 U/g input bank 按 chunk 奇偶轮转，把下一块输入搬入安排到当前计算/递推等待期间。
3. 保持 WH→V_new/right→KV→FP32 state update 的每 head 依赖、算术/指数/cast 顺序、H/vNew/finalState 及 tail/varlen 语义。不得提前读取尚未生成的下一块 H，也不跨 chunk 并行状态递推。
4. 重新证明 L1 W/K bank、UB U/g/gate bank、P/D/right 别名及 ready/free 计数闭环。W/K 当前共用部分 event 资源，不能仅把条件 ==1 改成 ==2；单 head 函数也只覆盖一个 head，直接放宽条件会遗漏第二个 head。

该策略作用于长序列中重复执行的输入准备和 Cube/Vector 交接，具有比单次局部 Copy 移动更明确的累计作用范围。是否能隐藏至少约 85 us 仍未知：同一 MTE2 队列、带宽、scalar 开销或严格状态依赖可能限制收益，不能因为缺少这条特化就宣称存在 15% H 改善。

## 最小定位与实验门槛

只围绕该候选补证，避免无限扩充所有 Stage 的计时：

- 查 AIC 的 LoadStage0W/LoadStage0H、LoadKg/LoadRight，AIV 的 PrefetchStage1Inputs、输入 bank 复用和 H/right ready 交接，区分独立输入与递推输入。
- 区分启动、稳定 chunk 和尾块。H 当前动态窗口并集只有约 1%～1.6%，且只保留前 32 次，不能从短 H_WH/H_KV 或 H_WAIT 推断算力/带宽/等待占比。
- 必要时结合 pipe profiling 判断暴露的 MTE2/搬运等待。NoBarrier API 窗口不等于真实执行完成时间，不为计时插入额外功能同步。
- 进入编码前，按实际分支完成地址/同步证明并判断目标空间；若独立输入准备已充分隐藏或潜在收益不足约 85 us，则停止这项候选，转查真正关键路径。

候选就绪后按仓库阶段 03→04→05：一个流水策略的 A/B，先精度，再普通 msprof。成功要求普通 H 接近上述目标、七项合计至少减少 5%、重复批次收益稳定，并通过模型 case 1 和其它受影响支持场景。只缩短 timer 或只让一项 H scope 下降 15% 不算达标。

ChunkFwdH 是共享算子，调用者还包括独立 H 公开入口、GDN 反向重算以及 KDA 前向/反向 V2。实际编码时须核对这些调用是否命中新条件，合并受影响场景的精度与性能回归；不以只优化 GDN case 0 为由忽略其它调用者。

## 当前顺序

| 顺序 | 方向 | 当前判断 |
| --- | --- | --- |
| 1 | H 双 head 跨 chunk 独立输入预取与流水 | 有具体源码策略缺口，优先做目标定位和设计；收益待设备验证 |
| 2 | Prepare Stage3/KKT producer/后处理关键等待 | 有较大普通时间预算，但现有混合 scope 无法选择具体改动；作为备用诊断 |
| 暂不选 | 单纯增 H 核数、重复做 state/P/D GM 消除、Stage6 微调 | 增核不直接缩短仍需两 head 的最慢组；state/P/D 已有驻留/直接 UB；Stage6 未显示收益 |

基线数据与限制见 [A0 分析](timer_analysis_case0_2026-10-13.md)，被停止候选见 [B1 分析](timer_analysis_stage6_b1_2026-10-13.md)。这里只给出下一轮选择，不保证任一候选已有至少 5% 的实测收益。
