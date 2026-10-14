# GDN Stage6 首轮 B1 性能分析（2026-10-13）

当前结论：**B1 与历史 A0 的普通性能基本持平，没有证明 Stage6 提前搬运 K 带来收益。** 七项 kernel 时间和均值从 1708.391 us 变为 1708.209 us，仅减少 0.182 us（0.0107%）；直接修改的 Prepare 均值增加 0.703 us（0.0983%）。Stage6 的采样 median、p95 和窗口之和均升高。用户随后明确仅投入至少 5% 收益的改动，本候选停止实验、未采用，不再要求为它补测 A1；实验源码与本次证据保留。

## 来源及可比性

A0 来源目录：[atk_output_profiling](D:/Code/results/atk_output_profiling)，B1 来源目录：[atk_output_stage6_overlap](D:/Code/results/atk_output_stage6_overlap)。两目录各包含 run_metadata.json、brief.txt、profiling.log 和三个汇总 CSV；原 raw、parsed CSV、timer_layout/capture metadata、安装预检日志、CPU 精度报告未提供。以下读取已有汇总，不声称重新验证原 raw 的完整性。

| 核对项 | A0 | B1 | 含义 |
| --- | --- | --- | --- |
| metadata status / case / device / SoC | complete / 0 / 0 / ascend950 | 相同 | 原性能 case 执行成功 |
| metadata source_commit | 29422b8fb49094110208479f7692bd5a2651ea2c | 6dd2b1569f343bf4e7852f103bcea466fe3dacb5 | B1 的内部 commit 不在当前本地 Git 对象库，无法核对它的完整源码 diff |
| case_json_sha256 | 33530343cec97fc0d7ea7257229221c1f429d547d6e86699ccb23696f39858c0 | 相同 | 两轮记录的性能配置文件字节一致 |
| input_sha256["0"] | 1f4a29b6a40feb83a424979370638c0b1fb6ee51fa80dd0e513e5d22de73ef38 | 相同 | 两轮记录的输入快照文件字节一致；本机未直接加载张量 |
| Python / ATK launcher / OPP vendor 路径 | python3.11.10 环境与包内 OPP | 相同 | 路径相同不等于证明依赖版本和二进制内容相同 |
| brief device / torch | Ascend950PR_957c / 2.10.0+cpu | 相同 | 可见设备型号与 torch 版本一致 |
| timer captures / warmup / sample_limit | 1 / 5 / 32 | 相同 | warmup=5 仅控制诊断重放 |
| 普通 msprof 报告 | 2026-10-13 10:02 批次 | 2026-10-13 20:07 批次 | 历史与新测相隔约 10 小时，未形成临近交替对照 |
| 普通性能记录 | 1 个 op_summary，140 条 | 相同 | 各 20 组、每组七项 |
| 汇总结构 | 216 个 kernel/core 行，27 个 stage/type 行 | 相同 | 核集合、样本数、实际调用次数范围和截断数一致 |

模型仍为 case 0：q/k BF16 [1,16,11274,128]，v BF16 [1,32,11274,128]，g FP32 / beta BF16 [1,11274,32]，BT64、BNSD、varlen、FP32 初态及最终状态输出。可见参数满足当前候选的 V128/G2/Hv%4=0 条件，但实际编译分支和产物仍需设备安装/指令证据。

两轮 metadata 未包含 reference_wheel 身份字段，目录也没有 installed_artifacts.log。source_commit 和 OPP 路径不足以证明实际安装 wheel 与源码一一对应。后续 A1 及需要复测的 B 使用更新入口的 `--reference-wheel`，保留其产物校验结果；不要仅用同名 wheel 或 version 字符串区分版本。

brief 均报告 CSV=passed、decode=complete 和十项普通/诊断输出完全一致。这是各自版本普通调用与诊断调用的一致性；原始 CPU golden 验收、A/B 输出比较、MSS、确定性及其它支持场景仍未在本批材料中提供。performance 报告中的 accuracy_failed_case_ids=[] 也不等于执行过 CPU 精度验收。

## 普通调用 msprof：收益依据

按 metadata 原顺序核对全部 140 行：20 次重复
`Cast → Transpose 1 → Transpose 2 → Prepare → H → O → Transpose 3`。
所有 duration 为有限正数，每轮来自同一个 DUT op_summary。七项时间和按连续七行构造；metadata 没有 Task ID、Stream ID 或开始时间，因此这是按固定模式推定的调用组，不构建完整时间轴。

均值变化按 `(B1/A0 - 1) × 100%` 计算，负数表示耗时下降。单位 us。

| 项目 | A0 mean | B1 mean | B1−A0，us | 变化 |
| --- | ---: | ---: | ---: | ---: |
| Cast | 4.971 | 4.891 | -0.081 | -1.620% |
| 前置 Transpose 1 | 15.911 | 15.908 | -0.003 | -0.019% |
| 前置 Transpose 2 | 14.345 | 14.207 | -0.138 | -0.964% |
| Prepare | 715.090 | 715.793 | +0.703 | +0.098% |
| H | 588.803 | 589.879 | +1.077 | +0.183% |
| O | 359.376 | 358.217 | -1.159 | -0.322% |
| 末置 Transpose | 9.895 | 9.315 | -0.581 | -5.870% |
| Prepare + H + O | 1663.269 | 1663.889 | +0.621 | +0.037% |
| 全部七项 kernel | 1708.391 | 1708.209 | -0.182 | -0.011% |

直接受改动影响的 Prepare 没有变快。三项主 kernel 合计反而增加 0.621 us；七项合计的微小下降来自 Cast/三个 Transpose 合计减少 0.803 us，与主路径增加相抵。O 和末置 Transpose 的变化也不能直接归因于 Stage6，需临近恢复基线排除批次/缓存/环境差异。

| 项目 / 批次 | min / median / p95 / max，us | 本批 CV |
| --- | --- | ---: |
| Prepare / A0 | 713.217 / 714.835 / 717.501 / 717.848 | 0.157% |
| Prepare / B1 | 714.219 / 715.630 / 717.415 / 717.666 | 0.147% |
| H / A0 | 573.946 / 589.859 / 597.398 / 599.280 | 1.106% |
| H / B1 | 578.072 / 590.224 / 599.487 / 602.024 | 1.086% |
| O / A0 | 356.778 / 358.993 / 363.401 / 364.413 | 0.542% |
| O / B1 | 356.457 / 358.333 / 359.750 / 360.004 | 0.310% |
| 七项时间和 / A0 | 1692.236 / 1707.582 / 1721.140 / 1721.705 | 0.453% |
| 七项时间和 / B1 | 1694.958 / 1708.784 / 1718.535 / 1723.408 | 0.429% |

p95 使用 nearest-rank，CV 为总体标准差/均值。各项 p95 不相加，总 p95 在每组的七项时间和上计算。

七项均值变化仅 0.0107%，而各批 20 组时间和的 CV 约 0.43%～0.45%，标准差约 7.3～7.7 us。A0 前十组/后十组均值为 1711.478/1705.305 us，B1 为 1706.786/1709.633 us，批内也存在漂移。这里的 CV 只作噪声尺度描述，不能设为统计显著性的固定门槛。20 个记录是每次采集内的重复测量，不是 20 个独立实验批次，不能由此证明跨运行稳定收益。

均值下降约 0.011%，median 升高约 0.070%，p95 下降约 0.151%，各统计量方向并不一致。当前结果应记为“普通性能基本持平，收益未证实”，也没有足够证据宣告稳定回退。

## Stage6 诊断：目标窗口未改善

`PREPARE/WU_PREPARE/AIV` 对应 [Stage6_AivOne](../../chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h#L654)。两轮均为 56 个 AIV，各采样前 32 次，共 1792 次；实际每核 100～102 次，仅覆盖调用前缀，所有 56 个计数均截断。

| Stage6 观测量 | A0 | B1 | 变化 |
| --- | ---: | ---: | ---: |
| 单次窗口 median | 3074.5 | 3103 | +0.93% |
| 单次窗口 p95 | 5092 | 5692 | +11.78% |
| 单次窗口 max | 7775 | 7367 | -5.25% |
| 1792 次采样窗口之和（core-cycles） | 4865939 | 5303412 | +8.99% |

采样窗口均值为 2715.368 → 2959.493 cycles（+8.99%）。最大值下降 5.25% 的同时，median、均值和 p95 升高，不能只选 max 下降作为优化成功依据。core-cycles 是跨核采样 scope 的和，不能换算为整个 Prepare 增加 8.99%，也不是整个调用的收益占比。

longest_windows 只保留每轮最大的 12 个已采样动态窗口。A0 中 SOLVE 9 条（均为序号 0）、WU_PREPARE 2 条、KKT_EPILOGUE 1 条；B1 中 WU_PREPARE 10 条、KKT_EPILOGUE 2 条。B1 的长 Stage6 样本分布于多个核的调用序号 2/4/6，并非仅一个序号 0 尖峰。例如逻辑 core 16 的第 2 次 WU_PREPARE 为 7367 cycles，core 59 的第 6 次为 7087 cycles。序号是各核各 stage 的调用顺序，不是共享 task/chunk ID；这些表不能直接关联跨核 producer/consumer。

其它采样 stage 的变化如下，median 表示单次 scope 窗口：

| stage/type | A0 / B1 median，cycles | median 变化 | 采样窗口之和变化 |
| --- | --- | ---: | ---: |
| PREPARE/KKT_EPILOGUE/AIV | 2451 / 2325.5 | -5.12% | -4.85% |
| PREPARE/PREPARE_NORM_GATE/AIV | 1066 / 930 | -12.76% | -6.58% |
| PREPARE/SOLVE/AIC | 1982 / 1925 | -2.88% | -1.54% |
| PREPARE/W_MM/AIC | 1021 / 1020 | -0.10% | -2.30% |
| PREPARE/U_MM/AIC | 29 / 29 | +0.00% | +4.33% |
| FWD_O/O_WAIT/AIC | 857.5 / 879.5 | +2.57% | +1.32% |
| FWD_O/O_WAIT/AIV | 366.5 / 375 | +2.32% | +0.46% |

KKT_EPILOGUE、NORM_GATE 等窗口下降与 Stage6 上升同时出现，说明窗口组成和等待暴露位置可能发生变化；也可能含批次运行差异。scope 包含原内部等待及异步下发，不是纯计算时长。不能把这些 scope 的变化相加，推算 Prepare 的真实收益，更不能据此证明 K 的 MTE2 已与 V 运算重叠。

## 逐核总窗口与普通性能的方向不一致

下表为各核诊断 KERNEL_TIMING，总窗口包含诊断 drain；单位 cycles：

| kernel/type | A0 median | B1 median | median 变化 | A0 / B1 max |
| --- | ---: | ---: | ---: | ---: |
| PREPARE/AIC | 707743.5 | 706081.5 | -0.23% | 713692 / 712327 |
| PREPARE/AIV | 707500 | 706995 | -0.07% | 715347 / 711149 |
| FWD_H/AIC | 591794 | 582266.5 | -1.61% | 595664 / 590881 |
| FWD_H/AIV | 592285.5 | 583117 | -1.55% | 596597 / 591861 |
| FWD_O/AIC | 354782 | 356581 | +0.51% | 360200 / 362151 |
| FWD_O/AIV | 353770 | 356133.5 | +0.67% | 358149 / 361677 |

Prepare 总窗口 median 仅下降约 0.07%～0.23%，但普通 Prepare mean 增加约 0.10%。H 的诊断总窗口下降约 1.55%～1.61%，普通 H mean 却增加约 0.18%；O 的诊断总窗口增加约 0.51%～0.67%，普通 O mean 则下降约 0.32%。这批 timer 与普通性能分别是不同调用，诊断有打点/采样扰动，不能用 timer 总窗口替代普通 msprof。

核总窗口仍较集中，B1 max/median 约 1.006～1.016；Prepare 两个 AIV 子核 median 均只小幅变化，未出现一个子核系统性承担大幅新增总时长的证据。当前不优先据此改分核或增加核数。两轮都有 984 个 core/stage 计数截断，不向后续未采样区域外推。

## 解释与停止实验的决定

本候选的假设是“把 K 搬运提前到 V 缩放/上传之前，隐藏一部分原先暴露的搬运延迟”。普通 Prepare 未下降且 Stage6 前缀窗口变长，当前证据不支持这项假设已经带来收益。可能原因包括提前搬运未在生成指令中形成有效重叠、新增条件判断的标量开销、MTE/内存资源竞争或已有 Stage4/5/7 流水的等待位置改变；这些都是待验证解释，现有汇总没有 pipe 活动或拆分等待，无法选择其中某个原因。

用户于 2026-10-13 明确仅认为至少 5% 的收益值得投入。按已采用口径，将其记为普通七项 kernel 时间和平均耗时至少下降 5%，而非某个 timer scope 下降 5%。A0 为 1708.391 us，目标需不高于 1622.972 us，即至少减少 85.420 us；当前只减少 0.182 us，且直接修改的 Prepare 没有改善，现有证据足以作停止本候选投入的决定。

本候选标记为“停止实验、未采用”，无需继续为它补测 A1、B2/A2 或完成正式采用验收。若 A1 已经采完，可保留作后续同环境普通基线。实验代码保留在 B 分支供追溯，不自动回滚或删除分支；后续新候选从基线 A 出发。

仅改 Prepare 且其它六项不变时，Prepare 至少需下降约 11.95% 才能带来整体 5%；仅改 H 需下降约 14.51%。后续优先选择能影响这些预算的计算复用、总搬运量或关键等待；先证明潜在空间足够，再安排一个主要变量的设备实验。Stage3 KKT_EPILOGUE 的 producer 等待/后处理拆分仍可用于定位，但现有数据还不能保证该方向达到 5%。

目前只分析 case 0 和各一次采集。case 1/回退路径、CPU 精度、MSS、确定性、Example/ST 与跨批收益尚未齐全，因此本次不归档正式采用或完整验收通过。

## 分析复核与原文件指纹

已核对：两轮核/stage 主键唯一且集合相同，216 个核行/27 个 stage 行、各 stage 的 actual/sample/truncated 计数和各核采样事件计数一致；140 个普通耗时为有限正数、七项顺序重复一致；七项均值等于各项均值之和。统计从原数字重算，保留原源文件，本次没有改写结果目录。

| 文件 | A0 SHA256 | B1 SHA256 |
| --- | --- | --- |
| run_metadata.json | 1fc92659f6a7b3ba782606646518c280e9836fc767ed2a1aff1cd4df3be20d48 | b5292a2ed864e16927aebbcb051ebd0ea824a6b3a2735a0c2f6eaa8f8ed411ab |
| core_summary.csv | 2f7f47f02256e46b712bc24e90b05d8d2b01c415278e4798d9ec109e046b2573 | 12ac61aac4c42be30943e06b08ae882d712e63389b994f861a6a4990dba4396d |
| stage_summary.csv | 9540796367236b436fcc317c16f29f1af835e986bff40d840553b82a8fc23d31 | 07bef06c2dc274031cd7d164f7f6aa0fb46160b92178de89a99a896150e45cdd |
| longest_windows.csv | 2bf21155e40e174d784a6e3e8d487f6c7a4015a1015d37c71a7566924b5ec715 | 5b77c3567e5644ea79ecb1f424cc8ce517d6da8a70bca19e2c6237787046c7ee |
| brief.txt | 80ef643dbe266432a104f14891fd3c663a339797c6bf3da9cfa3c201dba8a16b | d0ab157f7eac283ade3d37eca2dd7761ebf17f5fb2689e60fc927196430e394a |
| profiling.log | e45ebadfe2959e76df4633addc7fd48411e6c3e856cd369f0d002ab67c549049 | 861871b0cbf2d50cfcdcb3980e20ae3f80b2f0c7876729c638b897a831e2a44b |

既有背景见 [A0 分析](timer_analysis_case0_2026-10-13.md)，指标解释见 [timer_analysis.md](timer_analysis.md)，A/B 切换及参数见 [timer_run.md](timer_run.md)，候选状态见 [validation.md](validation.md)。

