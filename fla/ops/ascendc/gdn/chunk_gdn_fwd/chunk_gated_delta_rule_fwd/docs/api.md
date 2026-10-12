# GDN 前向 profiling 接口差异

本文件归档本次诊断接口契约；原有数学计算、支持组合及十项输出沿用本算子 README、
`aclnn_chunk_gated_delta_rule_fwd.h` 和现有 ATK CPU 标杆。
用户于 2026-10-11 选择方案 A，并限定本次为源码打点；未接入生产验证环境。

## 接口与参数顺序

正式 `aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize` / `aclnnChunkGatedDeltaRuleFwd`、
Stable-ABI schema 和正式 Python wrapper 的签名保持原样。
诊断入口增加以下两个独立 C 符号，准确声明位于 `../op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.h`：

```text
aclnnChunkGatedDeltaRuleFwdProfileGetWorkspaceSize(
  q, k, v, g, beta, aLogOptional, dtBiasOptional, initialStateOptional,
  cuSeqlensOptional, chunkIndicesOptional, layout, scale, chunkSize,
  useExp2, useQkL2norm, allowNegEigval, stateVFirst,
  oOut, finalStateOutOptional, qHatOutOptional, kHatOutOptional,
  qRstdOutOptional, kRstdOutOptional, betaEffOutOptional,
  gCumsumOutOptional, aOutOptional, hOutOptional,
  timer, workspaceSize, executor)
aclnnChunkGatedDeltaRuleFwdProfile(workspace, workspaceSize, executor, stream)
```

前 27 项与正式入口逐项同名、同序、同型；第 28 项 `timer` 是 `const aclTensor *`，
指向由调用方持有、供 kernel 原位写入的存储。最后两项仍是 workspaceSize/executor 输出指针。
launcher 复用原 executor 执行机制。诊断 API 不注册新 Stable schema，不增加 ATen 依赖。

内部 L0 的四个算子均在 `executor` 后增加 `timerOptional=nullptr`；OpDef 最后一个输入是
OPTIONAL INT64 ND `timer`，raw kernel 同位置接收 `timerAddr`，原输出和属性索引保持原样。
所有受影响的 host/OPP/kernel 需要一起重建，不能把新 host 与旧 kernel 混装。

## Timer 输入

| 项目 | 契约 |
| --- | --- |
| shape/dtype/format | 一维、连续、ND、INT64，元素数至少为共享头的 `AscendTimer::TOTAL_BUFFER_SIZE` |
| 当前布局 | 49,216 个 int64，393,728 字节；使用常量分配，不能把当前数值写死在调用代码 |
| device | 与 q 同一个 NPU；Python 诊断 helper 校验，直接 C 调用方负责使用同设备存储 |
| 生命周期 | 启动前清零；保持存活直到该 stream 完成；导出前执行 NPU synchronize |
| 所有权 | 独立于计算输入、输出和 workspace；不同并发调用必须使用不同 timer |
| 可选性 | 正式调用传空，诊断 L2 调用必须非空；无空输入采集事件 |
| 边界 | 预留最多 256 个 MIX AIC:2AIV 核组；实际 blockDim 写入 metadata，越界核心不写 GM，解析器拒绝超限 |
| 错误 | 空 timer 返回 ACLNN_ERR_PARAM_NULLPTR；dtype/rank/尺寸/连续性/ND 错误返回 ACLNN_ERR_PARAM_INVALID |

禁止为 timer 调用 AutoContiguous/MakeContiguous，避免记录写进临时副本。
空计算输入仍按原入口返回；不会启动 kernel，解析器以“无采集事件”报告，不能当成耗时零。
timer 使用 kernel 的 GM 写入路径，未验证 graph capture；本诊断入口仅用于 eager 独立采集。

Python 私有诊断 helper 位于 `_aclnn_ctypes.py`：
`_profile_chunk_gated_delta_rule_fwd(q,k,v,g,beta,*,timer,**原有选项)`。
返回与正式入口同序的十元组；timer 由调用方单独读取，不插入返回 tuple。
该 helper 在 MUTATED_ARGUMENTS 中声明 timer mutation，拒绝 timer 求梯度，
成功 launch 后推进其 PyTorch version counter，校验/launch 失败时不推进。
`tests/ut/profile_gdn.py` 的 `capture` 为实际模型 tensor 提供采集入口，也可通过 `--inputs` 加载保存的输入。

## 数值语义与支持声明

本次增加诊断观察与 kernel 末尾局部 PIPE_ALL drain，未修改公式、tile、分核、循环、
原有同步或 CPU 标杆；阶段 02 复用现有 CPU / 六算子双标杆。
覆盖源码中 arch22/arch35 Phase6 和公开 Prepare/H/O 两条路由。
原有 SoC/dtype/layout/fixed/varlen/特性限制继续由原 host 校验。
这些是源码覆盖范围，不能据此宣称本次在 A2/A3/A5 编译或设备验收通过。
