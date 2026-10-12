# GDN 前向内核计时接入方案

状态：用户已选择方案 A，host/kernel/Python 源码接入完成；本次仅要求源码打点。
生产验证环境不可接入，未编译 CANN、未运行 NPU、未采集真实设备数据。
方案 B 保留为当时的备选分析，本轮不实施。独立审核结果见 `timer_review.md`。

## 1. 来源与范围

- 项目级 skill：`.agents/skills/op-timer-ascendc/SKILL.md`。
- 克隆命令：`git clone git@gitcode.com:zhihengyan/op-timer-ascendc.git .agents/skills/op-timer-ascendc`。
- skill 固定提交：`52e396e599a0101b00757ab575c029cb4284ad24`，许可为 MIT。
- 目标仓库扫描提交：`bead79ae45fca5d10a8414b215d4f762cc66538d`。
- 仓库流程：根 `AGENTS.md`、`docs/agents/01-接口确认.md`、`03-方案设计.md`、
  `04-算子开发.md`、`05-算子测试.md`；skill 提供计时方法，仓库文件决定接口与验证流程。
- `fla/ops/ascendc/gdn/` 是模块目录，包含预处理、前向和反向；
  `chunk_gdn_fwd/` 是前向算子集合。
- 完整前向入口所在目录为 `chunk_gdn_fwd/chunk_gated_delta_rule_fwd/`。
  本次以这个公开前向入口为对象，沿实际调用关系覆盖其 kernel；反向和独立预处理不在本轮范围内。

### 源码锚点

下面的路径相对于 `fla/ops/ascendc/gdn/chunk_gdn_fwd/`，行号是打点前的源码行号。

| 编号 | 文件与代码位置 | 已确认事实 |
| --- | --- | --- |
| H0 | `chunk_gated_delta_rule_fwd/op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp:103`，`UsePreparePath` | 当前代码按 SoC、shape 和特性选择路径，不能仅依据 README 的旧条件 |
| H1 | 同文件 `:600`，`ChunkGatedDeltaRuleFwdGetWorkspaceSizeImpl` | 校验、连续化、gate 转换、构建 executor DAG |
| H2 | 同文件 `:746/:759/:765` | 新路径依次下发 Prepare、ChunkFwdH、ChunkFwdO |
| H3 | 同文件 `:863/:929` | 另一分支下发 Phase6；执行阶段调用已有 executor |
| K0 | `chunk_gated_delta_rule_fwd/op_kernel/chunk_gated_delta_rule_fwd.cpp:5` | `__CCE_AICORE__ == 310` 引入 arch35，其余引入 arch22 |
| K1 | `chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:325`，`RunPhase6` | 私有 cumsum/KKT/solve/WU/H/O 和已有同步边界 |
| K2 | `chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:472`，`RunPhase6` | A2/A3 私有前端与可选 H/O 重叠调度 |
| P0 | `chunk_gated_delta_rule_fwd_prepare/op_kernel/chunk_gated_delta_rule_fwd_prepare.cpp:17` | 独立 Prepare kernel，MIX AIC:2AIV |
| P1 | `chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:775/:828` | AIV/AIC 两套 pack 循环，有跨 pack 延迟写回 |
| S0 | `chunk_fwd_h/op_kernel/chunk_fwd_h.cpp:47/:82` | 新 H kernel 使用 ChunkFwdH，部分短序列 AIC 提前返回 |
| S1 | `chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:669/:709` | 单 head 预取分支及多 head 的 work-unit/chunk 循环 |
| S2 | `chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h:1148`，`ProcessWorkUnit` | 初态转换、Stage1/3、round 尾 drain/握手 |
| O0 | `chunk_fwd_o/op_kernel/chunk_fwd_o.cpp:62/:100` | A5 token-first 输出进入专用实现 |
| O1 | `chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_cube.h:102` | groupTask 循环：Stage2、Stage4、Stage5 free 等待 |
| O2 | `chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_vector.h:278` | groupTask 循环：Stage1、Stage3、Stage5、非 owner 握手 |
| B0 | `chunk_gated_delta_rule_fwd/op_host/CMakeLists.txt:2` | 构建依赖包含 Prepare、ChunkFwdH、ChunkFwdO |

Python 默认入口实际使用
`torch_custom/fla_npu/csrc/src/stable_chunk_gated_delta_rule_fwd.cpp` 的 Stable-ABI 适配层，
不是 skill 示例中的 ATen `_torch_adpt.h`。接入应沿现有架构完成，不引入示例的 ATen 依赖。

## 2. 改造前原始执行流程

图中 F 为一次性固定流程，D 为重复流程；方括号注明执行单元和原有完成/复用锚点。
这是改造前的流程，没有计时调用。布局转换、Cast、ViewCopy 是 executor 中的辅助任务，
本轮内核计时的主要对象是下图的 GDN kernel。

```text
Python chunk_gated_delta_rule_fwd
  -> Stable-ABI schema/run_ -> aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize
  -> F [host] 校验/连续化/shape/metadata/gate 布局转换
  -> UsePreparePath
     |
     +-- true: F [host] 构建 Prepare -> H -> O -> 按需导出的 DAG
     |    -> 各 op tiling -> aclnnChunkGatedDeltaRuleFwd 执行 executor
     |    -> F [Prepare AIC/AIV] Init/资源与事件初始化
     |    -> F [Prepare AIV] Stage0 resident aux [MTE3_V 等原有事件]
     |    -> D [Prepare] pack=coreIdx; pack<nPacks; pack+=numCore
     |         AIV: S1 gate/beta/上传 -> S3 系数处理
     |              -> drain 前一 pack 的 W/U -> S6 -> 发布 S6 ready
     |         AIC: 等 S1 -> S2 KKT -> 等 S3 -> S4/S5 solve
     |              -> 等 S6 -> S7 W/U -> 发布 S7 done
     |         [WaitAivStage*Done / WaitAicStage7Done / HardEvent]
     |    -> F [Prepare AIV] 等最后 S7，drain 最后一 pack W/U
     |    -> kernel 边界，executor 依赖后续 H
     |    -> F [H AIC/AIV] Init；无 Cube 工作的 AIC 分支直接返回
     |    -> D [H] sequence/head work unit -> chunkId
     |         AIC: S0 W@H -> S2 K^T@right
     |         AIV: S-1 初态转换 -> S1 V_new/right -> S3 state 更新
     |         [P/D ready/free、RIGHT_READY/FREE、H_READY、round DONE/ACK]
     |         [首 chunk 无初态跳 S0；最后 chunk 无 final_state 跳 S2/S3]
     |         [单 head FP32 标量 gate 有独立预取循环]
     |    -> F [H] round drain / DrainEvents -> kernel 边界
     |    -> F [O AIC/AIV] Init
     |    -> D [O] groupTaskIdx=coreIdx; +=coreNum -> headOffset
     |         AIV S1 gate -> AIC S2 QK/QH -> AIV S3 gate/mask
     |         -> AIC S4 A'@V_new -> AIV S5 输出融合/GM 写回
     |         [cubeToVec/vecToCube、HardEvent、streamSlot 复用]
     |         [非 owner AIV 仍参与 ready/free 握手]
     |    -> F [O] 最后 Stage5 free / L1 event drain -> kernel 返回
     |
     +-- false: F [host] Phase6 输入/输出、私有 workspace -> tiling
          -> aclnnChunkGatedDeltaRuleFwd 执行 executor
          -> arch22 或 arch35 RunPhase6
          -> F [AIC/AIV] tiling/workspace/调度上下文
          -> D [AIC] KKT score；D [AIV] cumsum/系数
          -> D [AIC/AIV] solve -> recompute W/U -> 按需导出 cumsum
             arch35: cumsum+score 原有 SyncAll，solve->WU 原有 flags
             arch22: 可选 prepared-cumsum/分批前端与原有 flags/SyncAll
          -> D [私有 H] sequence/head/chunk 递推
          -> H/O 调度分支
             串行: 原有 H->O SyncAll -> 私有 O
             重叠: 原有生产者/消费者 ready 协议 -> 私有 O
                   -> 原有最终 SyncAll
          -> kernel 返回 -> host ViewCopy/按需布局转换
```

**路径结论：**当前 A5 的 BF16、K=V=128、chunk=64、合法 head ratio 不大于 4 的输入，
即使关闭 exp2 和 L2Norm，也可能进入三 kernel 路径。给 Phase6 主文件打点不能覆盖这些输入。
新路径 H 使用 `chunk_fwd_h`；同级 `chunk_gated_delta_rule_fwd_h` 是另一算子。
Phase6 还使用自己的私有算子副本，不能通过修改同级 H/O 源码自动覆盖。

2026-10-12 GM 发布修复不改变上图业务节点或方案 A 的 Tik/Tok 位置：
绑定 timer 时，AIC 0 在总 Tik 前逐 cache line 写回 metadata；共同 Finish 在两个结束时间戳保存后
逐 cache line 写回本核事件行。Prepare 显式关闭编译器的 kernel-end DCCI，因此不能依赖自动写回。
只操作 timer 的独占 cache line，保留原有 flags、SyncAll 和流水；细节见 `design.md` 的 GM 发布节。

## 3. 两种候选方案

两个方案都覆盖同一前向入口的两条路径。初次设备验收按实际目标 SoC 执行，
其他 SoC 只标明适用实现，未经设备验收不声明通过。

### A：kernel 与主要边界

目的：先获得逐核总时长、核间负载差异、三 kernel 的耗时分布，以及 Phase6 大阶段的调度时间线。
事件少，循环内部的计算与等待不能充分分离。

```text
原 host 主线 H0 -> H1 -> 路径选择 -> tiling -> executor launch
  新路径:
    [Tik KERNEL] -> [Tik INIT] Init [Tok INIT]
      -> [Tik BODY] Prepare pack 流水 + 最后 W/U drain [Tok BODY/KERNEL]
    [Tik KERNEL] -> [Tik INIT] H Init [Tok INIT]
      -> [Tik BODY] H work-unit/chunk + 原有 drain [Tok BODY/KERNEL]
    [Tik KERNEL] -> [Tik INIT] O Init [Tok INIT]
      -> [Tik BODY] O groupTask/head + 原有 drain [Tok BODY/KERNEL]
  Phase6:
    [Tik KERNEL] -> [Tik INIT] Init [Tok INIT]
      -> [Tik BODY] 前端 [瞬时 PHASE6_FRONT_DONE]
      -> 私有 H [瞬时 PHASE6_H_DISPATCH_DONE]
      -> [PHASE6_HO_WAIT 包围原串行 SyncAll] 或原重叠 ready 协议
      -> 私有 O/原最终会合 -> [Tok BODY/KERNEL]
```

| 阶段名 | 位置/归属 | 类型 | 起止与模式 | 完成含义 |
| --- | --- | --- | --- | --- |
| `KERNEL_TIMING_IDX` / `KERNEL_TIMING` | P0/S0/O0/K1/K2 的整个实际 kernel | F，所有核型 | `TikNoBarrier`；末尾 `Tok<Overwrite>` | 末尾本核 drain 后的总窗口，不能测 host 辅助任务 |
| `MAIN_INIT` | P0 的 `op.Init`，S0 的参数/服务初始化，O0 的服务初始化，K1/K2 的 workspace 解析 | F | `TikNoBarrier`；初始化纯标量段用 `TokNoBarrier<Overwrite>`；包含异步初始化时仅记 issue window | 初始化窗口必须注明 scalar/issue，不把异步下发视为完成 |
| `MAIN_BODY` | 三个独立 kernel 主体；Phase6 的完整主体 | F | 独立 timer 对象 `TikNoBarrier`，共享已有末尾 drain 后 `TokNoBarrier<Overwrite>` | 不为 BODY 重复添加一次排空 |
| `PHASE6_FRONT_DONE` | K1/K2 前端结束、进入 H 之前 | F，瞬时时刻对 | 相邻 `TikNoBarrier` + `TokNoBarrier<Overwrite>` | 到达调度点；不会声称所有 pipe 已完成 |
| `PHASE6_H_DISPATCH_DONE` | K1/K2 `DispatchFwdH` 返回、原有 H/O 分支之前 | F，瞬时时刻对 | 相邻 `TikNoBarrier` + `TokNoBarrier<Overwrite>` | H 返回/下发时刻；重叠分支不据此宣称 H 全局结束 |
| `PHASE6_HO_WAIT` | K1/K2 原串行 H/O SyncAll | F | `TikNoBarrier`；原同步之后 `TokNoBarrier<Overwrite>` | 原有会合的时间；重叠分支不新增这个同步 |

两个瞬时时刻之间只解释为调度时间窗。相邻两行的 start/end 平均数作为时刻，
不能把单个瞬时事件本身的 duration 当作业务耗时。

### B：A + 循环时刻点与已有等待（推荐）

目的：在保留原始流水的前提下，把 pack、head round/chunk、groupTask 的分布和等待位置展开，
定位慢核、长尾和同步气泡。采用 skill 的时刻点法处理没有安全完成锚点的异步计算段。
本方案不会在每个 MMAD/VF 调用后强制排空流水；计算段是明确标注的 issue window，
真实完成时刻只在对应的既有完成锚点之后记录。

```text
原 host 主线 H0 -> H1 -> 路径选择 -> tiling -> executor launch
  新路径，沿用 A 的 KERNEL/INIT/BODY:
    Prepare pack:
      [PREP_PACK_BEGIN] -> 原 S1/S3/S6 与 S2/S4/S5/S7 流水
      -> [PREP_WAIT_* 包围已有等待] -> [PREP_PACK_END]
      -> 原末 pack S7 wait + W/U drain，包含在 BODY
    H work-unit/chunk（含单 head 独立预取分支）:
      [H_CHUNK_BEGIN] -> S0/S1/S2/S3、原 ready/free
      -> [H_WAIT_* 包围已有等待] -> [H_CHUNK_END]
      -> 原 round drain/ACK，包含在 BODY
    O groupTask/head:
      [O_GROUP_BEGIN] -> S1/S2/S3/S4/S5、原 owner/non-owner 握手
      -> [O_WAIT_* 包围已有等待] -> [O_GROUP_END]
  Phase6，沿用 A 的边界:
    私有 H 的 sequence/head/chunk -> [H_CHUNK_BEGIN/END + H_WAIT_*]
    原 H/O SyncAll 或 ready 协议 -> [PHASE6_HO_WAIT]
    私有 O 的 task/head -> [O_GROUP_BEGIN/END + O_WAIT_*]
```

| 动态类型（候选名称） | 原代码位置/归属 | iter 轴 | 起止与模式 | 解释与完成锚点 |
| --- | --- | --- | --- | --- |
| `DYN_PREP_PACK_BEGIN/END` | P1 AIC/AIV 各自 pack 循环首尾 | 本核执行的 pack 序号 | 两个不同槽位各记相邻瞬时 Tik/Tok，`NoBarrier<Overwrite>` | pack 调度窗口；AIV 尾点不等价于本 pack W/U 已写完，存在延迟 drain |
| `DYN_PREP_WAIT_INPUT` | P1 `WaitAivStage1Done/Stage3Done/Stage6Done` | 同一 pack 序号 | 包围既有等待；可证明等待完成时 `TokNoBarrier<Accumulate>` | 多个 head/tile 重复写同槽，CSV 只分析累计 duration |
| `DYN_PREP_WAIT_REUSE` | P1 `WaitAicStage7Done`、原复用等待 | 同一 pack；最后 drain 的等待归最后 pack | 同上 | 等待上一 pack 或最后一 pack；metadata 标明其对象 |
| `DYN_H_CHUNK_BEGIN/END` | S1/S2 以及 Phase6 私有 H 的真实 chunk 循环 | 本核跨 work-unit 单调递增的 chunk 序号 | 两个不同槽位的瞬时对，`NoBarrier<Overwrite>` | 不直接用 `chunkId`，避免另一 sequence/head round 覆盖 |
| `DYN_H_WAIT_STATE` | S1 `H_READY` 等 state ready 的实际等待；私有 H 对应位置 | chunk 序号 | 包围既有等待；条件同上，`NoBarrier<Accumulate>` | 首 chunk 无初态时可以完全无此事件 |
| `DYN_H_WAIT_P` | S2 Stage1 中 P ready/free 的实际等待；私有 H 对应位置 | chunk 序号 | 同上 | 保留所有原 flag，不添加完成通知 |
| `DYN_H_WAIT_RIGHT` | S1/S2 `RIGHT_READY/FREE` 等实际等待 | chunk 序号 | 同上 | 多 head 累计，仅做 duration 分析 |
| `DYN_H_WAIT_D` | S2 Stage3 的 D ready/free 及对应私有 H 等待 | chunk 序号 | 同上 | 最后 chunk 不生成状态时不强制写入此类事件 |
| `DYN_O_GROUP_BEGIN/END` | O1/O2 与私有 O 的真实 task 外循环 | 本核 groupTask 序号 | 两槽瞬时对，`NoBarrier<Overwrite>` | 保持两 AIV 的 owner/non-owner 关系，记录 task/head 关联 metadata |
| `DYN_O_WAIT_STAGE2` | O2 Stage3 的 `cubeToVecFlag_` 等待 | groupTask 序号 | 包围原等待，条件同上，`NoBarrier<Accumulate>` | owner/non-owner 分别统计，不删除非 owner 等待 |
| `DYN_O_WAIT_STAGE3` | O1 Stage4 的 `vecToCubeFlag_` 等待 | groupTask 序号 | 同上 | 同轮多 head 累计 |
| `DYN_O_WAIT_STAGE4` | O2 Stage5 的 `cubeToVecFlag_` 等待 | groupTask 序号 | 同上 | 与 Stage2 等待使用不同槽 |
| `DYN_O_WAIT_REUSE` | O1 下一组/尾部 Stage5 free，O2 原 UB 复用等待 | 对应当前或最后 groupTask 序号 | 同上 | 不把 tail drain 遗漏在总窗口之外 |

注意：按 pipeline 指定的 WaitFlag/CrossCoreWaitFlag 不能仅凭函数名就视为 Scalar 阻塞或所有工作完成。
实施时逐个核对作用 pipe；只有真正完成的等待才能导出为 wait duration。
否则在其原消费/完成锚点记瞬时时刻，命名为 issue/ready point，不报告假的阻塞时间。
普通 NoBarrier 段也不能直接解释成纯 Cube/VEC 执行时长。

### 枚举与解析映射提案

固定枚举（A/B 共用）按索引：

```text
0 KERNEL_TIMING_IDX       -> Python "KERNEL_TIMING"
1 MAIN_INIT              -> Python "MAIN_INIT"
2 MAIN_BODY              -> Python "MAIN_BODY"
3 PHASE6_FRONT_DONE      -> Python "PHASE6_FRONT_DONE"
4 PHASE6_H_DISPATCH_DONE -> Python "PHASE6_H_DISPATCH_DONE"
5 PHASE6_HO_WAIT         -> Python "PHASE6_HO_WAIT"
6 FIXED_TIMING_COUNT     -> 哨兵，不导出
```

`PHASE6_HO_WAIT` 只包围既有串行 SyncAll；重叠分支不新增同样的全核同步，
其 ready 等待由 B 的 H/O 动态事件观察。总窗口在计时器内部初始化完成后开始。

最终方案 A 不启用动态类型，不预留动态计数或循环槽。以下 B 枚举只保留为未实施备选。
B 动态枚举与 Python 字典依次为：

```text
 0 DYN_PREP_PACK_BEGIN    1 DYN_PREP_PACK_END
 2 DYN_PREP_WAIT_INPUT    3 DYN_PREP_WAIT_REUSE
 4 DYN_H_CHUNK_BEGIN      5 DYN_H_CHUNK_END
 6 DYN_H_WAIT_STATE       7 DYN_H_WAIT_P
 8 DYN_H_WAIT_RIGHT       9 DYN_H_WAIT_D
10 DYN_O_GROUP_BEGIN    11 DYN_O_GROUP_END
12 DYN_O_WAIT_STAGE2    13 DYN_O_WAIT_STAGE3
14 DYN_O_WAIT_STAGE4    15 DYN_O_WAIT_REUSE
16 DYNAMIC_TYPE_COUNT_ENUM（哨兵）
```

上述 B 类型未落地。A 的固定枚举已经替换上游占位符，解析器从共享 C++ 头派生命名与尺寸。
对不存在的分支不设置 IterCount、不制造零值伪事件；解析器隐藏未启用类型。
每个实际启用类型记录实际覆盖计数，超过 `MAX_DYNAMIC_ITER` 的样本明确标记截断。
拟先保留 64 个本核迭代，导出 retained/total 计数和 task 关联；若要观测稳态或尾部，
再增加可配置采样区间，不能把前 64 次误当成全序列。

## 4. host/kernel/Python 接入边界

1. 依照 skill 使用 int64、ND、contiguous、与主输入相同 NPU device 的 timer 缓冲。
   长度、对齐和枚举由同一布局定义推导，不能在 host/UT 中各写一个魔数。
2. 单独为 Prepare/H/O/Phase6 划分缓冲 slice，kernel identity 写入导出 metadata。
   多 kernel 共享同一组槽会把上一 kernel 的事件覆盖，不能直接照搬单 kernel 示例。
3. 拟提供独立诊断调用入口转发既有 host 实现并传入 timer，正常 Stable-ABI schema、
   ACLNN 函数签名及十项返回契约保留。内部 L0、OpDef 和 kernel 需一致传递可选 timer。
   不把 timer 插入已有公开 ACLNN ABI 中间，也不从 raw data-pointer/临时环境变量偷偷传递。
   新诊断入口的签名和可选输入差异在实施前按阶段 01 归档确认，复用现有 CPU 标杆验收数值行为。
4. GDN 新路径传同一个完整 timer 给 Prepare/H/O，由各 kernel 按 slice ID 计算独占地址；独立公开子算子调用传空 timer。
   因此影响范围至少包含完整 GDN 前向、Prepare、ChunkFwdH、ChunkFwdO 与对应调用合同测试。
5. 核映射与实际 MIX blockDim 匹配；不能未经核验沿用模板的 `N_CORE_COUNT=96`。
   核容量、缓冲布局版本、编译 SoC、实际 blockDim 写 metadata，host/kernel/parser 使用一致容量。
6. 修正目标副本中的开关：使用可外部定义的默认值与 `#if ENABLE_TIMER`；
   上游的 `#define ENABLE_TIMER 1` + `#ifdef ENABLE_TIMER` 会使 `ENABLE_TIMER=0` 仍执行打点。
7. 计时基准优先复用原有同步。不得在条件执行的 H/O、空任务或提前返回之后加入 SyncAll。
   如需入口统一基准，只能在所有实际核都会到达的入口建立，并通过有/无此同步的 A/B 校验；
   主体内部不为打点新增跨核屏障。异步阶段按时刻点法观测。
8. parser 的 cycles/us 必须按目标设备/时钟证据配置并写 metadata；
   未确认时先保留原始 cycle，不盲用模板的 `1000.0` 报告微秒。
9. 每次独立采集先清零，单次调用后同步，再导出 raw、逐核 CSV、Perfetto JSON。
   `Accumulate` 是合成区间，只放在 duration 汇总，不作为真实重叠时间线色块。
10. 原模板具有 MIT 许可；定制副本交付时保留来源和必要许可说明。

## 5. 验证计划与当前结果

| 检查 | 当前结果 / 后续判据 |
| --- | --- |
| SSH 克隆 | 成功，固定提交见第 1 节 |
| 原工作树 | 操作开始时 `git status --short` 为空；计时框架原本不存在 |
| 工程完整性 | 有 AscendC kernel、ACLNN host、Stable-ABI Python 适配、构建和 ATK 通道 |
| 四模板接入 | 计时头共享于 `ascendc/common/kernel_utils/gdn_timer/`，目标 op_kernel 保留转发头；定制 parser/trace 与采集脚本在 `../tests/ut/` |
| 离线原有 ABI 合同 | `python tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py`，退出码 1：`prepare kernel 缺少非 exp2 模式的局部指数换底选择`；发生于任何算子调用点修改之前 |
| 构建环境预检 | `python scripts/check_npu_env.py --build-only` 失败；当前为 Windows，缺 bash、CANN 环境、cmake/gcc/g++/make/patch/bisheng 等 |
| 编译与最小 NPU 调用 | 用户明确本轮仅源码；生产环境不可接入，均未执行 |
| 数据与耗时结论 | 源码已接入；无真实 raw/CSV/trace，不报告设备性能或正式验收通过 |
| 离线合同测试 | 新计时套件 20 项、现有前向消费合同 2 项通过；详细范围见 UT README |

实施后依次验证：

- 编译/安装本轮产物并确认实际加载的 OPP；接口/适配层修改使用完整 wheel 路线。
- timer dtype/device/长度/连续性异常校验，布局版本与枚举/parser 一致。
- 最小合法样本：短序列、跨 chunk、尾块、GVA、初态/final_state、单 head 特化、两条 host 分支。
- 不同 kernel slice 不覆盖，AIC/AIV 不越界；A 无动态计数与循环采样。
- enabled/disabled 输出与已有标杆一致；raw 长度准确、有效 duration 非负、事件位于对应 kernel 总窗口。
- 在同一输入/预热/采样条件下检查开关开销。正式性能口径仍遵循仓库 `msprof op_summary`，
  内部 cycle 用于阶段定位，不替代 Task Duration 的达标结论。
- 正式受影响算子的 ATK 与 GDN Example/ST 按仓库阶段 05 验收。
- skill Step 9：实现完成后，由 fresh context 的独立子 agent 对照本图、选定方案和代码核对
  Tik/Tok 配对、枚举、循环轴、分支、完成锚点与解析器映射；问题修正后重新校验。

skill Step 3.4 原文要求：“用户选定一个方案（或要求组合 / 修订）后方可进入 Step 4 起实施。”
用户已经选择 A；其后明确本轮仅要求源码打点，生产环境不能接入，因此本次不执行设备验收。


## 6. 最终落地映射（方案 A）

原始图保留在第 2 节；本节仅给出最终观察节点，与第 3 节 A 图逐一对应。

| 流程图节点 | 实现位置 | 实际打点 |
| --- | --- | --- |
| Prepare Init/pack/尾 drain | Prepare raw entry | timer.Init(PREPARE) → op.Init → InitDone → op.Process → Finish |
| H 参数/服务初始化与 work-unit/chunk | ChunkFwdH raw entry + RunFwdHTyped | timer.Init(FWD_H)；AIC cube.Init/AIV vec.Init 后 InitDone；idle AIC 返回前 InitDone；外层 Finish |
| O Init/groupTask/head | ChunkFwdO raw entry + Impl/A5 Process | timer.Init(FWD_O)；旧服务或 A5 cube/vector Init 后 InitDone；外层 Finish |
| Phase6 tiling/workspace/front/H/O | 两个架构的 RunPhase6 | timer.Init(PHASE6)；首 KKT 前 InitDone；私有 H 前 FRONT_DONE；H 返回后 H_DISPATCH_DONE；末尾 Finish |
| Phase6 串行 H/O SyncAll | 两个架构原串行分支 | HoWaitBegin → 原 SyncAll → HoWaitEnd；重叠分支不记录该槽 |

共同配对实现是 GdnKernelTimer：all Tik → Finish Tok，stage INIT Tik → InitDone Tok，
BODY Tik → Finish 共享总窗口 endCycle；boundary 的 Point 是紧邻 Tik/Tok 对，HoWait 是原同步前后对。
KERNEL_TIMING 在 timer 内部绑定后开始，测业务主流程窗口；不包含 entry 分派、计时绑定和 host 辅助任务。
INIT 是 issue window，两个 Point 是 dispatch boundary，只有总末尾增加局部 PIPE_ALL 排空。
ENABLE_TIMER=0 或 timer=null 时没有计时 GM 写入与新增 barrier。

布局为四个 slice、六对固定事件、256 个容量核组，每核三个成员、行 stride=16 word。
每个 slice 16-word header + 12,288-word 事件区，总计 49,216 个 int64。
解析器读取共享头，拒绝坏版本/溢出/缺核/越窗/未配对/不完整路由。
未核验周期频率和共同时钟时不生成微秒时间轴；Perfetto 为每个 kernel 单独 origin，不表示调用间隔。
