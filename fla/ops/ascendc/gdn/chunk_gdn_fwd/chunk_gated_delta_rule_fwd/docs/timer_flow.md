# GDN 原始流程与计时方案

基线：main 分支 `90176b44321ebdbf3ae2a9057243157d7085c6e1`。本次仅增加诊断计时；公式、task 划分、原有 event/flag、工作区和输出数值沿用基线。公开算子入口为 ChunkGatedDeltaRuleFwd；其内部覆盖 Arch22/Arch35 Phase6 和 A5 Prepare/H/O 两条路径。

## 改造前执行流程

下图为原始业务调用链，计时仅附着在节点边界。`动态`表示每核的 task/batch/head/chunk 或 tile 循环；`固定`表示每次 kernel 启动一次。下发窗口的返回不等于异步运算完成。

```text
ACLNN GetWorkspaceSize（host，固定）
  校验布局/属性 → 连续化/transpose/cast → 按 SoC/dtype/gate 选择路由
  ├─ Phase6（Arch22/Arch35，一次混合 kernel）
  │  tiling/key → workspace 地址、实际 blockDim
  │  ├─ AIC KKT = K Kᵀ（动态，KKT service，Fixpipe 下发）
  │  └─ AIV cumsum/gate（动态，prepared 为独立 service；fused 合入 KKT epilogue）
  │  原 SyncAll 或 SCORE_READY → AIV KKT gate/下三角 mask（动态）
  │  solve（动态，AIC/AIV，保持原内部 event/flag）
  │  原 solve-done 握手 → W/U（动态，recompute service）
  │  Arch22 front-batch 循环交错 KKT epilogue / solve / WU / ACK_WAIT
  │  H（动态，head/chunk/tile 循环）
  │    AIC WH → AIV vNew = U-WH → AIC KV → AIV 状态更新
  │    各步保持原 ready/free、跨核等待、tail 和 idle-producer 路由
  │  serial：原 H/O SyncAll；pipeline：原每 chunk ready handshake
  │  O（动态，head/chunk/tile 循环）
  │    AIC QK、QH → AIV causal mask/gate → AIC Attn×vNew → AIV 输出融合
  │    原 pipeline 收尾 SyncAll（仅 enabled 分支）
  └─ A5 Prepare/H/O（三次混合 kernel）
     Prepare pack/task 循环：
       AIV gate/cumsum、qk norm → AIC KKT → AIV 下三角系数
       AIC solve S4/S5 → AIV WU 输入 → AIC W、U → AIV 写回
       每段沿用原完成/复用信号，issue 与 drain 分开观测
     ChunkFwdH：WH → vNew → KV → state update（head/chunk/slot）
     ChunkFwdO：QK/QH → mask/gate → Attn×vNew → output fuse（head round）
ACLNN launch（host，固定） → stream 同步（采集端，位于性能循环外）
```

## 计时契约与模式

四个 launch slice 分别为 PHASE6/PREPARE/FWD_H/FWD_O。每个物理组各有 AIC、AIV0、AIV1 三个独立行；行和 header 独占 64B cache line。预留 group 容量为 256，实际 blockDim 写入 header；越界拒绝解析。总窗口 KERNEL_TIMING、MAIN_INIT、MAIN_BODY 保留；MAIN_INIT 是初始化下发窗口，kernel 末尾仅诊断调用增加一次本核 PIPE_ALL drain。所有细分阶段均 TikNoBarrier/TokNoBarrier，不新增业务同步。

动态项以每核、每阶段真实调用次序为 iter 轴（0 起），同一 head/chunk 的不同 tile 不覆盖。每阶段采样前 32 次，计数保存未截断的实际调用次数；超过容量在 metadata 明确标记。每槽单写 Overwrite；不合成绝对时刻、不使用 Accumulate。异步操作没有独立完成锚点时标为 issue window，不能解释为纯计算耗时。现有等待独立计时；各窗口可重叠，不相加当总耗时。跨 launch 各自 origin，不推断 launch 间隔。

动态打点使用 GDN_TIMER_SCOPE：进入作用域调用 Begin → setDynamicActualIter → TikNoBarrier；析构调用 End → TokNoBarrier<Overwrite>，所有提前 return 同样配对。无 timer 或 ENABLE_TIMER=0 不读时钟、不写 GM、不 drain。服务用 SetTimerPtr 传递同一 kernel 的计时上下文。

最终逐点清单在完成插桩后附于本文；每一项列出源文件、代码段、阶段、核型、循环轴和锚点含义，并接受独立 agent 的逐项交叉校验。

## 原图节点与观测名称

原图 KKT → KKT_MM/PREPARE_KKT_GATE/KKT_EPILOGUE；cumsum → prepared 路由的 CUMSUM；Arch22 fused cumsum 随 KKT_EPILOGUE 合窗（Prepare 的 cumsum 与 norm/gate 同在 PREPARE_NORM_GATE）；solve → SOLVE/PREPARE_SOLVE_PREFILL/PREPARE_SOLVE_ISSUE/PREPARE_SOLVE_DUMP/SOLVE_WAIT；WU → WU_PREPARE/WU_WAIT/W_MM/U_MM；H → H_WH/H_VNEW/H_KV/H_STATE_UPDATE/H_WAIT；O → O_QK/O_QH/O_GATE/O_MASK/O_ATTN_V/O_OUTPUT/O_WAIT。PREPARE_KKT_GATE 观察 GateLowerLVF 两个 Exp 分支；PREPARE_SOLVE_PREFILL 按 t 记录异步 I@I 预填充；O_GATE 单独观察 A5 预建 gate；O_MASK 覆盖其 full/tail VF 和其余融合 mask/gate epilogue；front-batch ACK → FRONT_WAIT；原全核会合 → PHASE_WAIT。

每条动态 scope 的构造即 Tik，作用域结束/提前退出时析构即 Tok，两端来自同一对象、同一阶段与 iter。下表每一行都是一个完整配对。原始 SyncAll/CrossCoreWaitFlag/WaitFlag 顺序由离线源码检查与 main 比对；等待段以已有 wait 返回为锚点，其余均按 issue window 解释，含该调用内的原有搬运、等待或计算，未额外排空。具体等待对象可在代码段查看。

融合 H 的 blockTokens<16 路由可由 AIV 的 ComputeTailVWorkspace/ComputeTailHWorkspace 替代 WH/KV；这两个 helper 也分别打 H_WH/H_KV，清单明确标注 AIV。A5 Recompute 的背靠背 blockMmad U/W 及原 ring-ready/free 等待也分别观测。

对于 Phase6 KKT_MM 的 Process 调用、非 batch solve 路由与 WU_PREPARE 的 Process，单窗可能包含本核多个 task；它们的 iter 是服务调用序号。H/O matmul 与 epilogue 则是 head/chunk/tile 内的实际操作调用序号。各阶段 iter 不能当成相同 task id 对齐；需要由原调度顺序分析。Prepare 的 W/U 窗仅计 WuMatmulToL0C 下发，后续 M_FIX 等待在原位置；它们不是完成时延。

固定项：KERNEL_TIMING_IDX（导出 KERNEL_TIMING）是 Init 到 Finish 的总窗口；MAIN_INIT 是 Init 到 InitDone 的 issue window；MAIN_BODY 是 InitDone 到同一个总窗口结束戳；PHASE6_FRONT_DONE 与 PHASE6_H_DISPATCH_DONE 是相邻 Tik/Tok 的瞬时点；PHASE6_HO_WAIT 是串行 H/O 交接的已有 SyncAll 窗口。前三项四个 launch 都记录；后三项仅 Phase6。Finish 后只发布本核 timer 行与计数行，header 仅 AIC0 发布。

## 逐点配对清单

固定项 6，动态类型 28。下表所有项均为动态、Overwrite、NoBarrier；iter 为上文定义的每核每阶段调用序号。wait 类型记录已有等待返回，其余是含原同步的调用 issue window。源文件的所属服务以目录和代码段为准。

| 阶段 / 原理节点 | 文件:行（Tik/Tok scope） | 代码段 | 核型 | 锚点 / 含义 |
| --- | --- | --- | --- | --- |
| H_WH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_cube.h:322` | `const uint32_t slot = head.roundHead;` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_cube.h:384` | `AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_cube.h:391` | `AscendC::CrossCoreWaitFlag<0x2, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_KV | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_cube.h:490` | `const FwdHKgBinding binding = FwdHBuildKgBinding(unit.headRound, head.kgSlot);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_cube.h:570` | `AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_cube.h:576` | `AscendC::CrossCoreWaitFlag<0x2, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_cube.h:620` | `AscendC::CrossCoreWaitFlag<0x2, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_cube.h:627` | `AscendC::CrossCoreWaitFlag<0x2, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_cube.h:635` | `AscendC::CrossCoreWaitFlag<0x2, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_VNEW | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_vec.h:354` | `const uint32_t slot = head.localSlot;` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_vec.h:457` | `AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(` | AIV | 原 wait 返回；等待窗口 |
| H_STATE_UPDATE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_vec.h:587` | `const uint32_t slot = head.localSlot;` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_vec.h:725` | `AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(` | AIV | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_vec.h:759` | `AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(` | AIV | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_vec.h:773` | `AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(` | AIV | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_vec.h:788` | `AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(FwdHAivLocalFlag(FWD_H_ROUND_ACK_FLAG, 0));` | AIV | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:355` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(` | AIC | 原 wait 返回；等待窗口 |
| H_WH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:375` | `const uint32_t m = FwdHAlignCube(chunk.validTokens);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:417` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:425` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:433` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:525` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(` | AIC | 原 wait 返回；等待窗口 |
| H_KV | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:563` | `const uint32_t m = FWD_H_K;` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:628` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:636` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:644` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:786` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:792` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:803` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:809` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:819` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_S>(` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:824` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_S>(` | AIC | 原 wait 返回；等待窗口 |
| H_VNEW | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h:647` | `const uint32_t slot = DataSlot(head);` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h:673` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(` | AIV | 原 wait 返回；等待窗口 |
| H_STATE_UPDATE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h:873` | `static_assert(!RESIDENT_FP32_STATE // CompilePolicy::STATE_FP32,` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h:879` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(` | AIV | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h:1143` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(` | AIV | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h:1180` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(` | AIV | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h:1200` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(` | AIV | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h:1217` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(` | AIV | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h:1233` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_S>(FwdHAivLocalFlag(FWD_H_ROUND_ACK_FLAG, 0));` | AIV | 原 wait 返回；等待窗口 |
| O_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_cube.h:170` | `Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);` | AIC | 原 wait 返回；等待窗口 |
| O_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_cube.h:182` | `Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);` | AIC | 原 wait 返回；等待窗口 |
| O_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_cube.h:195` | `Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);` | AIC | 原 wait 返回；等待窗口 |
| O_ATTN_V | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_cube.h:279` | `TileMmad{}(tileL0C, tileL0A, tileL0B, kBt, kV, kBt, true, 0);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_ATTN_V | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_cube.h:298` | `TileMmad{}(tensorL0C, tensorL0A, tensorL0B, true, 0);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_cube.h:386` | `Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);` | AIC | 原 wait 返回；等待窗口 |
| O_QK | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_cube.h:442` | `QkTileMmad{}(tileQktL0C, tileQktL0A, tileQktL0B, m, m, kK, true, 0);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_QH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_cube.h:508` | `QhTileMmad{}(tileQhL0C, tileQhL0A, tileQhL0B, m, kV, kK, true, 0);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_vector.h:333` | `Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);` | AIV | 原 wait 返回；等待窗口 |
| O_MASK | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_vector.h:339` | `AscendC::VF_CALL<Stage3Gate64VF<true>>(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| O_MASK | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_vector.h:350` | `AscendC::VF_CALL<Stage3Gate64VF<false>>(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| O_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_vector.h:380` | `Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);` | AIV | 原 wait 返回；等待窗口 |
| O_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_vector.h:394` | `Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);` | AIV | 原 wait 返回；等待窗口 |
| O_OUTPUT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_vector.h:410` | `AscendC::VF_CALL<Stage5Fuse64VF>(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| O_GATE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_vector.h:464` | `const uint32_t bt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| O_QK | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/gemm/kernel/gdn_fwd_o_kernel.hpp:321` | `blockMmadQK(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_QH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/gemm/kernel/gdn_fwd_o_kernel.hpp:434` | `blockMmadQH128ToUB.executeCompute(tensorBlockHWorkUb, cube2Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_QH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/gemm/kernel/gdn_fwd_o_kernel.hpp:440` | `blockMmadQH256.executeCompute(tensorBlockHWork, cube2Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_QH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/gemm/kernel/gdn_fwd_o_kernel.hpp:448` | `blockMmadQH128.executeCompute(tensorBlockHWork, cube2Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_QH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/gemm/kernel/gdn_fwd_o_kernel.hpp:454` | `blockMmadQH256.executeCompute(tensorBlockHWork, cube2Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_ATTN_V | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/gemm/kernel/gdn_fwd_o_kernel.hpp:499` | `blockMmadAttenVNEW128ToUB.executeCompute(tensorBlockVWorkUb, cube3Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_ATTN_V | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/gemm/kernel/gdn_fwd_o_kernel.hpp:506` | `blockMmadAttenVNEW256.executeCompute(tensorBlockVWork, cube3Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_ATTN_V | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/gemm/kernel/gdn_fwd_o_kernel.hpp:515` | `blockMmadAttenVNEW128.executeCompute(tensorBlockVWork, cube3Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_ATTN_V | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/gemm/kernel/gdn_fwd_o_kernel.hpp:522` | `blockMmadAttenVNEW256.executeCompute(tensorBlockVWork, cube3Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_MASK | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/gemm/kernel/gdn_fwd_o_kernel.hpp:603` | `epilogueGDNFwdOQkmask(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| O_OUTPUT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/gemm/kernel/gdn_fwd_o_kernel.hpp:639` | `epilogueGDNFwdOOutput(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| O_OUTPUT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/gemm/kernel/gdn_fwd_o_kernel.hpp:650` | `epilogueGDNFwdOOutput(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| O_OUTPUT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/gemm/kernel/gdn_fwd_o_kernel.hpp:662` | `epilogueGDNFwdOOutput(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| O_QK | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:289` | `blockMmadQK(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_QH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:317` | `blockMmadQH128(tensorBlockQ, tensorBlockH, tensorBlockHWork, cube2Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_QH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:324` | `blockMmadQH256(tensorBlockQ, tensorBlockH, tensorBlockHWork, cube2Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_ATTN_V | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:351` | `blockMmadAttenVNEW128(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_ATTN_V | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:358` | `blockMmadAttenVNEW256(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_MASK | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:403` | `epilogueGDNFwdOQkmask(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| O_OUTPUT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:425` | `epilogueGDNFwdOOutput(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| KKT_EPILOGUE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:353` | `AscendC::TPipe pipe;` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| FRONT_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:418` | `AscendC::CrossCoreWaitFlag(SCORE_READY_FLAG);` | AIV | 原 wait 返回；等待窗口 |
| FRONT_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:434` | `AscendC::CrossCoreWaitFlag(FRONT_ACK_DONE_FLAG);` | AIV | 原 wait 返回；等待窗口 |
| FRONT_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:440` | `AscendC::CrossCoreWaitFlag(FRONT_READY_FLAG);` | AIC | 原 wait 返回；等待窗口 |
| SOLVE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:448` | `GdnFp32Solve::RunOwnedBatch<InputT>(` | AIC/AIV | NoBarrier 调用/下发窗口；原同步保留 |
| FRONT_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:478` | `AscendC::CrossCoreWaitFlag(FRONT_ACK_DONE_FLAG);` | AIV | 原 wait 返回；等待窗口 |
| KKT_MM | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:518` | `kktCube.Process(k, cuSeqlens, chunkIndices, scoreWorkspace, &abc);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| CUMSUM | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:536` | `prepare.ProcessMixed();` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| PHASE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:545` | `AscendC::SyncAll<false>();` | AIC/AIV | 原 wait 返回；等待窗口 |
| SOLVE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:581` | `AscendC::CrossCoreWaitFlag(SCORE_READY_FLAG);` | AIV | 原 wait 返回；等待窗口 |
| KKT_EPILOGUE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:585` | `kkt.ProcessEpilogueForSolve(abc.tilesPerCore);` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| PHASE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:597` | `AscendC::SyncAll<false>();` | AIC/AIV | 原 wait 返回；等待窗口 |
| PHASE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:603` | `AscendC::SyncAll<false>();` | AIC/AIV | 原 wait 返回；等待窗口 |
| SOLVE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:610` | `RunSolvePhase<InputT, 64>(tndInput, cuSeqlens, chunkIndices,` | AIC/AIV | NoBarrier 调用/下发窗口；原同步保留 |
| PHASE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:616` | `AscendC::SyncAll<false>();` | AIC/AIV | 原 wait 返回；等待窗口 |
| PHASE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:622` | `AscendC::SyncAll<false>();` | AIC/AIV | 原 wait 返回；等待窗口 |
| PHASE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:630` | `AscendC::SyncAll<false>();` | AIC/AIV | 原 wait 返回；等待窗口 |
| SOLVE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:634` | `RunSolvePhase<InputT, 64>(aWorkspace, cuSeqlens, chunkIndices, A,` | AIC/AIV | NoBarrier 调用/下发窗口；原同步保留 |
| SOLVE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:640` | `RunSolvePhase<InputT, 128>(aWorkspace, cuSeqlens, chunkIndices, A,` | AIC/AIV | NoBarrier 调用/下发窗口；原同步保留 |
| SOLVE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:652` | `AscendC::CrossCoreWaitFlag(PHASE6_SOLVE_DONE_FLAG);` | AIV | 原 wait 返回；等待窗口 |
| PHASE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:703` | `AscendC::SyncAll<false, PHASE6_HO_SYNC_CONFIG>();` | AIC/AIV | 原 wait 返回；等待窗口 |
| PHASE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:709` | `AscendC::SyncAll<false>();` | AIC/AIV | 原 wait 返回；等待窗口 |
| PHASE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:740` | `AscendC::SyncAll<false>();` | AIC/AIV | 原 wait 返回；等待窗口 |
| O_QK | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:389` | `blockMmadQK(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_QH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:417` | `blockMmadQH128(tensorBlockQ, tensorBlockH, tensorBlockHWork, cube2Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_QH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:424` | `blockMmadQH256(tensorBlockQ, tensorBlockH, tensorBlockHWork, cube2Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_ATTN_V | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:452` | `blockMmadAttenVNEW128(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_ATTN_V | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:459` | `blockMmadAttenVNEW256(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_MASK | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:505` | `epilogueGDNFwdOQkmask(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| O_OUTPUT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:533` | `epilogueGDNFwdOOutput(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| H_WH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:407` | `uint32_t subBlockIdx = AscendC::GetSubBlockIdx();` | AIV tail 替代 | NoBarrier 调用/下发窗口；原同步保留 |
| H_KV | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:481` | `uint32_t subBlockIdx = AscendC::GetSubBlockIdx();` | AIV tail 替代 | NoBarrier 调用/下发窗口；原同步保留 |
| H_WH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:661` | `blockMmadWHTail(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_WH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:670` | `blockMmadWH(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_KV | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:712` | `blockMmadKVTail(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_KV | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:721` | `blockMmadKV(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_VNEW | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:885` | `epilogueGDNFwdHVnew(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| H_STATE_UPDATE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:925` | `epilogueGDNFwdHUpdate(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| U_MM | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_cube.h:199` | `BlockMmadU(tensorBlockA, tensorBlockVb, tensorBlockU, actualBlockShape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| W_MM | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_cube.h:269` | `BlockMmadW(tensorBlockA, tensorBlockKbgExp, tensorBlockW, actualBlockShape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| WU_PREPARE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_vector.h:138` | `taskRange = range;` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| KKT_MM | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:357` | `kktCube.Process(k, cuSeqlens, chunkIndices, scoreWorkspace, &coefficient);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| CUMSUM | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:363` | `RunPhase6Cumsum(rawG, cuSeqlens, chunkIndices, gCumsumBht, coefficient);` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| PHASE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:373` | `AscendC::SyncAll<false>();` | AIC/AIV | 原 wait 返回；等待窗口 |
| KKT_EPILOGUE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:385` | `kkt.ProcessEpilogueForSolve(coefficient.tilesPerCore);` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| SOLVE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:393` | `RunSolvePhase<InputT, 64, Variant>(aWorkspace, cuSeqlens, chunkIndices, A,` | AIC/AIV | NoBarrier 调用/下发窗口；原同步保留 |
| SOLVE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:399` | `RunSolvePhase<InputT, 64, Variant>(aWorkspace, cuSeqlens, chunkIndices, A,` | AIC/AIV | NoBarrier 调用/下发窗口；原同步保留 |
| SOLVE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:405` | `RunSolvePhase<InputT, 128, Variant>(aWorkspace, cuSeqlens, chunkIndices, A,` | AIC/AIV | NoBarrier 调用/下发窗口；原同步保留 |
| SOLVE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:419` | `AscendC::CrossCoreWaitFlag(PHASE6_SOLVE_DONE_FLAG);` | AIV | 原 wait 返回；等待窗口 |
| SOLVE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:435` | `AscendC::CrossCoreWaitFlag(PHASE6_SOLVE_AIV_DONE_FLAG);` | AIC | 原 wait 返回；等待窗口 |
| SOLVE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:444` | `AscendC::CrossCoreWaitFlag(PHASE6_SOLVE_AIC_ALL_DONE_FLAG);` | AIC | 原 wait 返回；等待窗口 |
| SOLVE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:451` | `AscendC::CrossCoreWaitFlag(PHASE6_SOLVE_DONE_FLAG);` | AIV | 原 wait 返回；等待窗口 |
| PHASE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:492` | `AscendC::SyncAll<false, PHASE6_HO_SYNC_CONFIG>();` | AIC/AIV | 原 wait 返回；等待窗口 |
| PHASE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:498` | `AscendC::SyncAll<false>();` | AIC/AIV | 原 wait 返回；等待窗口 |
| PHASE_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:517` | `AscendC::SyncAll<false>();` | AIC/AIV | 原 wait 返回；等待窗口 |
| O_QK | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:408` | `blockMmadQK(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_QH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:515` | `blockMmadQH128.executeCompute(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_QH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:522` | `blockMmadQH256.executeCompute(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_ATTN_V | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:558` | `blockMmadAttenVNEW128.executeCompute(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_ATTN_V | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:566` | `blockMmadAttenVNEW256.executeCompute(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_QK | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:601` | `blockMmadQK(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_QH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:629` | `blockMmadQH128(tensorBlockQ, tensorBlockH, tensorBlockHWork, cube2Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_QH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:636` | `blockMmadQH256(tensorBlockQ, tensorBlockH, tensorBlockHWork, cube2Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_ATTN_V | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:664` | `blockMmadAttenVNEW128(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_ATTN_V | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:671` | `blockMmadAttenVNEW256(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| O_MASK | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:720` | `epilogueGDNFwdOQkmask(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| O_OUTPUT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp:762` | `epilogueGDNFwdOOutput(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| H_WH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp:498` | `uint32_t subBlockIdx = AscendC::GetSubBlockIdx();` | AIV tail 替代 | NoBarrier 调用/下发窗口；原同步保留 |
| H_KV | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp:578` | `uint32_t subBlockIdx = AscendC::GetSubBlockIdx();` | AIV tail 替代 | NoBarrier 调用/下发窗口；原同步保留 |
| H_WH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp:722` | `blockMmadWHDirectUb(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_WH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp:763` | `blockMmadWHTail(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_WH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp:772` | `blockMmadWH(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_WH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp:815` | `blockMmadWH(tensorBlockW, tensorBlockH, tensorBlockV, cube1Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_KV | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp:864` | `blockMmadKVDirectUb(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_KV | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp:917` | `blockMmadKVTail(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_KV | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp:926` | `blockMmadKV(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_KV | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp:970` | `blockMmadKV(tensorBlockK, tensorBlockVwork, tensorBlockHwork, cube2Shape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp:986` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(DIRECT_UB_FREE_FLAG_BEGIN + slot);` | AIC | 原 wait 返回；等待窗口 |
| H_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp:990` | `AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(` | AIC | 原 wait 返回；等待窗口 |
| H_VNEW | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp:1160` | `epilogueGDNFwdHVnew(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| H_STATE_UPDATE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp:1202` | `epilogueGDNFwdHUpdate(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| H_WH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:357` | `uint32_t subBlockIdx = AscendC::GetSubBlockIdx();` | AIV tail 替代 | NoBarrier 调用/下发窗口；原同步保留 |
| H_KV | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:409` | `uint32_t subBlockIdx = AscendC::GetSubBlockIdx();` | AIV tail 替代 | NoBarrier 调用/下发窗口；原同步保留 |
| H_WH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:573` | `blockMmadWHTail(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_WH | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:582` | `blockMmadWH(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_KV | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:624` | `blockMmadKVTail(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_KV | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:633` | `blockMmadKV(` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| H_VNEW | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:777` | `epilogueGDNFwdHVnew(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| H_STATE_UPDATE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp:817` | `epilogueGDNFwdHUpdate(` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| WU_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_cube.h:189` | `Arch::CrossCoreWaitFlag(flagAivVbReady);` | AIC | 原 wait 返回；等待窗口 |
| U_MM | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_cube.h:213` | `blockMmad(tensorBlockA, tensorBlockVb, tensorBlockU, actualBlockShape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| WU_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_cube.h:219` | `Arch::CrossCoreWaitFlag(flagAivKbgExpReady);` | AIC | 原 wait 返回；等待窗口 |
| W_MM | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_cube.h:241` | `blockMmad(tensorBlockA, tensorBlockKbgExp, tensorBlockW, actualBlockShape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| U_MM | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_cube.h:313` | `BlockMmadU(tensorBlockA, tensorBlockVb, tensorBlockU, actualBlockShape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| W_MM | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_cube.h:370` | `BlockMmadW(tensorBlockA, tensorBlockKbgExp, tensorBlockW, actualBlockShape);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| WU_PREPARE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_vector.h:172` | `#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| WU_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_vector.h:207` | `Arch::CrossCoreWaitFlag(flagAicSlotFree);` | AIV | 原 wait 返回；等待窗口 |
| FRONT_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:324` | `WaitAicStage4Done(taskIdx);` | AIC/AIV | 原 wait 返回；等待窗口 |
| FRONT_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:345` | `WaitAicStage4Done(taskIdx);` | AIC/AIV | 原 wait 返回；等待窗口 |
| FRONT_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:365` | `WaitAicStage4Done(taskIdx);` | AIC/AIV | 原 wait 返回；等待窗口 |
| PREPARE_NORM_GATE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:422` | `const int32_t db = PingPongSlot(taskIdx);` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| KKT_MM | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:481` | `const int32_t bt = static_cast<int32_t>(chunkSize);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| PREPARE_KKT_GATE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:506` | `const int32_t db = PingPongSlot(taskIdx);` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| KKT_EPILOGUE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:518` | `WaitCubeKktDone(taskIdx);` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| PREPARE_SOLVE_PREFILL | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:554` | `MmadL0ABToL0C<float>(l0Af[0], l0Bf[0], l0CTask[t], bt, bt, bt, true, bank);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| PREPARE_SOLVE_ISSUE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:566` | `const int32_t bt = static_cast<int32_t>(chunkSize);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| PREPARE_SOLVE_DUMP | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:578` | `const uint8_t bank = static_cast<uint8_t>(taskIdx & 1);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| SOLVE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:598` | `const int32_t bt = static_cast<int32_t>(chunkSize);` | AIC/AIV | NoBarrier 调用/下发窗口；原同步保留 |
| PREPARE_SOLVE_DUMP | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:626` | `const uint8_t bank = static_cast<uint8_t>(taskIdx & 1);` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| WU_PREPARE | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:656` | `const int64_t hk = hv / HRatio;` | AIV | NoBarrier 调用/下发窗口；原同步保留 |
| W_MM | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:731` | `WuMatmulToL0C<InDtype>(l1A[taskIdx], l1Kbg[taskIdx], l0AS7[0], l0BS7[0], l0CS7[0], bt, static_cast<int32_t>(nK` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| U_MM | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:745` | `WuMatmulToL0C<InDtype>(l1A[taskIdx], l1Vb[taskIdx], l0AS7[1], l0BS7[1], l0CS7[1], bt,` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| U_MM | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:763` | `WuMatmulToL0C<InDtype>(l1A[taskIdx], l1Vb1[taskIdx], l0AS7[1], l0BS7[1], l0CS7[1], bt,` | AIC | NoBarrier 调用/下发窗口；原同步保留 |
| FRONT_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:832` | `WaitAicStage7Done();` | AIC/AIV | 原 wait 返回；等待窗口 |
| FRONT_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:869` | `WaitAicStage7Done();` | AIC/AIV | 原 wait 返回；等待窗口 |
| FRONT_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:896` | `WaitAivStage1Done(t);` | AIC/AIV | 原 wait 返回；等待窗口 |
| FRONT_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:907` | `WaitAivStage3Done(0);` | AIC/AIV | 原 wait 返回；等待窗口 |
| FRONT_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:913` | `WaitAivStage3Done(t);` | AIC/AIV | 原 wait 返回；等待窗口 |
| FRONT_WAIT | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:933` | `WaitAivStage6Done(t);` | AIC/AIV | 原 wait 返回；等待窗口 |

当前共有 195 个动态 scope。每次诊断分配 5652544 个 int64（43.13 MiB）；采样 32 次/类型/核，实际调用总数另存。
