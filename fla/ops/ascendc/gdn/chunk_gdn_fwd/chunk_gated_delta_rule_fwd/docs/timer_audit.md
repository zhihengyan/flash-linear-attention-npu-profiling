# GDN timer Step 9 独立源码审核

审核日期：2026-10-12。审核者为 fresh-context 独立 agent，未参与实现，未修改业务代码；基线为 main `90176b44321ebdbf3ae2a9057243157d7085c6e1`。按仓库 `AGENTS.md`、`docs/agents/03–05` 的实现及验证路由、runtime 架构文档和 `op-timer-ascendc` Step 9 执行。

**结论：修复后独立源码审核通过，设备验收未完成。** 初审发现的实际可达运算遗漏已逐处修复并复查。最终为 6 个固定项、28 个动态类型、195 个动态 scope。没有剩余已确认的源码配对、指针传递或关键计算覆盖缺陷。本结论不能替代 CANN 编译、OPP 安装或 NPU 验证。

## 初审问题与修复复查

初审的 179 个 scope / 24 个类型虽然通过自动 inventory 检查，仍存在下列遗漏；这些问题来自阅读真实替代运算与调用链，而非只统计宏数量。下表行号为复查后的工作树。

| 初审问题及实际影响 | 最终证据 | 复查 |
| --- | --- | --- |
| A5 独立 O 的 Stage3 full/tail VF 未计时，Prepare/H/O 路由的 causal mask/gate 没有 O_MASK 事件；Stage1 gate 预建也在已有 scope 外 | `chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_vector.h:339,350` 的 O_MASK 包住 `Stage3Gate64VF<true/false>`；`:464` 的 O_GATE 覆盖 `ComputeStage1Gate` | 两个可达 VF 分支及 gate 函数均配对，未移动原 wait/set |
| Prepare 的 gate/Exp 下三角预处理在 KKT_EPILOGUE 之前，原窗口未包含它 | `chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:506` 的 PREPARE_KKT_GATE | 同一函数包含 `GateLowerLVF<true/false>`，两种 useExp2 路由都被覆盖 |
| Prepare solve S4 的 I@I 预填充在 SOLVE/ISSUE 窗口外 | 同文件 `:554` 的 PREPARE_SOLVE_PREFILL | 放在逐 t 的 `MmadL0ABToL0C` 调用处；后续 M_FIX 原等待仍在原位置，按 issue window 解释 |
| 融合 H 的小 tail 由 AIV 计算 WH/KV，AIC 跳过 matmul；原 H_VNEW/H_STATE_UPDATE 窗口从 helper 返回后才开始 | 三份融合 H header 的 `ComputeTailVWorkspace` / `ComputeTailHWorkspace` 入口分别新增 H_WH / H_KV；完整位置见下文逐点索引 | 六处 helper 已配对；核型为 AIV。Arch22 的 `<16` 路由得到观测；Arch35 保留的分支还要求 `!useBoundedMmad`，不声称该分支已在设备命中 |
| A5 Phase6 真正执行的背靠背 U/W 分支未计时，原 U_MM/W_MM 只位于非 310 分支 | `internal/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_cube.h:213,241` | `__CCE_AICORE__==310` 分支的两次 `blockMmad` 分别计 U_MM/W_MM；原 ready/free 等待额外计 WU_WAIT，不新增同步 |
| 原文把 cumsum 笼统映射为 CUMSUM，无法解释 Arch22 fused 路由没有该事件；wait 清单核型过于笼统 | `timer_flow.md` 原图、节点映射及逐行核型 | prepared→CUMSUM；fused→KKT_EPILOGUE 合窗；Prepare→PREPARE_NORM_GATE。等待项按实际 AIC/AIV 分支标注 |

上表路径均相对于 `fla/ops/ascendc/gdn/chunk_gdn_fwd/`。累计补 13 个计算 scope 和 3 个 WU_WAIT scope，179→195；新增 O_GATE、PREPARE_KKT_GATE、PREPARE_SOLVE_PREFILL、WU_WAIT，24→28。

## 原图与实际运算交叉审核

| 原理步骤 | 已人工追踪的实现与替代分支 | 观察点与边界 |
| --- | --- | --- |
| cumsum / KKT | Arch22 prepared/fused、front-batch 与非 batch；Arch35 Phase6；Prepare 的 owner/sibling、full/tail、useExp2 两分支 | CUMSUM、KKT_MM、KKT_EPILOGUE、PREPARE_NORM_GATE、PREPARE_KKT_GATE。Phase6 Process 单窗可覆盖多个 task，fused cumsum 为合窗 |
| 下三角 solve | Arch22 owned-batch、BT64 定长/变长 staging、BT128；Arch35 B0/B30 及 BT64/128；Prepare S4/S5 的预填充、累加、dump | SOLVE、SOLVE_WAIT、PREPARE_SOLVE_PREFILL/ISSUE/DUMP。未通过新增排空分开测量；dump 中已有 wait 保留 |
| W/U | Arch22 U/W 分阶段；A5 310 背靠背 U/W、V128/256 tile；Prepare S7 的 W、首 U 及可达第二 U 分支 | WU_PREPARE、WU_WAIT、W_MM、U_MM；每次实际调用独立 ordinal，不覆盖同槽 |
| H 的 WH / vNew / KV / state | 融合 Arch22/Arch35 的常规、bounded tail、direct UB、FP32 state、AIV 小 tail；独立 H Arch22/Arch35 的 scalar-g/gk、stateVFirst、单 head/多 head 与初态/末态分支 | H_WH、H_VNEW、H_KV、H_STATE_UPDATE、H_WAIT；不执行的 math 分支不伪造时间戳，helper 提前 return 仍由析构收尾 |
| O 的 QK / QH / mask / Attn×V / output | 融合 Arch22/Arch35，V128/256、split executeCompute、GM/direct UB、聚合 epilogue；独立 O 通用与 A5 token-first 路由，full/tail Stage3 VF、stateVFirst | O_QK、O_QH、O_GATE、O_MASK、O_ATTN_V、O_OUTPUT、O_WAIT。A5 专用 VF 修复后有实际观测 |
| 阶段交接 | Arch22 front-batch ACK、serial H/O 会合、idle-producer/consumer；Arch35 serial/chunk pipeline 及收尾 | FRONT_WAIT、PHASE_WAIT 与固定 dispatch point / PHASE6_HO_WAIT；保留业务 flag/wait 的原次序 |

各计算 scope 均按 NoBarrier 调用窗口解释；某些窗口含内部既有等待或搬运，某些只记录裸 MMAD 下发，不能统一解释成纯计算完成时延，也不能相加当整算子耗时。原图、类型及逐点解释以 [timer_flow.md](timer_flow.md) 为对应清单。

## 每个 Tik/Tok 的配对与参数传递

动态 scope 的统一实现位于 `common/kernel_utils/gdn_timer/AscendTimerV2_device.hpp`：构造在 Enabled 时调用 Begin，取得本核本类型唯一 ordinal，保存局部 start；析构用同一个 timer/type/iter/start 调用 End。每个 scope 只有一个构造及一个析构，禁用拷贝；块结束、函数 return 和 continue 均执行析构。195 个实际宏调用各占不同源码行，未发现同一宏局部变量重名、未收尾的 start 或无起点的 end。嵌套等待使用独立 scope 的 start，不会覆盖外围窗口。

Begin 保存未截断实际计数；仅 ordinal 0–31 读取 start/写入时间戳，其余调用返回 -1 并不执行 End。每个槽单写，Overwrite 合理，不使用 Accumulate。无 timer 时 Init 先绑定 disabled 对象并返回，Begin 不访问未初始化的 next[]；不读时钟、不写 GM、不 drain。ENABLE_TIMER=0 在宏及方法层关闭这些动作。

固定配对逐项核对如下：

| 固定项 | 起点 / 收尾 | 路由 |
| --- | --- | --- |
| KERNEL_TIMING | all_.TikNoBarrier → all_.Tok<Overwrite> | 四个实际 kernel 的 Finish；仅非空有效 timer 在末尾局部 PIPE_ALL drain |
| MAIN_INIT | stage_.TikNoBarrier → InitDone 的 TokNoBarrier | 四个 kernel；独立 H 无 Cube 工作的早退也调用 InitDone |
| MAIN_BODY | InitDone 后 stage_.TikNoBarrier → Finish 的 TokNoBarrier | 复用 all_.GetEndCycle，end 与 KERNEL_TIMING 相同 |
| PHASE6_FRONT_DONE | Point 中相邻 TikNoBarrier/TokNoBarrier | Arch22/Arch35 front 完成后、H dispatch 前；是 dispatch 时刻点 |
| PHASE6_H_DISPATCH_DONE | Point 中相邻 TikNoBarrier/TokNoBarrier | H dispatch 返回后；pipeline 时并不代表全核 H 完成 |
| PHASE6_HO_WAIT | HoWaitBegin → 原 SyncAll → HoWaitEnd | 仅 serial 分支，跨核 pipeline 分支留零 |

all_/stage_/boundary_ 分开保存固定起点；MAIN_INIT/MAIN_BODY、总窗口与 dispatch/wait 不互相覆盖。Finish 后才发布本核所有时间戳和计数。

指针人工追踪包括：Arch22 RunPhase6→RunFrontBatch/RunKktEpilogueBatch→DispatchRecompute→process→MatmulKernel；Arch22/Arch35 DispatchFwdH/DispatchFwdO→RunFwdH/RunFwdO→实际 Kernel；独立 Prepare→op，独立 H→cube/vec，独立 O 通用→GDNFwdOKernel、A5→ChunkFwdOA5→cube/vector。新增默认 GdnKernelTimer* 参数在原 idleConfig、hoReadyAddr、layout、taskRange 参数之后逐层显式传递；检查了 V128/256、gate/state dtype 和 B30 分支。未发现指针被传入 bool/layout 参数位置、子类遗漏注入或 Init 覆盖指针。

## ABI、同步、布局与 Python 采集

四个 OpDef 的 timer 是末尾 OPTIONAL INT64/ND 输入，四个 L0 OP_INPUT 顺序与实际 kernel 形参顺序一致。L0 timerOptional 位于 executor 后并以 nullptr 为默认值，原调用继续编译为普通调用路径。稳定 `aclnnChunkGatedDeltaRuleFwd*` C 声明及 Python 生产签名保持；新增独立 Profile 符号和私有 ctypes 入口。Profile 的 timer 位于 hOut 与 workspace/executor 之间，ctypes argtypes 与声明一致。host 使用共享 TOTAL_BUFFER_SIZE 校验，不硬编码尺寸；timer 不作 AutoContiguous/复制，Python 检查同设备、连续 rank-1 int64，私有 eager 入口显式增加 mutation 版本计数。

与基线相比，40 个已修改 C++ 文件的业务同步调用序列相同，人工检查了上述新增窗口附近的 wait/set 与原控制流。新增同步仅为诊断 Finish 的本核 drain，没有新增业务 SyncAll、CrossCore barrier 或逐运算排空。这个源码结论并不表示计时开销为零。

布局 v2：4 slices，每组 AIC/AIV0/AIV1；预留 256 组而非假定物理核数。当前 row stride=1808 int64，count stride=32，header=16，slice=1,413,136，总计=5,652,544 int64。header、事件行、计数行和 slice 边界均按 64B 对齐并独占 cache line。AIC0 唯一写 header；合法本核只刷新自己的事件行和计数行，超容量不重定向到 core0。DCCI 在最后两个固定结束戳写完之后执行；header 单独发布，没有整 cache 刷新或新增跨核会合。实际 CANN API 支持、MIX 核编号及 GM 可见性留待设备验证。

parser 从同一个 canonical 头安全读取枚举及整数常量，固定/动态名唯一连续，清单双向集合一致。严格拒绝错长度/metadata、缺总窗、缺配对、负 duration、超 KERNEL_TIMING、actual blockDim 外数据、padding、超 IterCount、混合 route 或缺 Prepare/H/O slice。未截断 actual 与 sampled/truncated 分别归档。trace 保持整数差值后再换算，要求显式频率和 shared_clock，跨 launch 独立 origin；默认只有 cycles。

主 collector 复用原 ATK 两个模型 case；executor 只在初始化阶段保存实际输入及完整调用参数，性能循环不增加输入导出或 timer 操作。主入口在可能初始化后端前注册当前已安装 OPP，核对 Python/ATK launcher、安装产物及 Profile 符号；额外要求 successful statistic case 和真实 msprof `Task Duration(us)` 证据。重放自行注册 OPP，warmup/普通对照/诊断在原性能循环外，raw 在输出比较及严格解码之前写盘，失败保留 raw/metadata/log。trace 仅由已确认时钟参数启用。普通/诊断精确对照可检查打点造成的行为差异，不能替代原 CPU/六算子精度验收。

## 执行证据与逐 scope 索引

从仓库根目录执行：

```text
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/check_timer_source.py > .timer_audit_source_final.json
status=source_checks_passed
baseline=90176b44321ebdbf3ae2a9057243157d7085c6e1
fixed_count=6, dynamic_count=28, scope_count=195, synchronization_files=40

python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/test_timer_contract.py
Ran 14 tests in 8.357s
OK
```

另以独立脚本反向提取 timer_flow 表格的 stage/file/line，要求清单与源码集合双向相等、无重复，并读取每个实际行核对完整 GDN_TIMER_SCOPE 语句：`independent_reverse_inventory: PASS, 195 unique stage/file/line pairs; dynamic types: 28`。逐 scope 的完整代码段、核型及锚点证据就是已核对的 [timer_flow.md 逐点配对清单](timer_flow.md#逐点配对清单)，其中每一行都判定构造 Tik 与析构 Tok 配对成立；不能用该自动一致性代替前述可达运算人工审核。

以下附录列出每个文件的全部 scope 行号与阶段，便于识别复查的准确工作树。

| Source file (relative to chunk_gdn_fwd/) | Scope count | All audited line:stage pairs |
| --- | --- | --- |
| `chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_cube.h` | 9 | `322:H_WH`, `384:H_WAIT`, `391:H_WAIT`, `490:H_KV`, `570:H_WAIT`, `576:H_WAIT`, `620:H_WAIT`, `627:H_WAIT`, `635:H_WAIT` |
| `chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_vec.h` | 7 | `354:H_VNEW`, `457:H_WAIT`, `587:H_STATE_UPDATE`, `725:H_WAIT`, `759:H_WAIT`, `773:H_WAIT`, `788:H_WAIT` |
| `chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h` | 16 | `355:H_WAIT`, `375:H_WH`, `417:H_WAIT`, `425:H_WAIT`, `433:H_WAIT`, `525:H_WAIT`, `563:H_KV`, `628:H_WAIT`, `636:H_WAIT`, `644:H_WAIT`, `786:H_WAIT`, `792:H_WAIT`, `803:H_WAIT`, `809:H_WAIT`, `819:H_WAIT`, `824:H_WAIT` |
| `chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h` | 9 | `647:H_VNEW`, `673:H_WAIT`, `873:H_STATE_UPDATE`, `879:H_WAIT`, `1143:H_WAIT`, `1180:H_WAIT`, `1200:H_WAIT`, `1217:H_WAIT`, `1233:H_WAIT` |
| `chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_cube.h` | 8 | `170:O_WAIT`, `182:O_WAIT`, `195:O_WAIT`, `279:O_ATTN_V`, `298:O_ATTN_V`, `386:O_WAIT`, `442:O_QK`, `508:O_QH` |
| `chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_vector.h` | 7 | `333:O_WAIT`, `339:O_MASK`, `350:O_MASK`, `380:O_WAIT`, `394:O_WAIT`, `410:O_OUTPUT`, `464:O_GATE` |
| `chunk_fwd_o/op_kernel/arch35/gemm/kernel/gdn_fwd_o_kernel.hpp` | 13 | `321:O_QK`, `434:O_QH`, `440:O_QH`, `448:O_QH`, `454:O_QH`, `499:O_ATTN_V`, `506:O_ATTN_V`, `515:O_ATTN_V`, `522:O_ATTN_V`, `603:O_MASK`, `639:O_OUTPUT`, `650:O_OUTPUT`, `662:O_OUTPUT` |
| `chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp` | 7 | `289:O_QK`, `317:O_QH`, `324:O_QH`, `351:O_ATTN_V`, `358:O_ATTN_V`, `403:O_MASK`, `425:O_OUTPUT` |
| `chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp` | 23 | `353:KKT_EPILOGUE`, `418:FRONT_WAIT`, `434:FRONT_WAIT`, `440:FRONT_WAIT`, `448:SOLVE`, `478:FRONT_WAIT`, `518:KKT_MM`, `536:CUMSUM`, `545:PHASE_WAIT`, `581:SOLVE_WAIT`, `585:KKT_EPILOGUE`, `597:PHASE_WAIT`, `603:PHASE_WAIT`, `610:SOLVE`, `616:PHASE_WAIT`, `622:PHASE_WAIT`, `630:PHASE_WAIT`, `634:SOLVE`, `640:SOLVE`, `652:SOLVE_WAIT`, `703:PHASE_WAIT`, `709:PHASE_WAIT`, `740:PHASE_WAIT` |
| `chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp` | 7 | `389:O_QK`, `417:O_QH`, `424:O_QH`, `452:O_ATTN_V`, `459:O_ATTN_V`, `505:O_MASK`, `533:O_OUTPUT` |
| `chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp` | 8 | `407:H_WH`, `481:H_KV`, `661:H_WH`, `670:H_WH`, `712:H_KV`, `721:H_KV`, `885:H_VNEW`, `925:H_STATE_UPDATE` |
| `chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_cube.h` | 2 | `199:U_MM`, `269:W_MM` |
| `chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_vector.h` | 1 | `138:WU_PREPARE` |
| `chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp` | 14 | `357:KKT_MM`, `363:CUMSUM`, `373:PHASE_WAIT`, `385:KKT_EPILOGUE`, `393:SOLVE`, `399:SOLVE`, `405:SOLVE`, `419:SOLVE_WAIT`, `435:SOLVE_WAIT`, `444:SOLVE_WAIT`, `451:SOLVE_WAIT`, `492:PHASE_WAIT`, `498:PHASE_WAIT`, `517:PHASE_WAIT` |
| `chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp` | 12 | `408:O_QK`, `515:O_QH`, `522:O_QH`, `558:O_ATTN_V`, `566:O_ATTN_V`, `601:O_QK`, `629:O_QH`, `636:O_QH`, `664:O_ATTN_V`, `671:O_ATTN_V`, `720:O_MASK`, `762:O_OUTPUT` |
| `chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp` | 14 | `498:H_WH`, `578:H_KV`, `722:H_WH`, `763:H_WH`, `772:H_WH`, `815:H_WH`, `864:H_KV`, `917:H_KV`, `926:H_KV`, `970:H_KV`, `986:H_WAIT`, `990:H_WAIT`, `1160:H_VNEW`, `1202:H_STATE_UPDATE` |
| `chunk_gated_delta_rule_fwd/op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp` | 8 | `357:H_WH`, `409:H_KV`, `573:H_WH`, `582:H_WH`, `624:H_KV`, `633:H_KV`, `777:H_VNEW`, `817:H_STATE_UPDATE` |
| `chunk_gated_delta_rule_fwd/op_kernel/internal/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_cube.h` | 6 | `189:WU_WAIT`, `213:U_MM`, `219:WU_WAIT`, `241:W_MM`, `313:U_MM`, `370:W_MM` |
| `chunk_gated_delta_rule_fwd/op_kernel/internal/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_vector.h` | 2 | `172:WU_PREPARE`, `207:WU_WAIT` |
| `chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h` | 22 | `324:FRONT_WAIT`, `345:FRONT_WAIT`, `365:FRONT_WAIT`, `422:PREPARE_NORM_GATE`, `481:KKT_MM`, `506:PREPARE_KKT_GATE`, `518:KKT_EPILOGUE`, `554:PREPARE_SOLVE_PREFILL`, `566:PREPARE_SOLVE_ISSUE`, `578:PREPARE_SOLVE_DUMP`, `598:SOLVE`, `626:PREPARE_SOLVE_DUMP`, `656:WU_PREPARE`, `722:W_MM`, `736:U_MM`, `754:U_MM`, `823:FRONT_WAIT`, `860:FRONT_WAIT`, `887:FRONT_WAIT`, `898:FRONT_WAIT`, `904:FRONT_WAIT`, `924:FRONT_WAIT` |

## 设备验证边界

本机无 CANN/NPU/C++ 编译器。没有设备 raw、CSV、trace、阶段耗时或性能收益证据；没有证明 CANN 生成参数映射、DCCI API、全部模板编译、MIX 核映射或缓存发布已在设备通过。A2/A3/A5 分支实际选择、四个受影响算子普通调用、Profile 真实加载、精度/确定性/内存、Example/ST 与 ENABLE_TIMER=0 编译/开销对照仍按 [timer_run.md](timer_run.md) 和仓库阶段 05 执行。本报告只完成 Step 9 的独立源码交叉审核，不将阶段 05 归档为完成。
