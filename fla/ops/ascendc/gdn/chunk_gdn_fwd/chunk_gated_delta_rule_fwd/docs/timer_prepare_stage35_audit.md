# Prepare Stage3–5 独立源码审核（2026-10-14）

审核者为 fresh-context 独立 agent，未参与插桩实现，未修改实现代码，也未委派其它 agent。依据 op-timer-ascendc Step 9，先读取根 AGENTS、docs/agents 的 03→04→05、公共 runtime 架构与 ATK 规则，再对照 timer_prepare_stage35_design.md、timer_flow.md 和最终工作树源码。

工作树：C:/Users/zhihengyan/.codex/worktrees/gdn-prepare-stage3-5-timer/flash-linear-attention-npu；分支 codex/gdn-prepare-stage3-5-timer；比较基点 809bce6887639e831caa856b8ac3a197a6a94294（codex/gdn-profiling）。旧 timer_audit.md 和历史设备结果没有用作本轮通过证据。

## 结论

本轮 Step 9 源码审核通过。确认追加 32 个类型（ID 28–59）、37 处动态 scope；前 28 类 ID 和 195 处既有 scope 保留，总计 6 固定项、60 动态类、232 处动态 scope。源码与完整流程清单的 stage/file/line 三元组双向一致，37 处新增 scope 均有独立 RAII 起止，未发现未收尾 Tik 或没有起点的 Tok。

相对基点，Prepare 只增加计时宏和包围词法块，去除这些诊断元素后的业务 token 与原同步调用有序序列完全相同。Stage6/H 优化、数学、业务布局、task 重排、workspace、原 event/flag 与分支未改。

审核中发现的“详细 actual/事件整体遗漏可被当成空窗口”“只校验事件组合、未核对真实 A 输出/tail 契约”“采集请求与 header 控件未核对”均已由实现方修复，并由本审核者重跑反例和最终 56 项离线测试确认。继承的 ABI 正则检查失败仍保留；CANN/NPU 编译与设备验收未完成，因此不能宣称工程阶段 05 或设备计时通过。

## 原流程、枚举、解析器与清单的双向对应

canonical AscendTimerV2.hpp 给出连续且不重名的枚举；parse_timer_csv.read_layout 直接安全解析该头，不再维护可能漂移的手写名字。FixedTiming 为 KERNEL_TIMING_IDX、MAIN_INIT、MAIN_BODY、PHASE6_FRONT_DONE、PHASE6_H_DISPATCH_DONE、PHASE6_HO_WAIT；仅导出名 KERNEL_TIMING_IDX→KERNEL_TIMING 有显式映射。

独立检查结果：

~~~text
baseline_scopes = 195
existing_scopes_unchanged_by_file_and_type = 195
new_types = 32
new_scope_pairs = 37
all_scope_pairs = 232
unique_scope_locations = 232
documented_anchors = 232
source_only = []
doc_only = []
source_stage_file_line_equals_document_stage_file_line = true
~~~

对全部实际宏调用逐文件剥离注释/字符串后扫描花括号，找到包含该构造点的最内层词法块及其匹配关闭点，共 232 个。原 195 处的 (file, stage) 多重集合与基点相同；新增类型集合严格为 ID 28–59，全部只落在 Prepare arch35 实现。完整逐点清单仍以 [timer_flow.md](timer_flow.md) 的 232 行为准，本报告完整列出新增 37 处。

## 逐新增类型和调用位置的配对证据

以下全部位置属于 fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h。源码行写为“构造点→所在词法块末尾”；Tok 由该对象析构执行。每行是独立的一对，均为 NoBarrier / 单槽 Overwrite 等价写入。

循环轴：PACK 行为每核 grid-stride pack ordinal；S3 为每 pack 的 AIV 本地 task（t=subBlock，步长2）；S4/S5/S7 为每 pack 的 AIC task（t=0..nThis-1）。iteration 是类型内部的密集采样槽，任务身份由 work_id 保存。未到达的 full/tail/no-output 分支不制造记录。

| ID | 类型 | 源码行（Tik→Tok块末尾） | 核型 / 条件 | 原业务节点与测量含义 |
| --- | --- | --- | --- | --- |
| 28 | PREPARE_PACK | 897→937 | AIV / 每pack | 本次 AIV pack 循环窗；含上一pack的 DrainPackWuToGm，当前最后pack写回在循环外；不是该pack输出完成时延 |
| 28 | PREPARE_PACK | 964→1040 | AIC / 每pack | 原 AIC pack 业务循环及已有等待/通知窗口；异步下发和同步混合，不等同纯计算 |
| 29 | PREPARE_S3_POST_KKT | 523→565 | AIV / 每task | 原 KKT wait 返回之后至 Stage3 ready 发布；包含所有 S3 子窗与原依赖 |
| 30 | PREPARE_S3_KKT_WAIT | 520→522 | AIV / 每task | WaitCubeKktDone 的原 CrossCoreWaitFlag 返回锚点 |
| 31 | PREPARE_S3_GATE_MUL | 527→529 | AIV / 每task | MulKktGateVF 下发；随后 V→MTE3 wait 才提供相应就绪锚点 |
| 32 | PREPARE_S3_L_READY_WAIT | 531→534 | AIV / 每task | 原 V_MTE3 SetFlag+WaitFlag，观察 L 的就绪等待 |
| 33 | PREPARE_S3_LEAF_PACK | 536→538 | AIV / 每task | PackDiagLeavesFromUb 原布局/打包下发 |
| 34 | PREPARE_S3_PACK_READY_WAIT | 540→543 | AIV / 每task | 原 MTE3_V SetFlag+WaitFlag 返回 |
| 35 | PREPARE_S3_NEG_L_UPLOAD | 545→547 | AIV / 每task | UbNd64ToL1Nz8 负 L 上传下发，未新增排空 |
| 36 | PREPARE_S3_LEAF_SOLVE | 549→551 | AIV / 每task | MulReduceScatterVF32 原 leaf solve 调用窗，保留其内部依赖；不是纯计算完成时延 |
| 37 | PREPARE_S3_SOLVE_READY_WAIT | 553→556 | AIV / 每task | 原 V_MTE3 SetFlag+WaitFlag 返回 |
| 38 | PREPARE_S3_LEAF_UPLOAD | 558→560 | AIV / 每task | UploadDiagLeavesToL1 两个 leaf 的原上传下发 |
| 39 | PREPARE_S3_READY_PUBLISH | 562→564 | AIV / 每task | NotifyAicStage3Done 原 PIPE_MTE3 CrossCoreSetFlag 下发 |
| 40 | PREPARE_S4_INPUT_WAIT | 983→985 | AIC / 首task0 | WaitAivStage3Done 原输入就绪等待返回 |
| 40 | PREPARE_S4_INPUT_WAIT | 990→992 | AIC / 后续task | 同一原等待；与首task分支互斥，每task一次 |
| 41 | PREPARE_S4_ACC_ISSUE | 598→606 | AIC / 每task | 原 Stage4 Acc 的 M_MTE1 wait、MatmulToL0C 和 SetFlag；含原内部等待的 issue 窗 |
| 42 | PREPARE_S4_Y_TRANSFER | 611→636 | AIC / 每task | 原 Stage4_Dump 完整 transfer 窗；包含 Cube/GM 等待、Y写出/回读和 ready 发布 |
| 43 | PREPARE_S4_CUBE_DONE_WAIT | 614→616 | AIC / 每task | 原 M_FIX WaitFlag，Stage4 Cube 完成锚点 |
| 44 | PREPARE_S4_Y_GM_WRITE | 618→620 | AIC / 每task | 原 Y Fixpipe L0C→GM 下发 |
| 45 | PREPARE_S4_Y_GM_DONE_WAIT | 623→626 | AIC / 每task | 原 FIX_MTE2 SetFlag+WaitFlag，Y写出就绪锚点 |
| 46 | PREPARE_S4_Y_L1_READ | 628→630 | AIC / 每task | 原 CopyGmNzToL1Fp32 回读下发 |
| 47 | PREPARE_S4_READY_PUBLISH | 632→635 | AIC / 每task | 原 NotifyAivStage4Done 和 MTE2_MTE1 SetFlag 下发 |
| 48 | PREPARE_S5_Y_READY_WAIT | 1001→1003 | AIC / 每task | 原 MTE2_MTE1 WaitFlag，Y回读就绪锚点 |
| 49 | PREPARE_S5_COMPUTE | 647→658 | AIC / 每task | 原 Stage5 两次 MatmulToL0C 与原 wait/flag；计算下发窗，未新增完成等待 |
| 50 | PREPARE_S5_A_TRANSFER | 676→728 | AIC / 每task | 原 Stage5_Dump 及既有 full/tail/no-output 分支完整窗 |
| 51 | PREPARE_S5_CUBE_DONE_WAIT | 682→684 | AIC / 每task | 原 M_FIX WaitFlag，Stage5 Cube 完成锚点 |
| 52 | PREPARE_S5_A_GM_WRITE | 687→689 | AIC / outputA | 原 A Fixpipe L0C→GM 下发，只写有效 M 行 |
| 53 | PREPARE_S5_A_GM_DONE_WAIT | 691→694 | AIC / outputA | 原 FIX_MTE2 SetFlag+WaitFlag 返回 |
| 54 | PREPARE_S5_A_L1_READ | 697→699 | AIC / outputA且full | 原 GM A→L1 Nd2Nz 回读下发 |
| 55 | PREPARE_S5_A_L1_WRITE | 706→708 | AIC / outputA且tail | 原 DumpAL0cToL1 直接写L1下发；仍保留A GM输出 |
| 55 | PREPARE_S5_A_L1_WRITE | 716→718 | AIC / no-output | 原 DumpAL0cToL1 直接写L1下发 |
| 56 | PREPARE_S5_A_READY_PUBLISH | 701→703 | AIC / outputA且full | 原 MTE2_MTE1 SetFlag 下发 |
| 56 | PREPARE_S5_A_READY_PUBLISH | 710→712 | AIC / outputA且tail | 原 FIX_MTE1 SetFlag 下发 |
| 56 | PREPARE_S5_A_READY_PUBLISH | 720→722 | AIC / no-output | 原 FIX_MTE1 SetFlag 下发 |
| 57 | PREPARE_S5_READY_PUBLISH | 725→727 | AIC / 每task | 原 FIX_M SetFlag，维持原 bank 复用通知 |
| 58 | PREPARE_S7_INPUT_WAIT | 1014→1016 | AIC / 每task | 原 WaitAivStage6Done 返回，仅观测下游既有等待 |
| 59 | PREPARE_S7_A_READY_WAIT | 1021→1027 | AIC / 每task | 原 outputA/full→MTE2_MTE1，否则→FIX_MTE1 等待返回，分支未改 |

与设计原流程逐节点核对：S3 KKT等待→gate乘→L就绪→leaf打包→pack就绪→负L上传→leaf solve→solve就绪→leaf上传→ready，S4 输入wait→Acc→Cube完成wait→Y GM写→GM完成wait→L1读→ready，S5 Y就绪wait→Compute→Cube完成wait→A输出/回读或直接写L1→A ready→bank ready，以及下游 S7 输入/A ready wait，均有实际调用和对应命名；额外 parent 窗与 PACK 窗也列入清单。未将 issue 返回描述为异步运算完成。

## RAII、开关和原行为

GdnTimerScope 在 Enabled 时 Begin，保存同一个 timer/type/iter/workId 和独立 start；析构 End→TokNoBarrier，拷贝和赋值删除。嵌套父子窗保存各自 start，互不覆盖。宏用 __LINE__ 生成变量名，232 个位置在各自文件行上唯一。每个被选中的单槽只写一次，无需 Accumulate；互斥分支复用同一个类型不会重复写同一 task 槽。

动态起止实际是 GetSystemCycle 的 NoBarrier 读，End 不调用 PipeBarrier。唯一 timer PIPE_ALL 位于固定总窗 Finish 的 Tok；每个有效诊断核一次，MAIN_BODY 复用同一结束戳。原带等待的 helper 返回作为各 wait 的锚点，通知/拷贝/矩阵运算仅记录下发窗。

NULL timer：Init 先禁用后返回；scope不 Begin、不读 clock、不写GM，PACK宏也不计算 PackWorkId；Finish 直接返回。ENABLE_TIMER=0：三种 scope/context 宏为空，Init/Finish/计数路径在预处理阶段关闭。局部对象尺寸和编译资源仍需真实编译检查，源码不能证明普通路径零扰动。

check_timer_source 对唯一修改的 Prepare kernel 文件检查原 WaitFlag/SetFlag/CrossCore/SyncAll/PipeBarrier 有序调用完全相同，并检查去 timer 宏/新花括号之后的全部业务 token 一致。人工核对新增词法块只改变诊断对象生命周期，没有移动业务变量的可见区、原 if 条件、loop、call 参数或数据地址。

## v3 GM 区、所有权和 Finish 发布

全部单位为 int64 words：

| 区域 | slice内基址 | 每核stride | 总长度 |
| --- | --- | --- | --- |
| header | 0 | AIC0唯一写者 | 16 |
| timing | 16 | 3856 | 2961408 |
| actual | 2961424 | 64 | 49152 |
| sampled | 3010576 | 64 | 49152 |
| work tags | 3059728 | 1024 | 786432 |

BUFFER_WORDS_PER_KERNEL=3846160；四slice的 TOTAL_BUFFER_SIZE=15384640（117.38 MiB）。各基址、每核行、slice stride 均为64B整行；c=3g/3g+1/3g+2分别由 AIC/AIV0/AIV1独占。超容量不转写core0，解析器拒绝实际blockDim溢出及任何越界行数据。

header[12..14] 由采集端先填并同步，所有核读取；AIC0发布完整header，读写的控制值在launch中等价不变。header[15]记录tag stride。Finish先drain并写两个结束戳，再一次性写actual/sample，最后 FlushRow 对本核timing、actual、sampled、tag四行逐cache line DCCI；header只由AIC0发布。不刷新整片DCache，也不增加跨核同步。

Prepare 的既有编译选项关闭 kernel-end 自动DCCI；显式发布是源码可见的契约，实际 DataCacheCleanAndInvalid API、MIX 核映射、数据可见性和编译资源尚无设备证据。新header影响 ChunkGatedDeltaRuleFwd（Phase6）、ChunkGatedDeltaRuleFwdPrepare、ChunkFwdH、ChunkFwdO 四算子，host尺寸和全部 kernel 必须同套v3重建，不能只替换Prepare。H/O/Phase6仍按前32调用采样，但新stride、sampled写回及tag行发布影响其诊断开销。

### timer地址复用与采样控件的GM可见性待验

新增路径让所有核在 Init 中标量读取 header[12..14]；AIC0在Init末尾对自身header行DCCI，其它核只在Finish发布各自timing/count/tag行。Prepare仍关闭compiler-end自动DCCI。现有源码和工程资料没有足够证据证明其它核读入的控制cache line在kernel边界一定失效；不能仅根据host copy_之后synchronize推断反复复用同一timer地址时，各核总能读到新控件。

本轮不为此新增init clean，避免对可能包含旧数据的cache line作写回影响新host控制。设备验证必须记录实际timer地址复用，交替pack_start=0/16/32/48和coarse/detail，逐核检查actual/sample和work ID确属本次请求窗口；还应验证同地址连续调用和窗口回退。未发生地址复用不能宣称覆盖该场景。采集请求/header检查与严格work ID关联会拒绝可见的不一致或缺失，但这些拒绝检查不构成kernel-boundary缓存失效保证。采集metadata已增加timer_data_ptr、timer_words和capture_process_id，只记录采集信息，没有改变计时/业务路径，AST复查通过。主要入口每capture使用独立子进程；跨进程出现相同数值地址不证明同一缓存实例，必须另做同进程地址复用重放。当前无真实设备证据，此项明确未解决。
## 采样、业务身份和解析证据

Prepare PACK macro 在每次 MakePackWalk 后设置每核ordinal=(pack-coreIdx)/numCore，并从原 PackWorkId 填最多4个本地task ID。HRatio=3 packSize=3；其它为4；G=2满pack将task1/2映射为base+2/base+1，计时没有另造顺序。tag=work_id+1使task0有效，0表示未写。每scope构造保存自身ID，不会被下一pack映射覆盖。只允许同核、同work ID关联，不能用各stage的iteration对齐。

每类型 Begin 的actual记录全部调用，sampled只对选中pack和容量内调用累计。详细新类在detail=false时不计数；既有粗类仍观测选择窗口。最多8pack使 AIC必经stage≤32、AIV≤16；原 PREPARE_SOLVE_DUMP 两次/task最多64调用，仍可能只存32，已明确partial sampling。Stage0前相保留一次，AIV循环后drain沿最后pack选择；晚窗口确实超出实际pack时允许sampled=0。

解析器检查准确长度/v3 metadata、每核总窗、配对/非负duration、padding、actual/sample约束、未采样槽为0、tag非0且同类型不重复、三launch完整性和互斥路由。Stage35分析器进一步从实际rank4布局、chunk_size、cu_seqlens和chunk_indices值独立推导总work；校验每核pack和S3/4/5/7必经actual、A输出/full-tail分支计数、选择窗完整task/pack集合、原G=2重排、stage顺序、父子窗包含及采集请求/header一致。

本入口 NTD/TND 是rank4别名：NTD与BNSD均[B,H,T,D]，TND与BSND均[B,T,H,D]，以当前host ResolveShapeInfo和生产ctypes为证。未采用其它算子的rank3命名惯例。独立G=1/2/3/4及partial pack合成关联检查通过；它们只验证解析算法。

## host、调用层、主要入口和扰动口径

host既有Profile尺寸检查使用canonical TOTAL_BUFFER_SIZE，要求rank1、ND、INT64、连续；没有对timer作Contiguous复制。既有C公开声明和生产Python签名与基点相同；私有Profile沿原输出十项、同设备/连续检查、mutation记录和生命周期传递。新采集器从同一头创建清零tensor，检查64B对齐，填写控制后同步；调用后同步，先保存raw再做十输出精确比较和解析，失败保留raw/metadata。

主要入口仍是原ATK profile_performance.py，复用原case JSON、generator/executor及保存输入；普通msprof循环外进行coarse/detail和导出。--ordinary-only 跳过诊断符号和timer重放，可用同一checkout切换旧A/newB wheel；--reference-wheel 比较安装包里的Python、host、config、kernel文件hash，并拒绝缺失或多余native文件。离线测试是伪造wheel合同，不证明当前内部实际装载。

coarse/detail对照共享同一输入快照、header hash、options、blockDim和pack选择，并分别报告AIC/AIV固定窗口的capture median及min/max。它证明窗口内详细观测变化的统计方式可审查；固定先coarse后detail的顺序和有限重复仍可能受缓存/频率/噪声影响，单对数据不能证明稳定扰动。

KERNEL_TIMING结束戳早于计数和DCCI发布，coarse/detail不包含该收尾成本。两者也共同包含新计时器编译资源和更大v3行的成本。必须另外做旧A/newB普通调用的重复七项msprof Task Duration总和，必要时ENABLE_TIMER=0编译对照；诊断窗变化不能替代≥5%目标验收。现有主入口的“有成功case且有实际GDN kernel”证据检查不是完整七项性能聚合验收，需按运行说明核对完整报告。

默认cycles；频率和共同时钟未核实不导出Perfetto。trace保留work_id，转换前先减整数origin，四slice各自origin，Stage35工具无跨核/跨launch时间戳相减。wait union不等于硬件idle；parent/child窗口不可相加，core cycles总和不可当launch时长。

## 审核发现、修复与未解决项

| 项目 | 初审证据 | 当前状态 |
| --- | --- | --- |
| 详细计数和事件全遗漏被误作空窗口 | 非空v=[1,4,64,128]、detail=true、pack_start=0，将iterations/events均置空，初版task_rows返回([],[]) | 已修复：独立input work总数与各核必经actual/pack校验；原反例现在抛ValueError |
| A输出/tail真实分支未校验 | 初版仅判断事件组合内部合法；tail fixture仍T=64，no-output未显式disable_recompute=false | 已修复：a_transfer_contract按原dtype/layout无关的A输出flag与有效M校验实际分支及actual；fixtures使用真实T=63/false；相关回归通过 |
| capture/header控件未核对 | 初版只读kernel sampling | 已修复：analyze_capture逐项检查请求值；real CSV/layout/capture离线流水测试拒绝不一致 |
| 继承ABI源码检查失败 | aclnn_abi_contract.py:157–158 正则要求ChunkFwdO末尾executorPtr)，已有基点调用为executorPtr,timerOptional) | 未解决：本次未修改无关脚本；基点与当前checker及参与断言的host源码完全相同，执行基点checker同样退出1 |
| timer地址复用/控制可见性 | 非AIC0核只读header，无工程证据保证跨launch读缓存失效 | 未解决：需记录同地址复用并交替窗口/detail验证；严格关联拒绝不一致不能替代该设备证明 |
| CANN/NPU及性能验收 | 本机Get-Command未找到ccec/c++/g++/clang++/npu-smi/docker | 未完成：无设备编译、数值、MSS、分支、GM发布、频率或真实开销证据 |

未发现本轮新增源码中的未配对、枚举/清单漏项、原同步变更或数学/layout变更。表中的继承检查失败和设备缺口不能写为通过；旧audit、合成raw和历史设备性能不补齐这些证据。

## 执行命令与结果

以下命令在上述工作树执行，PY=D:/Program/miniforge3/python.exe，UT=fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut：

~~~text
PY -X utf8 UT/check_timer_source.py
exit 0；source_checks_passed；fixed_count=6；dynamic_count=60；scope_count=232
business_token_check=unchanged after removing timer macros and braces
synchronization_files=[Prepare arch35 header]

PY -X utf8 -m unittest discover -s UT -p test_*.py -v
exit 0；Ran 56 tests in 19.800s；OK
21 Stage35 + 10 reference-wheel + 25 existing timer contract/summary

git diff --check
exit 0；仅CRLF规范化提示，没有diff格式错误

PY -X utf8 tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py
exit 1；ACLNN 默认路径映射缺失：['direct BSND output']

独立执行git show 809bce68:.../aclnn_abi_contract.py所得源码：
baseline ABI checker == current: True
baseline host == current: True
executed baseline checker: ACLNN 默认路径映射缺失：['direct BSND output']
exit 1
~~~

另外执行Python AST语法检查（分析器、重放、trace、主要入口），exit0；逐scope词法关闭检查232/232，新增37/37；stage/file/line清单双向比对232/232；原195处(file,stage)多重集合一致。独立关联合成检查结果：

~~~text
G=1, pack4: 8 task-core rows / 3 pack rows
G=2, partial pack2: 4 task-core rows / 3 pack rows
G=3, pack3: 6 task-core rows / 3 pack rows
G=4, pack4: 8 task-core rows / 3 pack rows
~~~

所有离线测试和反例均使用合成数据或源码，不是设备raw，不是CANN编译通过。审核过程中新增的临时Windows短路径fixture断言失败由实现方改为resolve比较后重跑通过，未隐藏失败历史。

## 最终审核源码指纹

以下为 source_text 读取、换行归一后的UTF-8 SHA256；用于定位本审核覆盖的源码状态，hash不是安装/设备装载证明。

| 文件 | SHA256 |
| --- | --- |
| canonical AscendTimerV2.hpp | 23beb460c52af7348b089f0af20bb02d08e96d9f77efdc465ad225b73144c66a |
| canonical AscendTimerV2_device.hpp | 43b96f0e51e560eaf51d3417c0b45f3a99870a2530db9f0e41f10338ad868aad |
| Prepare arch35 header | 69b96c8bc8861677d46ea31290a46817e18761e2f9e4cce9fd5e7f1f90199820 |
| analyze_prepare_stage35.py | ee497fb46b1ac11fef797040655eb3ae24704b8568814fca940d605768298726 |
| profile_gdn.py | 6e27c2be4450e9f95709861f37053233b2fba3708adaf57a33e6d83e08826456 |
| trace_parser.py | 3e0f5fdacc496d1c12d1b9058672b6a45c3d762742e090cc92a490cfb2bee39f |
| ATK profile_performance.py | d4eec4470c88f4a02df3d6c50441e11426133659291ac3d1d646de079e35ec08 |

设备执行、安装、原ATK/Example/ST和真实数据回传以 [timer_run.md](timer_run.md) 和 [本轮汇总](timer_prepare_stage35_report.md) 为准。本审核仅归档独立源码交叉校验及离线证据。
