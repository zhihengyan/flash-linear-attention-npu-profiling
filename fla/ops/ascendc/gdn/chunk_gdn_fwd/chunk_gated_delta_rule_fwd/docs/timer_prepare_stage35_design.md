# Prepare Stage3–5 诊断设计（2026-10-14）

基线：codex/gdn-profiling，809bce6887639e831caa856b8ac3a197a6a94294。分支 codex/gdn-prepare-stage3-5-timer。仅补观测，不采用 Stage6 或 H 双 head 优化，不改 host 编排、公开 ABI、数学、layout、任务、workspace 或原业务同步。目标仍为原 GDN case0 / A5 / device0。

## 原始流程与观测锚点

复用 timer_flow.md 的公开调用链：ACLNN→Cast/Transpose→Prepare→H→O→Transpose。Prepare 的原流程如下，括号为拟新增观测名的后缀，全部 NoBarrier：

~~~text
每核 grid-stride pack，原有 pack/task 编排
 AIV Stage1 → AIC KKT / 原 KKT ready
 AIV 对本地 task：
   WaitCubeKktDone                        S3_KKT_WAIT（原等待返回）
   MulKktGateVF                          S3_GATE_MUL（issue）
   V→MTE3 原wait                         S3_L_READY_WAIT
   PackDiagLeavesFromUb                  S3_LEAF_PACK（issue）
   MTE3→V 原wait                         S3_PACK_READY_WAIT
   UbNd64ToL1Nz8(-L)                     S3_NEG_L_UPLOAD（issue）
   MulReduceScatterVF32                  S3_LEAF_SOLVE（issue，保留内部依赖）
   V→MTE3 原wait                         S3_SOLVE_READY_WAIT
   UploadDiagLeavesToL1                  S3_LEAF_UPLOAD（issue）
   NotifyAicStage3Done                   S3_READY_PUBLISH（issue）
 AIC 对 task：
   WaitAivStage3Done                     S4_INPUT_WAIT
   Stage4 Acc（原 PREPARE_SOLVE_ISSUE）
                                          S4_ACC_ISSUE（同 work ID，含原内部等待）
   M→FIX 原wait                         S4_CUBE_DONE_WAIT
   Y L0C→GM                             S4_Y_GM_WRITE（issue）
   FIX→MTE2 原wait                      S4_Y_GM_DONE_WAIT
   Y GM→L1                              S4_Y_L1_READ（issue）
   NotifyAivStage4Done / MTE2→MTE1 Set    S4_READY_PUBLISH（issue）
   MTE2→MTE1 原wait                     S5_Y_READY_WAIT
   Stage5 Compute（原 SOLVE）
                                          S5_COMPUTE（同 work ID，含原内部等待）
   M→FIX 原wait                         S5_CUBE_DONE_WAIT
   outputA：A L0C→GM                    S5_A_GM_WRITE（issue）
   FIX→MTE2 原wait                      S5_A_GM_DONE_WAIT
   full：A GM→L1                        S5_A_L1_READ（issue）
   tail/no-output：A L0C→L1              S5_A_L1_WRITE（issue）
   原 MTE2/FIX→MTE1 Set                 S5_A_READY_PUBLISH（issue）
   原 FIX→M Set                        S5_READY_PUBLISH（issue）
 下游 AIC Stage7，原位置：
   WaitAivStage6Done                    S7_INPUT_WAIT
   原 MTE2/FIX→MTE1 Wait               S7_A_READY_WAIT
 原 Stage6/7 和下一pack依赖保持
~~~

另保留 S3_POST_KKT、S4_Y_TRANSFER、S5_A_TRANSFER 粗窗，和 PREPARE_PACK 业务循环窗；子窗可以嵌套，不把它们相加成纯计算或 launch 时长。输入 wait 只证明相应生产者已就绪；不证明随后异步操作完成。不加 barrier 来强制分段完成。

## 采样与布局 v3

每次 Prepare 选择每核最多8个 pack，默认详细采集窗口从 pack ordinal 0/16/32/48 开始。case0 每核约50/51 packs，分别观察启动、两个稳定期区段和尾部；其它shape只记录实际存在的pack。每stage最多32个记录，AIC每pack4task、AIV每pack2task，覆盖选中8pack时不盲目扩大容量。

header[12/13/14] 为 pack_start、pack_count（1..8，输入0使用8）、detail_enabled（0/1）；[15] 为 tag 行 stride。仅 PREPARE 使用窗口选择；H/O/Phase6保持前32次的粗观测。Stage0 在 pack 循环之前，仍保留其一次观测。每stage全部 actual 和独立 sampled 在本地累计，结束戳之后分别 setDynamicActualIter/setDynamicSampledIter 并发布；不在未选中的每次细分调用中写 GM 计数。晚窗口可实际采样为0。旧 PREPARE_SOLVE_DUMP 混合两次/task，选中8pack时仍可能到32槽上限；新增独立 S4/S5 transfer 避免这项混合。

追加32类事件的 work_id：值来自原 PackWorkId，task=chunk-major work ID，导出 hv=work_id%HV、global_chunk=work_id/HV；sequence/local chunk 需结合原 chunk_indices，工具不猜测。PREPARE_PACK 用该pack的首个task ID。GM tag 保存 work_id+1（0代表未写，task0的tag为1），每scope保存自身ID，不使用跨stage调用ordinal配对。同核、同work ID、不同stage才可关联；分析器同时用实际 Stage4 task 总数校验选中窗口全部 task/pack 和 G=2 重排。跨核原始cycles默认为各核局部观测。

布局由枚举/常量统一推导，升级为3：6固定项、60动态类（保留原28类ID）、32个带tag的新类，232处动态scope。每诊断调用15384640个int64，117.38MiB；显式业务workspace/UB/L1/L0划分不变，计时器局部数组增加可能影响编译资源或寄存器压力。host沿用TOTAL_BUFFER_SIZE校验，Python采集读取同一canonical头。新raw必须用同套v3 OPP/解析器，拒绝旧v2；旧数据继续在旧分支分析。所有计时/计数/tag行独占cache line，Finish仅诊断时本核一次PIPE_ALL，结束后发布本核行，header仍仅AIC0发布。

## 运行和结果

输入 shape/BT/cu_seqlens/chunk_indices 独立推导总work数，必须与Stage4 actual相符；每核完整pack数和S3/S4/S5/S7必经类型actual按原grid-stride验证。选中pack的全部业务ID、A输出与有效M/full/tail/no-output分支均严格校验；capture的采样控制必须与kernel header一致。不能将漏记全部计数和事件当作合法空尾窗。

原 profile_performance.py 主入口先运行普通 msprof、保存同一输入。追加 --prepare-stage35 时，每个 0/16/32/48 窗口先 coarse 后 detail，共8次诊断；--timer-repeats 2 采16次，均在普通性能循环外。coarse/detail对照仅检查详细观测的扰动，不用诊断window代替普通性能。新增 ordinary-only 模式可用同一个入口校验并测量原基线wheel，跳过v3诊断。应同时用旧A wheel和新诊断wheel普通调用，检查整个新计时实现是否改变了编译资源/普通性能。

每capture自动执行同输入普通/诊断十输出精确对照和严格解析，导出原raw/parsed/layout/capture metadata与brief/core/stage/longest摘要。另输出Prepare按work ID对齐的小诊断表与跨capture汇总。原始大数据留内部，优先提供run_metadata、诊断汇总与各窗口brief。

本机无CANN/NPU，仅完成源码、离线和独立审核；设备编译/数值/扰动/分支验证待内部环境执行。CPU标杆精度和MSS不由十输出同轮对照替代。

PREPARE_PACK为循环业务下发窗：AIV包含上一pack的DrainPackWuToGm，最后当前pack的写回在循环之外。固定KERNEL_TIMING在Finish计数/tag/DCCI之前结束；粗/细比较不覆盖这部分发布开销，也不覆盖新布局对H/O/Phase6的共同发布增量。新旧wheel普通性能对照和ENABLE_TIMER=0设备编译对照仍需实际结果。

## 解释与止损

先看稳定期S3_KKT_WAIT与S3_POST_KKT，再分leaf solve/布局/上传；结合S4_INPUT_WAIT判断AIC是否被AIV限制。Y/A输出与回读的issue、原完成wait分别报告；仅凭缩短issue或减少GM字节不能预报收益。

普通基线七项和1708.3914us，5%要节省85.41957us；单靠Prepare需约11.95%。若pipe指标和稳定期合法路径不足以支持该预算，停止候选，不立刻改求解公式、dtype或Y layout。本分支先交付诊断，没有预先选择优化代码。
