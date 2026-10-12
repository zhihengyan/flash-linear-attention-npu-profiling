# 方案 A 计时独立源码审核

审核日期：2026-10-11。两轮审核均由未参与实现的 Step 9 独立子 agent 执行；最终复核使用 fresh context，重新读取原图、方案 A、最终 diff 和源码。本次只更新审核报告，未修改实现。

**最终结论：方案 A 的源码审核通过。第一轮发现的 timer mutation/version 合同缺失已修复并独立复核通过；最终源码未发现新的阻断问题。**
计时合同 20 项、前向输出合同 2 项、原 mutation 合同 7 项均通过。既有 Prepare 非 exp2 合同失败仍然存在。用户限定本轮仅源码交付，生产 NPU/CANN 不可访问；此结论不表示 CANN 编译、实际 NPU 采集、精度、内存、确定性、性能或阶段 05 正式验收通过。

下文保留第一轮的真实缺陷与当时结论；当前有效结论和重新执行的证据在末尾“第二轮最终复核”。

## 第一轮审核记录（历史，修复前）

**结论：源码审核暂不通过，发现一项新增诊断 helper 的输入 mutation/version 合同缺失；计时位置、配对、槽位、ABI 和导出布局检查通过。设备验证尚未执行。**
本报告只覆盖用户已确认的方案 A 和源码交付范围，不表示 CANN 编译、NPU 精度、计时有效性、打点扰动或阶段 05 正式验收通过。

## 需要修复的问题

**[P2] 诊断 helper 原位写 timer，但没有推进其 PyTorch version counter。**

`torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py:2967`–`:2975` 在校验后直接返回私有实现；`:3149` 将 caller-owned timer 交给 kernel 原位写入。共享 device 头 `:116`–`:117` 确实修改 timer 存储，而 `_runtime.py:496`–`:515` 不处理 mutation。该私有 helper 不经过公共 `__init__.py:483`–`:511` 的 mutation 包装，`MUTATED_ARGUMENTS` 中也无对应项。

`docs/architecture/torch-npu-decoupled-architecture.md:471` 明确要求“不得修改输入 storage 却不维护 MUTATED_ARGUMENTS、version counter 和 grad 限制”。INT64 dtype 排除了 timer 自身 requires_grad，但不使外部 storage 写入自动对 PyTorch 可见；no_grad 和禁止 graph capture 也不会自动推进 version。需要在诊断调用成功后明确推进 timer version，并补充成功/异常返回路径的离线合同测试；若沿私有 eager helper 维护而不加入公共算子导出表，需在架构中记录这个诊断 mutation 处理边界。此修复不需要接入生产设备。

独立 mock 检查让 `_chunk_gated_delta_rule_fwd_impl` 成功返回，并对 `torch.autograd.graph.increment_version` 计数，输出：`diagnostic successful-return path: timer mutation version-bump calls = 0`。现有 18 项套件没有覆盖该合同。发现后已及时向主 agent 报告；修复后应重跑 Step 9。

## 审核输入与范围

- 已读取根 `AGENTS.md`、`docs/agents/README.md`、阶段 03/04/05 规则、runtime 架构，以及项目级 `.agents/skills/op-timer-ascendc/SKILL.md` 的 Step 9。
- 主判据是 `docs/timer_flow.md:47` 起的改造前原图、`:113` 起的方案 A、`:137` 的打点清单、`:200` 的枚举映射、`:289` 的最终落地映射；同时读取 `docs/api.md`、`docs/design.md`、`tests/ut/README.md`。
- 独立读取当前源码和 `git diff`，覆盖共享 timer、完整 GDN host、Prepare/H/O 的 OpDef/L0/raw entry、两种架构的私有 Phase6、ctypes 适配、parser/trace/capture、打包引用路径和相关测试。
- 原图的源码锚点注明为改造前行号；以下证据均使用审核时最终工作树行号。路径相对于仓库根目录；本报告中 `docs/` 和 `tests/ut/` 指完整 GDN 算子目录下对应路径。

## 原图节点与实际位置

| 原图与方案 A 节点 | 最终源码证据 | 核对结果 |
| --- | --- | --- |
| host 选择三 kernel 或 Phase6，沿同一公开前向调用链 | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp:105` 的 `UsePreparePath`；`:697` 的分支；`:763`、`:773`、`:779` 向 Prepare/H/O 传完整 timer；`:880` 向 Phase6 传 timer | 覆盖实际两条路由；未仅给同级旧 H/O 打点。正式入口 `:932` 仍不传 timer |
| Prepare Init、pack 流水及尾 W/U drain | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/chunk_gated_delta_rule_fwd_prepare.cpp:27`、`:33`、`:34`、`:35` | PREPARE slice 的 Init → 真实 op.Init → InitDone → 全部 op.Process → Finish；原 `arch35/chunk_gated_delta_rule_fwd_prepare.h:807`、`:824` 的 W/U drain 包含在 Process 内；无 pack 内额外打点 |
| H AIC 有工作、idle AIC、AIV；work-unit/chunk 及原尾 drain | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/chunk_fwd_h.cpp:93`；AIC idle `:54` 后 `:55` 返回；AIC InitDone `:64`；AIV InitDone `:76`；共同外层 Finish `:129` | 所有合法分支各结束 INIT 一次，idle AIC 返回的是内层函数，仍到外层 Finish；Process 的单 head/多 head/首末 chunk 分支均在 BODY 内；无循环轴覆盖槽位问题 |
| O 旧 kernel 分支与 A5 token-first AIC/AIV | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/chunk_fwd_o.cpp:109`、`:60`、`:116`；`op_kernel/arch35/chunk_fwd_o_a5.h:59`、`:61`、`:67` | FWD_O slice 共同 Init/Finish；旧服务 Init 后结束 INIT；A5 vector/cube Init 后分别结束 INIT；AIV 内层 return 仍回到 raw entry 的 Finish；owner/non-owner 原握手在 BODY 内 |
| arch22 Phase6 初始化、前端、私有 H、私有 O | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:479`、`:497`、`:632`、`:633`、`:637`、`:677` | INIT 在首 KKT 前结束；FRONT_DONE 在所有实际前端分支/按需 cumsum 导出及调度配置之后、私有 H 之前；H_DISPATCH_DONE 紧接私有 H 返回；总收尾在 O 或原最终会合之后。key 1/2/3/4 均传 timer |
| arch35 Phase6 初始化、前端、私有 H、私有 O | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:332`、`:353`、`:446`、`:447`、`:451`、`:479` | 同上；BT64/128、recompute V128/256、B0/B30 及按需 cumsum 输出均被包围；key 1/2/301 均传 timer |
| Phase6 串行 H/O 同步 | arch22 同文件 `:639`、`:640`、`:644`/`:647`、`:649`；arch35 同文件 `:453`、`:454`、`:458`/`:461`、`:463` | 同一条件分支内 Begin → 原 SyncAll → End；NoBarrier Tok 位于原锚点后 |
| Phase6 H/O 重叠与条件 O | arch22 同文件 `:663`、`:669`、`:675`、`:677`；arch35 同文件 `:474`、`:477`、`:479` | 重叠分支不执行 HoWaitBegin/End；arch22 producer 不执行 O 仍到原最终会合和 Finish；无新增跨核同步 |

上述实际观察节点与原图和方案 A 完全匹配，未发现漏标、多标或错位。原图中的 pack/head/chunk/groupTask 是业务循环；方案 A 没有在这些循环内增加动态事件。方案 B 及其枚举在流程文档中明确标为未实施，不作为 A 的必测槽位。

## 每个 Tik/Tok 的配对与锚点

下表的共享实现文件是 `fla/ops/ascendc/common/kernel_utils/gdn_timer/AscendTimerV2_device.hpp`。

| 每个实际起点 | 对应终点 | 配对和语义证据 |
| --- | --- | --- |
| all_.TikNoBarrier，`:144` | Finish 中 all_.Tok<Overwrite>，`:182` | 三独立 kernel 和两个 Phase6 均有单一共同 Finish；末尾 Tok 在 `:82` 执行一次本核 PIPE_ALL，再在 `:83` 读末尾 cycle；未新增全核入口基准同步 |
| stage_.TikNoBarrier，`:145`，MAIN_INIT | InitDone 中 stage_.TokNoBarrier<Overwrite>，`:152` | Prepare 一处、H 三个互斥核型/idle 分支、O 旧服务或 A5 两核型、两 Phase6 各一处；每条合法路径恰一次。INIT 明确是 issue window，不声称异步初始化已完成 |
| InitDone 中 stage_.TikNoBarrier，`:153`，MAIN_BODY | Finish 中 stage_.TokNoBarrier<Overwrite>(MAIN_BODY, all_.GetEndCycle())，`:183` | BODY 使用总窗口已经排空的同一个 endCycle；不重复排空，不把结束时间写到总窗口外 |
| Point 中 boundary_.TikNoBarrier，`:160` | 紧邻 boundary_.TokNoBarrier<Overwrite>，`:161` | 两个架构各有 FRONT_DONE 与 H_DISPATCH_DONE 一次调用；每次是瞬时对；解释为 dispatch boundary，非 pipe 全部完成 |
| HoWaitBegin 中 boundary_.TikNoBarrier，`:168` | HoWaitEnd 中 boundary_.TokNoBarrier<Overwrite>，`:175` | 两架构都在同一个串行条件内成对包围已有 SyncAll；重叠分支两端都不调用；只测该同步规定的窗口 |

未发现无收尾 Tik、无起点 Tok 或未成对的瞬时点。所有起点为 NoBarrier；计时实现内唯一新增 barrier 是总窗口末尾的局部 PIPE_ALL。固定槽位均使用 Overwrite，一次合法调用每核每槽至多一次；Accumulate 泛型存在但方案 A 未调用，不产生合成区间时间线。源码无 DynamicTimingType、setDynamicActualIter 或 MAX_DYNAMIC_ITER。

## 枚举、名称、清单与图

canonical 文件为 `fla/ops/ascendc/common/kernel_utils/gdn_timer/AscendTimerV2.hpp:11`；parser 从该文件派生，而不是复制计数或索引，见 `tests/ut/parse_timer_csv.py:21`、`:60`、`:62`。

| 索引 | C++ 枚举 / CSV 名称 | 方案 A 对照 |
| --- | --- | --- |
| 0 | KERNEL_TIMING_IDX / KERNEL_TIMING | `docs/timer_flow.md:137`、`:200`，A 图每个 kernel 的总窗口 |
| 1 | MAIN_INIT / MAIN_INIT | `docs/timer_flow.md:138`、`:201`，A 图 INIT |
| 2 | MAIN_BODY / MAIN_BODY | `docs/timer_flow.md:139`、`:202`，A 图 BODY |
| 3 | PHASE6_FRONT_DONE / PHASE6_FRONT_DONE | `docs/timer_flow.md:140`、`:203`，A 图前端边界 |
| 4 | PHASE6_H_DISPATCH_DONE / PHASE6_H_DISPATCH_DONE | `docs/timer_flow.md:141`、`:204`，A 图 H 返回边界 |
| 5 | PHASE6_HO_WAIT / PHASE6_HO_WAIT | `docs/timer_flow.md:142`、`:205`，A 图已有串行同步 |

六项数量、顺序、名称与图及清单一致；KERNEL_TIMING_IDX 到 KERNEL_TIMING 是明确的显示名称映射。FIXED_TIMING_COUNT 是哨兵，不导出；动态名称为空。KernelSlice 顺序为 PHASE6/PREPARE/FWD_H/FWD_O，canonical `:21`、parser `:64` 与 `docs/design.md:9` 一致。全仓定义扫描未发现 Overwrite/Accumulate/AscendTimerDevice/GdnKernelTimer 的重复定义。

## Null、关闭开关、地址边界与 ABI

| 检查 | 文件:行证据与结论 |
| --- | --- |
| null timer | 共享 device 头 `:21` 直接返回，`:69` 不读 cycle，`:78` 在 barrier 前返回，`:99` 不写 GM；正式 L2 `aclnn_chunk_gated_delta_rule_fwd.cpp:932` 使用默认 nullptr。未增加业务同步 |
| ENABLE_TIMER=0 | canonical 头 `:46` 可覆写默认值，`:50` 使用 #if；GdnKernelTimer `:140`、`:151`、`:159`、`:167`、`:174`、`:181` 条件编译全部调用，`:188` 条件编译计时成员。关闭时没有计时读取/写入/barrier 的可达路径；尚未做 CANN 编译对照 |
| slice 和核心行 | 共享 device 头 `:24`–`:32`：AIC group=GetBlockIdx，AIV group=GetBlockIdx/GetSubBlockNum、member=GetSubBlockIdx+1，行 ID=group×3+member；`:52` 超容量/非法成员直接禁写；`:104` 使用 16-word stride。映射与 H vector、Prepare、O A5 原有核组映射一致 |
| metadata 唯一写者 | 共享 device 头 `:34`–`:49` 只允许 AIC group 0 写 header；记录实际 AIC blockDim、容量、stride、六项、ENABLE_TIMER 和编译 arch。其他核不读 header，不需要新增跨核同步 |
| 全容量边界 | 独立枚举四个 slice、256 核组、每组 3 成员，3,072 个行地址全部唯一；最高事件 word=49,211，小于 TOTAL_BUFFER_SIZE=49,216。每行六对事件只占前 12 word，余 4 word 为 padding；不跨行或 slice |
| host 校验 | `aclnn_chunk_gated_delta_rule_fwd.cpp:644`–`:648` 使用 canonical TOTAL_BUFFER_SIZE 检查 INT64、一维、尺寸、ND、contiguous；`:981` 诊断 API 拒绝空 timer。没有为 timer 建连续副本；直接 C 调用者同设备责任写在 api.md |
| 四个 OpDef/L0/raw 顺序 | 主 OpDef `chunk_gated_delta_rule_fwd_def.cpp:46`；Prepare def `:50`；H def `:77`；O def `:77` 均末尾 OPTIONAL INT64 ND；各 L0 OP_INPUT 主 `chunk_gated_delta_rule_fwd.cpp:75`、Prepare `:125`、H `:108`、O `:64` 最后参数为 timerOptional；raw 最后输入位置主 arch22 `:685`/arch35 `:487`、Prepare `:20`、H `:88`、O `:103` 一致 |
| 旧 ABI 和十项输出 | header 保留旧两段接口，新增独立 Profile 符号；`_aclnn_ctypes.py:564` 在旧 ABI 末两输出指针前追加一个 timer descriptor；`:2967` 校验 NPU device/dtype/rank/连续性；`:3117` 仍为原十项 tuple；`:3148` 只在诊断调用追加 timer。独立 AST/声明对比证实正常 Python 签名/defaults、完整校验/输出分配前缀和原 C 两段签名保持一致 |
| 生命周期 | `tests/ut/profile_gdn.py:49` 使用 q 的 device；`:54` 先同步，`:55` 用 TOTAL_BUFFER_SIZE 新建全零独立 timer，`:57` 完成后同步，`:59` 再读 raw；直接 helper 的调用者持有责任在 api.md。没有并发复用示例 |
| 打包与 include | `cmake/config.cmake:41` canonical common 根；`cmake/func.cmake:379` 复制整个 common；`cmake/custom_build.cmake:957` 和 `CMakeLists.txt:550` 安装整个 common。四个 raw 实现使用 kernel_utils/gdn_timer include；目标 op_kernel 的两个转发头 `:2`/`:5` 兼容源码树和扁平 OPP。主 host CMakeLists.txt `:2` 包含 Prepare/H/O 依赖。安装和实际符号加载仍待设备构建验证 |

## Parser、trace 和采集

`tests/ut/parse_timer_csv.py:84`–`:151` 检查长度、非负 int64、magic/version/layout、blockDim 容量、实际 blockDim 外写入、缺核、未配对、负 duration、总窗口包含关系、padding、非 Phase6 的多余事件、INIT/BODY 顺序、dispatch 顺序、串行等待分支一致、混合/不完整路由和空采集。`:131` 将 INIT 标为 issue，两个边界标为 point；`:179` 的汇总排除 point 伪业务时长。

`tests/ut/trace_parser.py:16` 要求显式确认正有限 cycles/us 和 shared_clock；`:37` 先以整数减 kernel origin；`:42` 用瞬时对中点；`:46` 的 metadata 声明每个 kernel 独立 origin。它不表示 host 总时长或跨 kernel launch 间隔。CLI 读取的是已验证 CSV，`generate_trace_json` 接收 parser 已验证事件；未建立目标频率时只导出 cycle。

`tests/ut/profile_gdn.py:63` 保留完整 raw，`:65`–`:82` 写输入/选项、库路径、预热/单样本、时钟断言及 raw/header 哈希；`:83` 执行严格解析；`:86`–`:97` 对十项输出做 exact 比较，失败保留 raw。synthetic 导出测试只证明离线合同，不是设备采集。

## 执行证据

| 命令/独立检查 | 本次结果 |
| --- | --- |
| `python -X utf8 -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p test_timer_contract.py` | exit 0；Ran 18 tests，OK。含两路由、核行、坏记录、trace 精度、诊断 ABI、旧输出回归和 include 路径 |
| `python -X utf8 -m unittest discover -s ci/tests -p test_gdn_forward_output_contract.py` | exit 0；Ran 2 tests，OK |
| `python -X utf8 tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py` | exit 1；`prepare kernel 缺少非 exp2 模式的局部指数换底选择`，失败断言位于脚本 `:190`–`:191`。这是改造前已存在的失败；本轮只放宽 optional timer 调用正则，相关 Prepare 算法头未修改，不能报成通过 |
| `git -c core.autocrlf=false diff --check` | exit 0，无输出 |
| 独立 Python 全容量枚举、旧新 Python AST 前缀/签名和 C 声明对比、compile() 五个 Python 文件 | exit 0；输出：`4 slices / 3072 unique core rows; highest event word 49211 < 49216; no overlap`；正常 Python 签名/defaults 和计算前缀保持；原 C 两段签名保持；parser/trace/capture/tests/adapter 语法通过 |
| 独立以 git show HEAD 读取两 Phase6，再按源码顺序比较 SyncAll/CrossCoreSetFlag/CrossCoreWaitFlag/PipeBarrier/SetFlag/WaitFlag 语句 | exit 0；arch22 的 35 条和 arch35 的 25 条原同步调用语句保持原内容及顺序。新增局部 drain 仅在共享 timer 的总末尾 |
| 独立 mock 诊断 helper 成功返回，计数 `torch.autograd.graph.increment_version` | exit 0；观察到 version-bump calls = 0，证明本报告 P2 缺失；不是通过判据 |
| `rg` 扫描全部实际 timer.Init/InitDone/Point/HoWaitBegin/HoWaitEnd/Finish 及共享 Tik/Tok | 与前述逐节点/逐起点表一致；无额外业务循环打点；无动态枚举/计数器 |

第一次按无 `-X utf8` 的原命令执行前向消费合同时，Windows 默认 GBK 导致两项 UnicodeDecodeError；开启 UTF-8 后两项通过。它是本机文本读取问题，不能隐藏为原命令直接通过；生产 Linux 命令不因此发生源码变更。

## 尚未验证与交付判断

当前无本地 CANN/NPU/torch 和编译工具链，且用户明确本轮仅源码、生产验证环境不可接入。因此未执行 CANN 编译/安装、实际 Profile 符号加载、两种 SoC/路由的设备选择、核映射实测、GM 边界内存检测、ENABLE_TIMER=0 构建对照、CPU/六算子精度与确定性验收、真实 raw/CSV/Perfetto、时钟确认或打点扰动测量。这些事项仍按 UT README 和仓库阶段 05 留待设备验证。

Step 9 因上述新增 mutation 合同问题暂不通过。修复并重跑后才能给出源码通过结论；已有合同失败和设备待验证项均保留，不将其改写为完成。

## 第二轮最终复核（修复后，当前有效）

本轮先读取根 `AGENTS.md`、`docs/agents/README.md` 与阶段 01/03/04/05、runtime 架构、开发者指南、项目级 `op-timer-ascendc/SKILL.md` 和迁移指南，再读取本算子的 `api.md`、`design.md`、`timer_flow.md`、UT/ATK README。主判据仍是 `timer_flow.md:47` 原图、`:113` 方案 A、`:135` 清单、`:195` 枚举映射及 `:289` 最终落地映射。以下均为本轮实际工作树行号；没有把第一轮报告当作源码证据。

### 第一轮 P2 修复复核

| 检查 | 当前源码与执行证据 | 结论 |
| --- | --- | --- |
| mutation 清单 | `torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py:97`–`:99` 为私有 `_profile_chunk_gated_delta_rule_fwd` 登记 `("timer",)`；独立 AST 提取清单后验证同一值 | 通过；没有遗漏私有诊断 mutation |
| 成功返回后 version bump | `_aclnn_ctypes.py:2977` 先执行实现；`:2980` 再 `increment_version((timer,))`；`:2981` 返回原 outputs。独立编译该 helper AST，以 mock torch/实现记录顺序，观察 `returned` 后恰一项 `bump`，返回十元组保持 | 通过；helper 自身维护版本，不依赖未经过的公共 facade，也没有二次 bump |
| launch 失败 | 同文件 `:2977` 抛异常时不会到达 `:2980`。独立 mock 抛 `RuntimeError("launch failed")`，仅观察一次失败实现调用、零 bump | 通过 |
| 参数与 grad 失败 | `:2971`–`:2976` 在调用实现前检查 dtype/rank/连续性、同 NPU device 和 requires_grad。独立 mock 分别注入坏 dtype/rank/device/grad，观察零实现调用、零 bump | 通过；INT64 不作为省略版本维护的理由 |
| 合同测试 | `tests/ut/test_timer_contract.py:210` 成功后 once-with 断言，`:230`–`:248` launch/grad 失败不 bump，`:250`–`:255` 私有 mutation metadata；坏 timer 在 `:212`–`:223` 拒绝且不 launch。独立检查额外验证了坏参数不 bump | 通过；本轮套件实际 Ran 20 tests，OK |
| 架构和调用契约 | `docs/architecture/torch-npu-decoupled-architecture.md:327`–`:333` 明确私有 helper 的独立维护边界与 eager 范围；`docs/api.md:52`–`:54` 与 UT README 对成功/失败/grad 的表述一致 | 通过；未宣称 graph capture、compile/export 支持 |

因此第一轮 P2 已关闭。保留上面的原始缺陷和 mock“零 bump”证据作为历史，不能把修复前结果改写为原本通过。

### 全部实际观察节点重新核对

下表路径相对于 `fla/ops/ascendc/gdn/chunk_gdn_fwd/`；共享实现另列全路径。

| 原图 / 方案 A 位置 | 本轮实际代码位置及分支核对 | 结果 |
| --- | --- | --- |
| host 路由与实际 kernel 链 | `chunk_gated_delta_rule_fwd/op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp:105/:697` 选择两路；`:763/:773/:779` 将完整 timer 传给 Prepare/H/O；`:880` 传 Phase6；正式入口 `:932` 默认 null | 两路均覆盖；没有把旧同级 H/O 误作实际路由 |
| Prepare INIT → pack 主体 → 最后 W/U drain | `chunk_gated_delta_rule_fwd_prepare/op_kernel/chunk_gated_delta_rule_fwd_prepare.cpp:27/:33/:34/:35` 为 Init/InitDone/Process/Finish；原 `arch35/chunk_gated_delta_rule_fwd_prepare.h:807/:824` 的尾 W/U drain 仍在 Process 内 | 原 pack/延迟 drain 包含在 BODY；无额外循环事件 |
| H 有工作 AIC、idle AIC、AIV | `chunk_fwd_h/op_kernel/chunk_fwd_h.cpp:93` Init；`:54` 先 InitDone 后 `:55` idle 内层返回；`:64/:76` 是两个有工作分支的 InitDone；`:129` 外层共同 Finish | 每条合法核型路径恰一次 InitDone/Finish；idle AIC 无未收尾 Tik |
| H 单 head、多 head、首末 chunk 和尾事件 | `chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h:669/:709` 和 `chunk_fwd_h_vec.h:1148` 的原工作循环均未修改；vector Process 的 `:381` DrainLocalEvents 在返回前执行 | 业务条件和 loop 轴保持；都被整体 BODY 包含 |
| O 旧服务与 A5 token-first | `chunk_fwd_o/op_kernel/chunk_fwd_o.cpp:109/:116` 共同 Init/Finish；`:60` 旧服务 InitDone；`arch35/chunk_fwd_o_a5.h:59/:67` 分别为 AIV/AIC 服务 InitDone；`:61` AIV 内层返回回到外层 Finish | 每个实际分支收尾完整；原 owner/non-owner 握手和尾 drain 保持 |
| arch22 Phase6 | `chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp:479/:497/:632/:633/:637/:677` 对应总起点、首 KKT 前 InitDone、前端瞬时点、私有 H、H 返回点、末尾 Finish；`:695/:700/:705/:710` 四个 key 都传 timer | prepared-cumsum、前端分批、BT64/128、V128/256 与导出分支全部被包围 |
| arch35 Phase6 | `chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:332/:353/:446/:447/:451/:479` 同上；`:497/:502/:511` key 1/2/301 都传 timer | B0/B30、BT64/128、V128/256 与导出分支全部被包围 |
| 串行 H/O 的原 SyncAll | arch22 `:639/:640/:644/:647/:649`，arch35 `:453/:454/:458/:461/:463`：Begin 和 End 在同一串行分支，包围同一原 SyncAll | 配对完整；TokNoBarrier 在原完成锚点之后 |
| 重叠 / 条件 O | arch22 `:663/:669/:675/:677`：producer 可以不跑 O，仍到原最终会合和 Finish；arch35 `:474/:477/:479`：原重叠最终会合之后 Finish | 不记录 HO_WAIT，不新增跨核同步，未遗漏 producer/idle 分支 |

精确 `rg` 扫描 `timer.Init/InitDone/Point/HoWaitBegin/HoWaitEnd/Finish` 的全部实际调用，与上表及原图逐项匹配。共有五处 Init、五处 Finish，包含 H/O 核型的互斥 InitDone；两种 Phase6 各两处瞬时点和一对条件等待。未发现额外打点、漏标、错位或迭代轴不符。方案 A 无动态项；图中的 pack/head/chunk/groupTask 业务循环没有新增迭代槽。方案 B 明确未实施。

O 的合法模板覆盖也已核对：`chunk_fwd_o/op_kernel/chunk_fwd_o_tiling_key.h:20`–`:24` 仅注册 `(USE_EXP2, OUTPUT_TOKEN_FIRST)=(0,0)/(0,1)/(1,1)`；`op_host/chunk_fwd_o_tiling.cpp:80` 与 `chunk_fwd_o_tiling_processor.h:318`–`:323` 拒绝非 A5 token-first 和非 token-first exp2 组合。因此没有把不会执行服务 InitDone 的非法模板组合计为已覆盖路径。

### 每个实际 Tik/Tok、布局和入口合同重新核对

| 项目 | 当前文件:行 / 本轮证据 | 结果 |
| --- | --- | --- |
| 总窗口 | `fla/ops/ascendc/common/kernel_utils/gdn_timer/AscendTimerV2_device.hpp:144` all Tik → `:182` all Tok；Tok `:78` 先判有效，`:82` 唯一新增局部 PIPE_ALL，`:83/:90` 读取结束 cycle | 所有合法 entry 共同 Finish；不存在无终点总 Tik |
| INIT / BODY | 同文件 `:145` INIT Tik → `:152` NoBarrier Tok；`:153` BODY Tik → `:183` NoBarrier Tok，结束值为 all.GetEndCycle | INIT 明确为 issue；BODY 和 KERNEL 共享同一已排空末尾 timestamp，没有第二次 drain 或越窗 |
| 两个瞬时点 | 同文件 `:160/:161` 紧邻 boundary Tik/Tok，对应两 Phase6 FRONT_DONE/H_DISPATCH_DONE 调用 | 每次局部成对、不同槽位、Overwrite；不把下发边界解释为 pipeline 完成 |
| 原串行等待 | 同文件 `:168/:175` Begin/End；上一表两架构同一条件分支 | 起点 NoBarrier，锚点后 NoBarrier Tok；重叠分支两端均不调用 |
| 枚举 / parser / 清单 / 图 | canonical `AscendTimerV2.hpp:11` 六项；parser `:60/:62` 派生名称；图清单 `timer_flow.md:135/:195`。依次为 KERNEL_TIMING、MAIN_INIT、MAIN_BODY、PHASE6_FRONT_DONE、PHASE6_H_DISPATCH_DONE、PHASE6_HO_WAIT | 数量、顺序、名称完全匹配；KERNEL_TIMING_IDX 的显示改名明确。无动态枚举或 setDynamicActualIter；无 Accumulate 调用 |
| slice / core / 地址 | canonical `:21` 四 slice，`:34/:35` 容量 256×3；device `:24`–`:32` AIC group 与 AIV group/subblock 映射，`:52` 超限禁写，`:104` 16-word stride；独立遍历四 slice 全容量所有事件 word | 3,072 行和所有事件地址唯一；最高 word 49,211 < TOTAL_BUFFER_SIZE 49,216；不重叠、不回绕、每行 4 padding word |
| metadata | device `:34`–`:49` 仅 AIC group 0 写 header；stage/boundary `:141/:142` BindFrom 不再次 Init | 唯一写者；记录实际 blockDim/arch/layout；其他核不读，无需新跨核同步 |
| null / ENABLE_TIMER=0 | device `:21/:69/:78/:99` null 在读 clock/barrier/写 GM 前返回；canonical `:46/:50` 可覆写默认和 #if；GdnKernelTimer `:140/:151/:159/:167/:174/:181/:188` 条件编译动作和成员 | 静态无计时读写/barrier 可达路径；没有伪称已编译 ENABLE_TIMER=0 |
| host 验证 | 主 ACLNN cpp `:643`–`:648` INT64、rank 1、TOTAL_BUFFER_SIZE、ND、contiguous；Profile `:981` 拒绝 null。Python helper `:2971`–`:2976` 同 NPU device 等校验 | 使用 canonical 常量；未连续化 timer 副本；直接 C 调用者同设备和持有责任已归档 |
| OpDef / L0 / raw | 主/Prepare/H/O 的 def `:46/:50/:77/:77` 最后 optional INT64 ND；L0 cpp `:75/:125/:108/:64` 最后 OP_INPUT 为 timer；raw 主 arch22 `:685`/arch35 `:487`、Prepare `:20`、H `:88`、O `:103` 同一输入位置 | 四方顺序一致、原输出/属性索引保持；独立公开子算子由新增默认参数传 null |
| 正式 ABI / 诊断 ABI / 十项输出 | 独立 AST 对比 HEAD 的正式 wrapper 参数/defaults及原计算、校验、分配前缀；独立 C 声明比对保留旧两段接口；ctypes `:564` 独立 Profile argtypes 追加一个 descriptor；实现 `:3123/:3155/:3158/:3159` 仍十项 tuple，timer 仅追加诊断参数 | 旧 ABI 未插参；Profile 为独立符号；输出/别名合同的 mock 回归通过；实际符号加载待构建 |
| common / 扁平 OPP include | `cmake/config.cmake:41` common 根，`cmake/func.cmake:379` copy，`cmake/custom_build.cmake:957` / `CMakeLists.txt:550` install；两个转发头 `:2/:5`；独立解析 source 相对路径及 common fallback | 两种引用都解析到 canonical 文件；不依赖部署后保留源树。实际 OPP 安装/编译仍未验证 |

每个起点均使用 NoBarrier，固定事件全部 Overwrite；stage、all、boundary 的在途状态独立。没有发现无起点 Tok、无收尾 Tik 或额外业务同步。对 HEAD 与最终源码按顺序提取原同步/event 语句，arch22 34 条、arch35 25 条完全相同；另外四个被修改 entry/service 文件的同步/event 内容也相同。本轮按实际调用语句计数；第一轮 arch22 的 35 条历史统计不作为当前数量依据。

### Parser、trace、capture 重新核对

`parse_timer_csv.py:21/:60/:64` 从共享头派生布局和两组枚举；`:84`–`:151` 检查 raw 长度/INT64、magic/version/stride、容量、实际 blockDim 外写入、缺核、配对、非负时长、父窗包含、padding、INIT/BODY 与 dispatch 顺序、等待分支一致及互斥/完整路由。所有实际启用 fixed 项按分支要求存在；未执行 HO_WAIT 允许保持全零。`:131` 将 INIT 标为 issue、边界标为 point；`:179` 汇总跳过 point duration。

`trace_parser.py:16` 要求已确认的正有限频率和 shared clock；`:37` 先做整数 origin 相减，`:42` 用瞬时对中点；`:46` 明确每个 kernel 独立 origin。`profile_gdn.py:49/:54/:55/:57/:59` 同 device context、预同步、全零独立 buffer、诊断调用后同步再复制；`:63` 先保存 raw；`:65`–`:82` 保留输入/库路径/选项/时钟断言/hash；`:86`–`:97` exact 比较十项输出，失败保留 raw。未确认频率/共同时钟时仅输出 cycle。离线 fixture 的 CSV/JSON 导出通过，未冒充设备实测或跨 kernel launch 时间线。

### 本轮执行的命令与结果

| 命令 / 独立检查 | 实际结果 |
| --- | --- |
| `python -X utf8 -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p test_timer_contract.py` | exit 0；Ran 20 tests in 0.298s；OK |
| `python -X utf8 -m unittest discover -s ci/tests -p test_gdn_forward_output_contract.py` | exit 0；Ran 2 tests in 0.057s；OK |
| `python -X utf8 -m unittest discover -s torch_custom/fla_npu/test -p test_ascendc_mutation_contract.py` | exit 0；Ran 7 tests in 0.115s；OK。mock torch 缺 `_C` 的 Stable-ABI fallback RuntimeWarning 是离线测试输出，不是设备加载证据 |
| `python -X utf8 tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py` | exit 1；同一既有 `prepare kernel 缺少非 exp2 模式的局部指数换底选择`，脚本 `:191`；其检查的 Prepare 算法头独立与 HEAD 文本比较相同 |
| `git -c core.autocrlf=false diff --check` | exit 0，无输出 |
| 独立 `python -X utf8 -` 的全容量枚举、HEAD AST/C 声明、同步序列、helper mock 和语法检查 | exit 0；`4 slices / 3072 unique core rows; highest event word 49211 < 49216; no overlap`；正式参数/defaults与计算前缀/原 C ABI 保持；同步语句内容和顺序保持；成功一次 bump、失败零 bump；五个 Python 文件 compile() 通过 |
| 独立 `python -X utf8 -` 的转发路径和既有失败目标比较 | exit 0；source-tree 相对 include 与 flattened common fallback 都解析到 canonical header；既有失败检查的 Prepare 算法头文本未变化 |
| `rg -n` 精确扫描全部 timer 生命周期和 Tik/Tok、结合逐行阅读两 Phase6 与 raw/service 分支 | 五处 Init/Finish 和全部互斥 InitDone、四个 Point、两对 HoWait 均与原图/A 清单匹配；无动态项和额外同步 |

当前最终结论为**源码 Step 9 通过**，第一轮 P2 修复已独立确认；未发现仍需修复的本次源码阻断项。既有 Prepare 合同失败继续如实归档，不能据此声明原有合同全部通过。用户允许的源码交付范围已满足；CANN 编译/安装与实际 Profile 符号、SoC/路由/核映射、GM 边界、ENABLE_TIMER=0、真实 raw/CSV/Perfetto、频率和共同时钟、精度/确定性/内存/打点扰动仍未验证，按 UT README 留待可用设备环境执行。

## 2026-10-12 ATK 输入导出扩展独立审核

本节由未参与输入导出实现的独立子 agent 审核并追加，保留以上原始缺陷、修复和复核历史。
依据根 `AGENTS.md`、`docs/agents/05-算子测试.md`、通用及本算子 ATK README，
以及项目 `.agents/skills/op-timer-ascendc/SKILL.md` Step 9；重新读取当前 diff、真实性能 JSON、
`gdn_reference.py`、`profile_gdn.py`、`timer_flow.md` 和实际 host/kernel/共享计时头。
本次范围是 executor 的可选快照、离线合同和安装 wheel 后的 A2 操作说明，未修改数学输入规则或计时实现。

### 快照与 DUT 合同逐项核对

下表 executor 路径为 `tests/atk/chunk_gated_delta_rule_fwd/executor_chunk_gated_delta_rule_fwd.py`。

| 项目 | 独立源码证据 | 结论 |
| --- | --- | --- |
| q/k/v/g/beta 实际值与 dtype | `build_inputs:85` 调用既有 `effective_inputs`；其 `gdn_reference.py:112`–`:129` 在 CPU 冻结 q/k/v/beta 为 public dtype、g 为 FP32。初始化 `:309` 向快照传 `self._inputs`，`:173`–`:176` 使用 detach/cpu/contiguous，不重新抽样或转换 dtype；NPU DUT 使用同一 inputs 的同 dtype 设备副本 | 同输入合同；不是 JSON seed 重建 |
| 初态 | 导出 `:179` 和 DUT `:203` 均调用 `deterministic_initial_state(case)`；既有函数由 scenario、sequence_count/Hv/K/V 生成确定性 FP32 初态，DUT 仅移动设备并连续化 | 初态值、shape、dtype 一致 |
| final_state、scale、chunk_size | 快照 `:176`–`:183` 和 DUT `:234`–`:246` 均来自同一个 GdnCase；profiling `main` 读取保存的 kwargs，未用 CLI 默认值替换 | 保留原调用参数 |
| layout 与隐式选项 | 保存 `layout="BNSD"`，与原正式 ctypes wrapper `:2942` 默认值相同；exp2、L2Norm、gate/beta 处理等未传选项仍使用同一 wrapper/impl 默认值 | 与原 DUT 行为一致 |
| cu_seqlens / chunk_indices | 导出和 DUT 均将 offsets 转 list，并调用同一 `canonical_chunk_indices(case.cu_seqlens, case.chunk_size)`；定长两者均为 None | 变长 sequence-major 顺序保持 |
| 默认关闭、其它角色隔离 | `:161` 仅 DUT 读取 GDN_PROFILE_INPUT_DIR；空变量立即返回。golden、benchmark 初始化提前返回，调用循环不包含 export | 默认无文件；非 DUT 不导出 |
| 性能循环与原行为 | 独立 AST 比对 HEAD，`build_inputs`、`run_npu`、FunctionApi 的 `__init__`、`__call__`、`export_custom_data` 完全相同。唯一 export 调用在 `init_by_input_data:309` | 导出不进入 `__call__` 性能循环 |
| 保存首份与路径 | `:165` 校验绝对路径；`:169` 先检查已有文件，`:189` 独占 `xb` 打开并处理 FileExistsError | 已有同名快照不覆盖；README 每轮新 RUN_DIR |
| --inputs 消费 | `profile_gdn.py:134` weights_only=True/map_location=cpu 读取 tensor 和原生标量/list/None，`:135` 移动五项输入，`:137` 移动初态；capture 使用 kwargs 调用正式与诊断入口 | 格式与已有入口兼容；真实 torch.save/load 仍待目标环境 |

真实 `_perf.json` 仍只有两条，且与 HEAD 文本相同：case 0 为 BF16、B/Hk/Hv/T/K/V=1/16/32/11274/128/128、chunk=64、`cu_seqlens=[0,11274]`、确定性初态与 final_state；case 1 为 BF16、2/32/32/8192/128/128、chunk=64、定长无初态和 final_state。scale 均为 0.08838834764831843。文档明确这些原模型来自 A5，本轮命令在 A2 重放合同，不引用 A5 性能作为 A2 结论。

### 原计时节点与方案 A 重新核对

独立使用 `git ls-files` 列举主算子、Prepare、实际 ChunkFwdH/ChunkFwdO 的 host/kernel 和共享 timer 共 **168 个文件**，逐文件以 `git show HEAD:<path>` 和工作树 UTF-8 文本比较（统一 CRLF/LF）：全部相同。`profile_gdn.py`、parser、trace、CPU reference 和性能 JSON 同样与 HEAD 相同。以下再次读实际代码，不以旧审核结论代替定位证据。

| 原图/方案 A 节点 | 本轮实际位置 | 配对与锚点结论 |
| --- | --- | --- |
| Prepare INIT → pack/尾 drain → BODY 收尾 | Prepare raw `:27/:33/:35`，中间 `op.Process` | Init/InitDone/Finish 与图一致；无新增 pack 内事件 |
| H INIT → chunk 主体 → 收尾 | ChunkFwdH raw `:93/:129`；idle AIC `:54`、有效 AIC `:64`、AIV `:76` InitDone | idle 内层 return 仍回到外层 Finish；每条合法核分支恰一次初始化终点与总收尾 |
| O INIT → groupTask 主体 → 收尾 | ChunkFwdO raw `:109/:116`；旧服务 `:60`、A5 vector/cube `:59/:67` InitDone | 核型/服务分支保持，内层 return 不漏外层收尾 |
| arch22 Phase6 前端/H/O | `:479/:497/:632/:637/:677` | 总窗口、初始化、FRONT_DONE、H_DISPATCH_DONE、Finish 与方案 A 对齐 |
| arch35 Phase6 前端/H/O | `:332/:353/:446/:451/:479` | 同上；两架构均无业务循环新增打点 |
| 串行 H/O 等待 | arch22 `:640/:644/:647/:649`；arch35 `:454/:458/:461/:463` | 同一串行条件分支包围原 SyncAll；重叠分支不制造 HO_WAIT，原最终同步仍在 Finish 前 |
| 共享 Tik/Tok | canonical device header `:144` all Tik → `:182` all Tok；`:145` INIT Tik → `:152` Tok；`:153` BODY Tik → `:183` Tok；`:160/:161` Point；`:168/:175` HoWait | 每个起点有对应终点；Point 使用相邻瞬时对；唯一局部 PIPE_ALL 位于总 Tok，BODY 共用末尾时间戳 |
| 枚举/命名/清单/图 | canonical header 六项、parser `:62` 派生命名、timer_flow 第 3/6 节 | KERNEL_TIMING、MAIN_INIT、MAIN_BODY、PHASE6_FRONT_DONE、PHASE6_H_DISPATCH_DONE、PHASE6_HO_WAIT 的数量/顺序/名称一致；无动态枚举、setDynamicActualIter 或实际 Accumulate |

### 安装后命令与执行证据

主 UT README 已给出 Linux/Bash、仓库根目录、同一已安装 wheel 的 Python/ATK 环境、CANN set_env、ascend910b、设备变量及绝对 RUN_DIR；明确同步仓库 executor。PYTHONPATH 增加本 ATK 算子目录，供 worker 解析 reference/role 模块。使用 find_spec 定位 wheel OPP，在新 Python 进程前注册 ASCEND_CUSTOM_OPP_PATH，并调用实际 `load_ascendc_opapi_libraries` 检查两个 Profile 符号，与 host header `:54/:86` 及实现 `:949/:990` 名称相同；本轮仅核对源码存在，没有实际加载库。

范围前缀清空通用和性能 start/end，统一脚本只执行 performance_device 的完整两条 JSON，不启动 CPU/六 ACLNN benchmark；可选 case 0 使用 start=0/end=1。脚本使用该性能 JSON 和 DUT executor，将 ATK 输出置于 `$RUN_DIR/atk/perf`，快照及两 case 的独立 `--inputs` 采集路径完全一致。raw/CSV/layout/capture metadata/日志的实际文件名与采集代码一致；默认 cycles，只有核实时钟频率和 AIC/AIV 共同时钟后才请求 Perfetto，明确 kernel 各自 origin 和瞬时边界含义。源码归档没有 .git 时，提交号记录使用 if/else，不阻断设备命令。

| 本轮独立执行的命令/检查 | 实际输出 |
| --- | --- |
| `python -X utf8 -m unittest discover -s ci/tests -p test_gdn_profile_input_export.py -v` | exit 0；Ran 5 tests in 0.047s；OK |
| `python -X utf8 -m unittest discover -s ci/tests -p test_gdn_forward_output_contract.py -v` | exit 0；Ran 2 tests in 0.032s；OK |
| `python -X utf8 -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p test_timer_contract.py -v` | exit 0；Ran 20 tests in 0.137s；OK |
| 独立 `python -X utf8 -` AST、compile、HEAD 文本及 Markdown 链接检查 | build_inputs/run_npu/三个 FunctionApi 方法 AST 相同；新增测试与 executor compile() 通过；168 个 host/kernel/timer 文件及采集/parser/reference/perf JSON 不变；两个 README 的 4 个相对 Markdown 链接目标存在，ATK→UT 标题锚点匹配 |
| 提取主 UT README 的 Bash 代码块，逐项通过 Git Bash `--noprofile --norc -n`；随后单独重验最新提交号 if/else 块 | 10 个块通过；最新第 3 块也通过。首次沙箱检查被 Git Bash signal pipe 权限阻止，获授权只读检查后通过；`-n` 没有执行任何设备/安装命令 |
| `git diff --check` | exit 0，无输出。强制关闭 autocrlf 的额外尝试会将本 Windows 工作树 CRLF 误报为尾空白，未将该尝试当作实现缺陷 |

本次扩展的**源码独立审核通过**，未发现需要修复的新增阻断项。以上 27 项是无 torch/ATK/CANN/NPU 的离线合同，不能证明真实张量序列化、设备 worker 环境继承、wheel/OPP 安装及 Profile 符号加载、两 case 实际运行、kernel 路由/核映射、GM 边界、数值验收、真实 raw/CSV/Perfetto、时钟或观察扰动通过。原审核中的既有 Prepare 合同失败和设备待验事项继续保留；用户本轮只授权源码且生产设备不可接入，本轮没有设备验收或性能结论。

## 2026-10-12 A5 / Ascend950PR / Docker 单入口独立审核

本节由未参与入口实现的 fresh-context 子 agent 按 skill Step 9 独立审核并追加，保留以上历史。
审核依据为根 `AGENTS.md`、`docs/agents/05-算子测试.md`、通用和本算子 ATK README、
项目 `.agents/skills/op-timer-ascendc/SKILL.md` Step 9，以及构建部分的 `docs/开发者指南.md` 和实际脚本。
重新读取真实性能 JSON、旧 executor/reference/capture/parser/trace、原始流程图和实际计时代码。
审核基准 HEAD 为 `d1d9c2545d15690319f1f6138e0a3839a0d0f525`；范围是两个 README、
新增 `scripts/profile_performance.py` 和 `ci/tests/test_gdn_profile_performance.py`。
本机未接入生产 Docker、ATK、CANN 或 NPU，仅能进行源码和离线验证。

### 本轮发现的问题、修复及复核

1. **P2：产物校验和实际加载可能使用不同 OPP。** 初版保留 `FLA_NPU_OPP_PATH`，
   而 `scripts/verify_libcust_opapi_md5.py:146` 优先检查该覆盖目录；实际
   `torch_custom/fla_npu/fla_npu/__init__.py:24/:70/:140` 只加载当前包内嵌 OPP。
   因此调试覆盖可能造成错误拒绝，或用新覆盖目录替旧 wheel 通过校验。
   最终入口 `profile_performance.py:162` 清除该变量；模拟管线注入不同调试路径并断言子进程中不存在，
   README 同步说明。独立复核确认后续 MD5、符号和 capture 都指向当前 Python 已安装 wheel。
2. **P2：PATH 中的 ATK 可能属于另一 Python。** 初版仅检查 CLI 存在，
   `ATK_ENV` 未设置时可能选中全局或另一环境的 ATK，导致 worker 与后续 capture 环境不一致。
   最终 `verify_atk_launcher:116` 同时校验 launcher 的实际 bin、shebang 解释器目录和实际解释器文件，
   支持通过当前子进程 PATH 解析 `env python3` / `env -S python3`，无法确认时提前拒绝。
   `ATK_PROBE:23` 和调用 `:187` 还使用当前 Python 导入 executor 所需四组 ATK API，并保存 `atk_environment.log`。
   三项新增合同覆盖正常、跨环境/未知解释器和提前失败；独立额外模拟 venv/global Python
   指向同一实际文件但目录不同的情形，确认仍报 `current Python environment`，没有仅凭 inode 相同放行。

两项问题均由实现者修复后重新独立校验；审核者未修改入口、测试或算子实现。

### Docker、环境、ATK 和同输入诊断逐项核对

| 项目 | 独立证据 | 结论 |
| --- | --- | --- |
| Docker 主流程 | 主 UT README 的宿主机 `docker ps` / `docker exec -it -w`，后续容器 shell 与源码目录前提 | 使用已有运行容器；容器须已透传 NPU、包含本次源码并激活其 Python 环境；本机未实际执行 |
| A5 编包安装 | README 使用 `FLA_NPU_SOC=ascend950 FLA_NPU_OPS= python scripts/build_wheel.py`，与构建脚本、发行档位和开发者指南一致；安装准确 wheel 文件名并带 force-reinstall/no-cache-dir/no-deps | 显式清除可能遗留的局部编包范围；全量 Python/host/kernel/OPP 后仅需运行一个采集入口；A2 发行包先卸载再装 A5 的顺序已说明 |
| ROOT 与路径 | `profile_performance.py:18` 的 `parents[4]` 实算为仓库根；`:19/:20` 指向既有 ATK 与主算子 UT；所有子命令 cwd 为仓库根 | 不依赖调用者的工作目录；自定义输出和默认快照路径均解析为绝对路径 |
| 环境继承与 OPP | 复制已激活环境，前置当前 Python bin，仅追加本算子 ATK 源目录到 PYTHONPATH；清除环境重新 source 开关和调试 OPP；定位 wheel 后在任何 NPU 子进程启动前注册 vendor | 统一 runner 不会再次切换 ATK/CANN/旧 OPP；同 Python 的 ATK probe、NPU 检查、MD5 和两 Profile 符号探测均有独立日志 |
| ATK 参数与范围 | `:195` 调用原 `run_test_cpu.sh`，固定 `-soc=ascend950 -scope=performance` 并传设备；清空通用和性能范围；单 case 使用 start=id/end=id+1 | 对应原 `performance_device` NPU DUT 节点；默认完整 case 0/1，不启动 CPU golden 或六 ACLNN benchmark，不生成新 JSON |
| 原模型合同 | JSON 与 HEAD 相同：case 0 为 BF16、1/16/32/11274/128/128、chunk 64、cu_seqlens=[0,11274]、确定性初态和 final_state；case 1 为 BF16、2/32/32/8192/128/128、chunk 64、定长无状态；scale 均为 0.08838834764831843 | README 表格与真实资产一致；不将其它 SoC 的旧结果作为 A5 结论 |
| 真正同输入 | 旧 executor `:173` 保存冻结 q/k/v/g/beta，`:176` 保存原 kwargs，`:309` 仅在 DUT 初始化调用 export；capture `profile_gdn.py:134` 读取快照，`:135/:137` 移动原张量及初态 | 保留实际值、dtype、scale、chunk_size、初态、final_state、cu_seqlens、chunk_indices；不通过 JSON/seed 重建，性能循环仍无导出 |
| 每 case 独立采集 | 入口 `:198/:207` 确认快照后按 case 启动当前 Python 的 `profile_gdn.py --inputs --check-output`，默认 warmup 5 | 正式和诊断调用使用同份输入，沿用旧十项 exact 对比；每 case 有独立 timer/output/log；原 capture 在解析/输出对比前保存 raw |
| 失败、日志与保留 | `run_logged:67` 同步流式打印并 flush 日志，非零子进程抛错；新输出目录拒绝覆盖；缺快照/缺产物失败，metadata 记录 failed；main 返回 1 | 不会在 ATK 失败或输入未导出后继续诊断，也不会以缺失 CSV 声明 complete；既有结果目录保持 |
| 大快照 hash | `file_sha256:108` 按 1 MiB 读取，合同对 2 MiB+17 byte 文件验证摘要 | 不将模型 .pt 整体读入内存；复制性能 JSON并保存其 hash及输入 hash |
| 单位与 trace | 默认 cycles；只有正有限 cycles-per-us 才换算，shared-clock 必须伴随频率；两参数逐 case 转发；结果名与 parser/capture 一致 | 未核实时钟时没有虚构微秒/Perfetto；每 kernel 独立 origin，瞬时 dispatch 点不误作业务完成耗时，正式性能仍读 ATK op_summary |

安装检查复用既有 `verify_libcust_opapi_md5.py`：本流程先在同一源码目录全量编包，再比较已安装产物。
该检查器在缺少本地 build 参照时会 WARN 并跳过相应比较；本轮未修改其行为，不能把 WARN 或符号存在等同于已完成设备安装验收。

### 原图、Tik/Tok、枚举和流水保持

独立通过 `git ls-files` 列举主算子、Prepare、实际 ChunkFwdH/ChunkFwdO 的 host/kernel 以及共享 timer，
逐文件与 `git show HEAD:<path>` 的 UTF-8 文本比较并统一 CRLF/LF：**168 个文件全部相同**。
原 executor、CPU reference、性能 JSON、`profile_gdn.py`、parser、trace 和 `timer_flow.md` 七个文件也完全相同。
以下重新定位实际源码和原图第 3/6 节，并非仅沿用历史审核结论。

| 原图方案 A 节点 | 当前源码位置 | 配对与完成锚点 |
| --- | --- | --- |
| Prepare Init/pack/尾 drain | Prepare raw `:27/:33/:35` | Init → op.Init → InitDone → op.Process → Finish；pack 轴及最后 W/U drain 没有新增事件 |
| H Init/work-unit/chunk | H raw `:93/:129`；idle AIC `:54`、有效 AIC `:64`、AIV `:76` InitDone | idle 内层返回仍到共同 Finish；每合法核分支有初始化终点和总收尾 |
| O Init/groupTask/head | O raw `:109/:116`、旧服务 `:60`；A5 服务 `:59/:67` InitDone | AIC/AIV、owner/non-owner 和内层返回保持；外层共同收尾 |
| 两种 Phase6 前端/H/O | arch22 `:479/:497/:632/:637/:677`；arch35 `:332/:353/:446/:451/:479` | 五处总 Init/Finish、全部互斥 InitDone、四个 Point 与图匹配；不新增业务循环槽 |
| 串行 H/O SyncAll | arch22 `:640/:649`；arch35 `:454/:463` | Begin/End 在同一串行条件分支，包围原 SyncAll；重叠分支不制造 HO_WAIT，原最后会合保留 |
| 实际 Tik/Tok | canonical device header `:144`→`:182` all；`:145`→`:152` INIT；`:153`→`:183` BODY；`:160/:161` Point；`:168/:175` HoWait | 起点均 NoBarrier；固定槽均 Overwrite；总 Tok 唯一局部 PIPE_ALL drain，BODY 共用 endCycle，Point 相邻成对；无未收尾 Tik 或无起点 Tok |
| 枚举/命名/清单/图 | canonical header `:11`、parser `:60/:62`、timer_flow 第 3/6 节 | 六槽顺序为 KERNEL_TIMING、MAIN_INIT、MAIN_BODY、PHASE6_FRONT_DONE、PHASE6_H_DISPATCH_DONE、PHASE6_HO_WAIT；数量/顺序/名称一致，无实际动态计数或 Accumulate 打点 |

### 最终独立执行证据

| 命令或独立检查 | 实际输出 |
| --- | --- |
| `python -X utf8 -m unittest discover -s ci/tests -p test_gdn_profile_performance.py -v` | 修复后 exit 0；Ran 12 tests in 0.117s；OK。初版 9 项亦通过，但缺少上述 ATK 归属覆盖，未将初版通过视为审核完成 |
| `python -X utf8 -m unittest discover -s ci/tests -p test_gdn_profile_input_export.py -v` | exit 0；Ran 5 tests in 0.043s；OK |
| `python -X utf8 -m unittest discover -s ci/tests -p test_gdn_forward_output_contract.py -v` | exit 0；Ran 2 tests in 0.026s；OK |
| `python -X utf8 -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p test_timer_contract.py -v` | exit 0；Ran 20 tests in 0.141s；OK |
| 独立 `python -X utf8 -` compile、ROOT、CLI、ATK import AST 和 shebang模拟 | 新两文件 compile()、入口 --help、ROOT/子路径通过；ATK_PROBE import 集合与 executor 完全相同；venv/global 同 inode 不同 bin 的模拟被拒绝 |
| 独立 HEAD 文本比较、`rg -n` 定位和 README 链接检查 | 168 个 host/kernel/timer + 七个原链路文件不变；全部原打点对应原图；两个 README 的八个相对链接目标存在 |
| 提取最新主 UT README 的 Bash 块，调用 Git Bash `--noprofile --norc -n` | exit 0；六个块语法通过，未执行任何 Docker/安装/设备命令。初次沙箱 signal pipe 被拒绝，后经 auto-review 批准仅只读语法检查后通过 |
| `git diff --check` | exit 0，无输出 |

最终结论为本轮**源码 Step 9 独立审核通过**；两个新增环境 P2 均已修复并独立重验。
39 项是模拟/离线合同，不构成正式 ATK 阶段 05 验收或性能达标结论。
没有实际验证 Docker 进入/透传、CANN 编包、wheel 安装/真实符号加载、ATK worker 环境继承、
真实 torch.save/load、两条 A5 模型实测、真实 raw/CSV/trace、SoC 路由/核映射、GM 边界、
精度/确定性/内存、时钟频率/共同时钟或打点扰动。历史既有 Prepare ABI 合同失败和设备待验项继续保留。

## 2026-10-12 Prepare GM 发布修复独立审核

本轮由未参与实现的 Step 9 子 agent 使用 fresh context 执行，只追加此报告，未编辑实现。
先读取根 AGENTS、阶段 03/04/05、公共 runtime 架构、项目级 op-timer-ascendc skill，
再独立读取当前 api/design/timer_flow、UT/ATK README、git diff 和四个实际算子的源码。
主判据为 timer_flow 第 2 节原图、第 3 节方案 A 和第 6 节最终落地映射。

**结论：本轮源码 Step 9 审核通过，未发现新的阻断问题。45 项离线/模拟合同通过；
修复后 CANN 编译和设备完整采集仍待目标环境复测。**
上文历史缺陷、修复前观察、旧 Prepare 非 exp2 ABI 合同失败和设备待验事项全部保留。

### 用户旧 raw 与源码原因

用户反馈的 A5 / Ascend950PR case 0 统计为 49,216 word；PHASE6/PREPARE 全零；
FWD_H 非零 297 word、blockDim 16、arch 310，FWD_O 非零 513 word、blockDim 28、arch 310，
H/O 的每核三对事件完整。本轮接收的是这些旧版本统计，未在本机读取原设备 raw 文件或重新采集。

独立定位 `chunk_gated_delta_rule_fwd_prepare/op_host/CMakeLists.txt:25`–`:29`：
Ascend950PR_9599 仍配置 `-cce-aicore-dcci-before-kernel-end=false`。
HEAD 的 canonical timer 通过 GlobalTensor 标量写 metadata/时间戳，却没有显式 DCache 发布。
这与 Prepare 全零而 H/O 有完整记录的反馈相符，是源码中可确认的发布缺口；
不能据此声称已排除目标设备上的旧产物加载等其它因素，或已经证明修复后的硬件行为。

独立查看 [Ascend C DataCacheCleanAndInvalid 文档](https://asc.gitcode.com/api/SIMD-API/basic_api/cache_control/DataCacheCleanAndInvalid.html)：
页面列明 Ascend950PR 支持，Scalar 访问 GM 存在 DCache 一致性问题，Clean 将脏数据写回 GM，
GlobalTensor 原型及 64B SINGLE_CACHE_LINE 模式均与新增调用匹配。
该模式使用目标地址；页面还注明该模式下 dcciDst 不生效，故独占且对齐的地址是发布范围的关键。
页面来自持续开发版本，不能代替用户实际安装 CANN 的编译验证。

### 发布、布局、关闭开关与时序

本节共享设备头指 `fla/ops/ascendc/common/kernel_utils/gdn_timer/AscendTimerV2_device.hpp`，
公共常量头指同目录 `AscendTimerV2.hpp`。

| 检查 | 当前源码证据与结果 |
| --- | --- |
| header 发布 | 设备头 `:36`–`:50` 仅 AIC group 0 写 16-word header 并发布两条 cache line；GdnKernelTimer `:167` 绑定后才在 `:170/:171` 开始总窗口/INIT。其他核不写或读取 header，无新增会合 |
| 每核行发布 | `:126`–`:147` 从本核 coreId×16 word 构造 GlobalTensor，以 8 word 步长调用 SINGLE_CACHE_LINE；Finish `:208/:209` 先保存 KERNEL/BODY 结束时间戳，`:212` 再发布本核两条 line。发布开销位于测量窗口外，实际 launch 仍承担它 |
| 独占与边界 | 公共头的四 slice、六固定槽、布局版本 1、256×3 核容量、16-word header/row 和 TOTAL_BUFFER_SIZE=49,216 保持；每 slice 12,304 word 可整除 8。合同枚举所有 header/row 的 6,152 条缓存行，全部独占，无跨核/跨 slice line |
| 基址对齐 | 设备头 `:22` 在任何写入前拒绝非 64B 地址；capture `:56` 在诊断 launch 前拒绝不对齐的新 timer。api.md 明确直接 C 调用者的存储责任；这不是对任意偏移 tensor 已完成设备验证的声明 |
| null timer | Init 立即返回；Tik/Tok 和 FlushRow 均先检查有效地址，Tok 在 PIPE_ALL 前返回。正式调用无计时 GM 写入、cycle 读取、局部排空或缓存发布 |
| ENABLE_TIMER=0 | GdnKernelTimer 所有计时操作及成员都处于 #if ENABLE_TIMER；全部实际入口使用 TIMER_BLOCK。关闭时没有到达这些发布调用的实际入口路径；未做 CANN 关闭开关编译/设备对照 |
| 数学与同步 | 逐文件比较四算子 op_host/op_kernel 的 165 个 tracked 文件，统一 CRLF/LF 后全部与 HEAD 相同，包括 Prepare 编译选项、flags、SyncAll、计算循环与十项输出实现。当前 kernel 修改仅位于共享 timer 的对齐/定向发布逻辑 |

### 原图节点、全部 Tik/Tok 与四方一致性

路径相对于 `fla/ops/ascendc/gdn/chunk_gdn_fwd/`。四个实际算子中的 Phase6 有两个架构实现，
故下表包含五处总入口/收尾；没有给同级旧 H 算子替代实际 ChunkFwdH。

| 方案 A 原图节点 | 重新定位的实际调用 | 对照结果 |
| --- | --- | --- |
| Prepare Init/pack/尾 W/U drain | Prepare raw `:27/:33/:35` | Init → 真实 op.Init → InitDone → 完整 Process → Finish；业务 pack 循环及尾 drain 保持 |
| H idle/有效 AIC、AIV Init/work-unit/chunk | H raw `:93/:129`，InitDone `:54/:64/:76` | 每合法核分支结束 INIT 一次；idle AIC 的内层 return 仍返回共同 Finish |
| O Init/groupTask/head/owner 握手 | O raw `:109/:116`，旧服务 `:60`，A5 服务 `:59/:67` | 旧服务和 A5 AIC/AIV 分支都成对结束 INIT，内层返回仍到外层 Finish |
| arch22 Phase6 front/H/O | arch22 `:479/:497/:632/:637/:677` | Init/InitDone、两调度 Point、共同 Finish 与图匹配，包含条件 O 和原最终会合 |
| arch35 Phase6 front/H/O | arch35 `:332/:353/:446/:451/:479` | 同上，原流水和最终会合保持 |
| 串行 H/O SyncAll；重叠分支 | arch22 `:640/:649`，arch35 `:454/:463` | Begin/End 同分支包围原 SyncAll；重叠路径两端均不执行，不制造 HO_WAIT |

| 共享设备头实际起点 | 对应终点 | 配对/锚点 |
| --- | --- | --- |
| all TikNoBarrier `:170` | Finish Tok<Overwrite> `:208` | 总末尾仍只有原计时器的一次局部 PIPE_ALL `:84`；新增发布在它和末尾取时之后 |
| stage INIT TikNoBarrier `:171` | InitDone TokNoBarrier<Overwrite> `:178` | 每条合法分支一次，仍明确为初始化 issue window |
| stage BODY TikNoBarrier `:179` | Finish TokNoBarrier<Overwrite> `:209` | 共用 all.GetEndCycle，不重复 drain，不越出总窗口 |
| Point boundary TikNoBarrier `:186` | 紧邻 TokNoBarrier<Overwrite> `:187` | 两架构各两 Point，每次瞬时对完整；不是业务完成耗时 |
| HoWaitBegin TikNoBarrier `:194` | HoWaitEnd TokNoBarrier<Overwrite> `:201` | 两架构串行分支成对；末尾已有 SyncAll 是完成锚点 |

没有无收尾 Tik、无起点 Tok、漏标、多标或错位。固定槽均 Overwrite；没有实际 Accumulate、
动态类型或 setDynamicActualIter。枚举、parser 派生命名、方案 A 清单和图中的顺序逐项相同：
KERNEL_TIMING、MAIN_INIT、MAIN_BODY、PHASE6_FRONT_DONE、PHASE6_H_DISPATCH_DONE、PHASE6_HO_WAIT。

### 导出、诊断与原输入重放

`parse_timer_csv.py:84` 的 summarize_raw 只保存四 slice 的 word_count、非零 word/核行和 header，
没有放宽 parse_timer 的长度、metadata、缺核、配对、越窗、互斥及完整路由检查。
新 fixture 按用户统计重建 H=16/O=28、arch=310 的合成记录，得到非零 word `[0,0,297,513]`
和核行 `[0,0,48,84]`，仍抛 `captured=FWD_H,FWD_O; missing=PREPARE`。
这是 synthetic 复现合同，不是用户原 raw 文件的重新验证，更不能恢复缺失时间戳。

capture `:64`–`:84` 先保存 raw、hash 和 raw_summary；`:94` 的 exact 输出比较现在先于 `:102` 的解析。
解析失败写 decode_status=failed 和 decode_error 后原样抛错；输出不一致则保留 raw，
记录失败并停止，decode_status=pending 不冒称已解码。合成成功/缺 slice/输出不一致/不对齐四种采集路径均通过。
十项真实 NPU 输出一致性仍未验收。

UT README 的新重试命令先要求同步源码并全量重建安装 A5 wheel，然后以 `--inputs` 读取
原运行目录的 case_0.pt；加载路径继续使用保存的实际 tensor/kwargs，不重新构造模型输入，
新输出目录含 retry 时间戳，原 raw 与输入保留。`--check-output`、warmup=5、device=0
都是真实 CLI 参数；重试不执行 ATK 性能循环。真实 torch.save/load 与目标设备复跑尚未执行。
UT/ATK README 的当前新增段落、api/design/timer_flow 与上述实现及验证边界一致。

### 本轮独立执行证据与边界

| 命令/检查 | 实际结果 |
| --- | --- |
| `python -X utf8 -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p test_timer_contract.py -v` | exit 0；Ran 26 tests in 0.292s；OK |
| `python -X utf8 -m unittest discover -s ci/tests -p test_gdn_profile_performance.py -v` | exit 0；Ran 12 tests in 0.141s；OK |
| `python -X utf8 -m unittest discover -s ci/tests -p test_gdn_profile_input_export.py -v` | exit 0；Ran 5 tests in 0.045s；OK |
| `python -X utf8 -m unittest discover -s ci/tests -p test_gdn_forward_output_contract.py -v` | exit 0；Ran 2 tests in 0.029s；OK |
| `python -X utf8 tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py` | exit 1；仍为 `prepare kernel 缺少非 exp2 模式的局部指数换底选择`。相关算法头与 HEAD 相同，本轮未修复或隐藏 |
| Python compile()、`profile_gdn.py --help` | 三个修改后的 Python 文件语法通过；CLI help exit 0，含重放命令使用的参数 |
| 独立 git ls-files/git show 文本比较、rg 定位 | 165 个四算子 host/kernel 文件不变；全部实际观察节点和共享 Tik/Tok 重新定位匹配 |
| 相对链接与 Bash 语法 | UT/ATK README 与 api/design/timer_flow 中 8 个相对链接目标存在；7 个 UT Bash 块经 Git Bash --noprofile --norc -n -c 通过。首次沙箱 signal pipe 启动失败；随后只读语法检查经 auto-review 批准通过，未执行任何 Docker/安装/设备命令 |
| `git diff --check` | exit 0；无空白错误，仅 Git 的既有 LF/CRLF 提示 |

本机 PATH 和常见 LLVM/MSYS/Miniforge/Visual Studio 路径未找到可用 C++ compiler，
没有执行 C++ stub、CANN 编译或设备调用，不能把 Python 合同解释为任何编译结果。
45 项均为离线/模拟证据；用户旧 H/O 完整记录是修复前反馈。
修复后的真实 Prepare/H/O 完整采集、新 wheel/OPP 实际加载、GM 边界、正式精度/确定性/内存/性能、
时钟与打点扰动仍待生产环境复测；本轮源码审核通过不代表阶段 05 正式验收完成。

## 2026-10-12 独立重放 CLI 的 OPP 初始化顺序复核

本轮仍由未参与实现的 Step 9 独立子 agent 执行，只追加报告，不修改实现。
缓存发布修复已固定在提交 `3e06e104d03d9a0a24735c71f47ad369417e15b7`。
独立查看当前四文件 diff、真实包初始化及 pth 代码后，重新运行新增后的 28 项 timer 合同和 CLI help。
**结论：本轮源码复核通过，未发现新增阻断问题；硬件完整采集仍未复测。**
上轮 45 项证据和全部历史保持，以下是补充复核，不覆盖此前结果。

### 初始化缺口与修复范围

`torch_custom/fla_npu/fla_npu/__init__.py:101`–`:119` 明确说明 CANN 可能在 import torch_npu 时
初始化 host/tiling/kernel 注册表，随后 import fla_npu 再设 ASCEND_CUSTOM_OPP_PATH 可能已过晚。
同包 `fla_npu_opp_env.py` 通过 pth 提前设置路径，但允许禁用，且 vendor 不存在时会跳过。
原 CLI 在重放输入时先 import torch/torch_npu 并执行 inputs.to(NPU)，直到 capture 才 import fla_npu；
先前性能入口只修改自己的子进程环境，不能使之后在父 shell 启动的独立重放进程继承那些修改。
因此 CLI 需要在 backend 初始化前自行准备当前 Python 可发现包的 OPP 路径。

`profile_gdn.py:20` 的 prepare_capture_environment 使用顶层 find_spec("fla_npu") 查找 spec，
根据 spec.origin 定位包内 `opp/vendors/fla_npu_transformer`，只检查目录和修改环境变量。
`:26`–`:31` 将该 vendor 和 op_api/lib 置于现有路径前方，其它非空 OPP 路径保留原顺序。
`:24/:27` 在缺包、origin 或 vendor 时抛错；实际 main 的调用在 `:154`，早于 torch/torch_npu 导入、
输入加载与 NPU 转移。help 由 argparse 直接结束；parse-only 在调用准备函数前导出并返回。

该函数没有 import fla_npu、torch 或 torch_npu，没有 ctypes/CDLL、NPU 方法或 CANN 调用。
包源码、共享 timer、parser、capture 数值比较、raw 保留和完整性检查未因本轮初始化修复变化。
它只保证独立 CLI 的准备顺序；不宣称可以重新初始化已经启动的 CANN 注册表，或保证任何目标 wheel/OPP 已通过加载验收。

### 独立执行证据

| 命令/独立检查 | 实际结果 |
| --- | --- |
| `python -X utf8 -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p test_timer_contract.py -v` | exit 0；Ran 28 tests in 0.332s；OK。新增两项执行真实 main 的 mock，验证路径在 torch/torch_npu import 前就绪，以及缺 spec/origin/vendor 时无 backend import |
| `python -X utf8 fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/profile_gdn.py --help` | exit 0，参数保持，无 wheel/NPU 前置要求 |
| 独立调用 actual main 的 help/parse-only 分支 | 将 prepare_capture_environment 替换为一调用即失败的函数；help 正常 exit 0，parse-only 恰调用 export 一次，两者均未进入环境准备 |
| 真实 find_spec 的合成包实验 | 临时包的 __init__.py 设为一执行即抛错，vendor 目录存在；真实 prepare_capture_environment 成功，fla_npu 未进入 sys.modules，说明源文件未执行。重复 vendor/op_api 路径排到前两项，其它两个 OPP 路径顺序保留；临时目录已清理。这是 synthetic 文件系统证据，不是已安装 wheel/NPU 验收 |
| 两个修改后 Python 文件 compile() | profile_gdn/test_timer_contract 语法通过 |
| 与提交 3e06e104 的独立字节比较 | 两个共享 timer 头和原 timer_review 全文在追加本节前完全相同（统一 CRLF/LF）；原 45 项和旧 ABI 失败记录未改写 |
| 当前 git diff、UT/ATK README 对照 | 仅 CLI 环境准备/调用时机、两项测试、README 计数及初始化前提；没有新增 kernel、数学、flags、SyncAll、计时槽位或 CANN 调用 |
| `git diff --check` | exit 0，无空白错误，仅已有 LF/CRLF 提示 |

当前计数为 28 项 timer 加上上轮 12 项入口、5 项快照、2 项输出，合计 47 项离线/模拟合同。
后三组及对应入口/快照/输出实现本轮未修改，复用上轮实际通过证据，本轮没有重复运行这 19 项。
既有 `prepare kernel 缺少非 exp2 模式的局部指数换底选择` 未修复，上一轮失败结论继续有效。
没有执行真实 wheel 安装、torch.save/load、CANN 注册表初始化、编译、NPU 采集或正式 ATK 验收；
初始化顺序与缓存发布两个源码修复的目标设备效果均仍待复测。
