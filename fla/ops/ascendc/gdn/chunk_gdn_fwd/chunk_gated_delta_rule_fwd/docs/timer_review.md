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
