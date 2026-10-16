# GDN 计时接入设计

当前独立实验为 [Stage5 A直接供L1](prepare_stage5_a_direct_l1_design.md)，分支 `codex/gdn-stage5-a-direct-l1`，直接基于K32 `d0cb628d`。保留GM A输出，使用已有Fixpipe路径给L1供数，普通/Profile共享实现；Stage3/N32/Block32修改均不继承。下文计时与K32设计保留为历史基线，设备状态见 [validation.md](validation.md)。

当前规则版本：V2。本设计仅描述诊断增量；基线为 main `90176b44321ebdbf3ae2a9057243157d7085c6e1`。按仓库阶段 03→04→05 路由读取现有实现、ChunkFwdH 设计、原 ATK generator/executor/reference。采用 CHUNK_DEPENDENT V1 开发参考（来源 commit a14961e9dfe8c8264768d67fb447101873da4ecf）的现有 H 状态依赖链，不改算子调度。host/kernel/调用层接入技术写法参考本地 timer_by_skill 的已存在诊断实现；本轮新增完整关键运算与循环观测、布局 v2 和独立源码审核，未沿用旧审核结论。

计算链保持：KKT=KKᵀ 的 gate/下三角系数 → solve → W/U；H 内 WH、vNew=U−WH、KV、状态更新；O 内 QK、QH、causal mask/gate、Attn×vNew、输出融合。原 fixed/varlen、head ratio、BT64/128、V128/256、initial/final state、B30 和 H/O idle/pipeline 分支均保持。原图、关键步骤与源码配对清单由 [timer_flow.md](timer_flow.md) 定义。

## 新增 GM 布局与所有权

设 F=6、D=28、I=32、Gcap=256、C=3Gcap=768。每核 timing 槽数 F+D×I，两个 int64/槽；每行 words 按 16 对齐。每核实际调用计数 D 个 int64，按 16 对齐；计数保存总数，事件仅采样前 I 次。一个 kernel slice = 16-word header + C×timing row + C×count row；四个 slice 对应 Phase6/Prepare/H/O。具体派生常量只维护在 canonical C++ 头，parser 读取其枚举与 constexpr 算术，host 按 TOTAL_BUFFER_SIZE 校验。

地址半开区间（单位 int64 words）：slice s 基址 s×BUFFER_WORDS_PER_KERNEL；header=[base,base+16)；核心 c timing 行=[base+16+c×stride,base+16+(c+1)×stride)；counts 基址=base+16+N_TIMING_COUNTER，c 行=[counts+c×countStride,counts+(c+1)×countStride)。AIC group g 使用 c=3g，AIV0/1 使用 c=3g+1/2。各行和 header 独占完整 cache line；没有其他写者。header 由 AIC0 写实际 blockDim、arch、layout/version/开关/容量。超容量不会转写 core0，解析时拒绝。

新增 timer 张量为独立 GM 区域，生命周期从采集端 torch.zeros 到 launch 同步、导出之后；不占原 workspace，不分配 UB/L1/L0，不改变原 task/output 地址。每 scope 的局部 start/iter 独立，嵌套与重复不会覆盖；每类的 per-core actual counter 在开始处更新。诊断末尾本核局部 PIPE_ALL 只执行一次，结束戳写完后 DataCacheCleanAndInvalid 发布本核 timing/count 行，适配 Prepare 关闭 kernel-end 自动 DCache 写回的已有选项。该发布和新增标量打点会扰动诊断调用，不能宣称零开销。

## 同步与 R01–R21 增量审查

所有原同步资源、ready/free、数据驻留份数、CG、task/work/head/chunk 映射、tail/padding 和 workspace 生命周期保持。新增诊断行只有本核读写，header 唯一写者；不增加跨核依赖、event/flag/SyncAll 或 UB/L1 资源。因此 R01–R21 中业务 Stage、空间和调度条目沿用基线，仅补充上述独立 GM 所有权与容量。源码校验按顺序比较原 WaitFlag/SetFlag/CrossCore*/SyncAll/PipeBarrier 调用；编译、实际模板选择、同步、精度和内存验收仍需设备证据。

所有细分 scope 使用 NoBarrier，等待项以原 wait 返回为锚点；matmul/VF/epilogue 调用窗含原内部同步，不能称纯计算完成时延。固定总窗包含末尾诊断 drain；GM 发布在结束戳之后。ENABLE_TIMER=0 移除实际 scope/Init/Finish 操作；正常 null timer 不读时钟、不写 GM、不 drain。开关 A/B 的真实打点开销留待设备测量。

上述计时接入的验证入口及状态统一见 [timer_run.md](timer_run.md)、[timer_report.md](timer_report.md)。计时接入本身没有性能优化目标，也不修改原性能 case 的 shape 或比较标准。

## 后续性能候选：Prepare Stage4/5 K32

从 `codex/gdn-profiling` 出发的独立实验在三个 FP32 求解乘法中只归约 leaf 的有效32个 K 元素。候选详设、物理布局与同步边界见 [prepare_k32_design.md](prepare_k32_design.md)，开发期证据和设备待验项见 [validation.md](validation.md)。该候选保持现有计时 ABI 和布局，尚未获得设备精度/性能验收，也未替代上述诊断接入基线。
