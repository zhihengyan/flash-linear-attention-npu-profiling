# GDN 计时接入设计

本轮诊断布局 v3，基于 codex/gdn-profiling 的809bce6887639e831caa856b8ac3a197a6a94294。原计时接入来自 main 90176b44321ebdbf3ae2a9057243157d7085c6e1；历史设计与验证见该基点。本轮按仓库阶段03→04→05进行内部诊断增量，采用原 CHUNK_DEPENDENT V1 开发参考（a14961e9dfe8c8264768d67fb447101873da4ecf），沿用 H 状态依赖链和原调度。用户已授权新建分支与针对性打点；不采用 Stage6/H 双head优化。

业务仍为 KKT/gate→solve→W/U→H→O。公开 ABI、数学、dtype/layout、tile、key、task重排、workspace、显式UB/L1/L0地址、已有event/flag、tail/varlen/head-ratio与输入/输出契约均保持。当前接口、原CPU标杆和ATK标准沿用基点；本轮只增加 timer GM 独立观测。具体 Stage3–5 原流程、完整 scope 映射和候选判据见 [诊断设计](timer_prepare_stage35_design.md)、[逐点清单](timer_flow.md)。

## GM布局与所有权

F=6，D=60，I=32，group容量256，每组AIC/AIV0/AIV1三行（C=768），实际blockDim由header记录。每核时间戳行=align16(2×(F+D×I))；actual与sampled各为align16(D)的int64行；新32类tag行=align16(32×I)。四个launch slice均使用相同stride；所有行独占64B cache line。

slice s 基址=s×BUFFER_WORDS_PER_KERNEL，单位int64 words：

- header=[base,base+16)
- timing基址=base+16，长度N_TIMING_COUNTER
- actual基址=timing+N_TIMING_COUNTER，长度DYNAMIC_ITER_TOTAL_SIZE
- sampled基址=actual+DYNAMIC_ITER_TOTAL_SIZE，长度DYNAMIC_ITER_TOTAL_SIZE
- tags基址=sampled+DYNAMIC_ITER_TOTAL_SIZE，长度DETAIL_TAG_TOTAL_SIZE

每区核心c的行按对应stride偏移，c=3g/3g+1/3g+2分别属于AIC/AIV0/AIV1。写者只写本核行，超容量不会转写core0。header的12..14由采集端在launch前初始化并同步，值在launch中保持等价不变；所有核先读原控制，AIC0唯一发布完整header与实际默认pack_count=8，15记录tag stride。不引入跨核同步。

每次timer为15384640个int64（117.38MiB），生命周期从torch.zeros到同步/导出结束，不占业务workspace。host原Profile契约通过canonical TOTAL_BUFFER_SIZE校验；采集器从该头推导尺寸，ctypes私有timer入口沿用。必须完整重建四算子和host的v3 wheel；旧v2 raw留在对应旧分支解析。

## 采样、配对与发布

Prepare选择每核pack ordinal区间，最多8pack，每类最多32槽。H/O/Phase6仍前32调用。每类本地累计全launch actual和独立sampled；每个scope独立保存start、slot、work ID，重复/嵌套不会覆盖。只有选中槽读clock/写GM；新详细类只在Prepare的detail开关开启时计数。Stage0在pack循环前仍保留一次；循环后既有drain随最后pack选择，粗窗不提供任务身份。

work ID来自原PackWorkId；原G=2满pack交换task1/2，已有编排不动。tag=work_id+1，0未写，work_id=0有效。PREPARE_PACK使用首taskID；原28粗类不带ID（导出-1）。密集iteration不是完整调用ordinal，不用于跨stage配对。解析器验证actual/sample、越界/未写/重复tag及每核固定窗；分析器验证完整选中task/pack覆盖、原分支与本核操作顺序。

全部动态scope NoBarrier；wait以原同步返回为锚点，math/copy仍为含原内部等待的下发窗口，不称纯计算完成时延。Finish诊断末尾本核一次PIPE_ALL，复用end给MAIN_BODY，结束戳之后setDynamicActualIter/setDynamicSampledIter并DCCI发布本核所有行。发布开销不在固定窗口内，诊断call真实用时包含它。

NULL timer不计算pack映射、不读clock、不写GM、不drain；ENABLE_TIMER=0的三种scope/context宏为空。计时器局部数组及分支会改变编译资源或寄存器压力，不能声称普通路径零开销。

## 增量审查与完成条件

R01–R21业务同步、资源、生命周期和边界条目沿用基点，只新增上述独立GM区域、控制与采样。源码检查对Prepare比较全部原WaitFlag/SetFlag/CrossCore等顺序，并移除timer宏/新增词法块后比较全部业务token；公开C声明和生产Python签名保持。独立fresh-context审核按op-timer-ascendc Step9逐项审查，状态见 [本轮记录](timer_prepare_stage35_report.md)。

主入口先普通msprof，再同输入粗/细成对诊断；新旧wheel普通调用和粗/细窗口是两种不同扰动检查。诊断只定位，≥5%验收仍用重复普通七项kernel时间和（基线1708.3914us，至少节省85.41957us）。本机未具备CANN/NPU，阶段05设备构建/数值/内存/性能与实际模板覆盖待内部执行。命令和小数据回传见 [运行说明](timer_run.md)。
