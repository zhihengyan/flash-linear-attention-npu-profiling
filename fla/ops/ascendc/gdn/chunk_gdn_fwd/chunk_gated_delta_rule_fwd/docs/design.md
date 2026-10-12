# GDN 方案 A 计时设计

设计规则版本 V2。本次只设计诊断观察差异，接口以 `api.md` 为准，
原始计算图、原有同步和用户选择见 `timer_flow.md`，采集与设备验收说明见 `../tests/ut/README.md`。

## 存储与所有权

单次前向分配独立 GM INT64 timer；每个 kernel 有独立 slice：
PHASE6=0、PREPARE=1、FWD_H=2、FWD_O=3。
三 kernel 路径分别写后三个 slice，融合路径只写第一个；不改变计算 workspace。
每 slice 有 16 个 metadata word 和 256×3 个核心行；每行六对 start/end，向 16 word 对齐。
每 slice 12,304 word，共 49,216 word。每行 128 字节，边界不跨其它核心行。

| 资源 | 大小/数量 | 分配与复用 |
| --- | --- | --- |
| timer GM | 393,728 字节/调用 | 调用方清零并持有，完成后导出；并发调用不共用 |
| 每核标量计时对象 | 三个独立 start/end 状态 | all、stage、boundary 分开；不占用 L1/UB/L0 队列 |
| metadata | 128 字节/kernel slice | AIC 0 唯一写者；其他核不读，不需新跨核同步 |
| 每核事件行 | 128 字节 | AIC 行 group×3，AIV 行 group×3+subBlock+1；越界不回绕 |

`CORE_GROUP_CAPACITY=256` 是存储容量，不是设备物理核数。
metadata 含 magic、version、kernel ID、实际 AIC blockDim、容量、stride、事件数、ENABLE_TIMER 和编译 arch。
未启动 slice、未执行等待和 padding 保持零；解析器拒绝不完整路由、缺核、版本不匹配及超限。

## 观察阶段与锚点

| 槽 | 核心对象/起点 | 终点 | 解释 |
| --- | --- | --- | --- |
| KERNEL_TIMING | all：绑定 timer 后 TikNoBarrier | 整个实际 kernel 返回前 Tok，局部 PIPE_ALL drain | 逐核完整业务窗口，含打点扰动，排除 timer 绑定及 host 辅助任务 |
| MAIN_INIT | stage：all 起点之后 TikNoBarrier | 实际服务 Init 后 TokNoBarrier | 初始化 issue 窗口，异步初始化可能尚未完成 |
| MAIN_BODY | stage：InitDone 后 TikNoBarrier | 与 KERNEL_TIMING 共享末尾已排空 timestamp | 主体窗口，不再次 drain，不越出总窗口 |
| PHASE6_FRONT_DONE | boundary：进入私有 DispatchFwdH 前 | 紧邻 Tik/Tok 瞬时对 | 前端调度边界，非所有 pipe 完成 |
| PHASE6_H_DISPATCH_DONE | boundary：DispatchFwdH 返回后 | 紧邻 Tik/Tok 瞬时对 | H 调度返回，非全局 H 完成 |
| PHASE6_HO_WAIT | boundary：既有串行 SyncAll 前 | 同一既有 SyncAll 后 TokNoBarrier | 原同步调用窗口；只覆盖该同步规定的 pipeline |

H 无 Cube 工作的 AIC 分支先 InitDone 再返回到外层 Finish；不遗漏 idle core。
O 的 A5 AIC/AIV 分支和旧 kernel 分支均在真实服务 Init 后 InitDone。
Phase6 初始化在首个 KKT 工作前结束。重叠路由无 H/O 全核等待，保留原 ready 协议和最终会合，
PHASE6_HO_WAIT 为空；本轮不进 pack/head/chunk/task 计算循环。
INIT 没有完成锚点，刻意按 issue 窗口命名和解析，不用额外 barrier 把初始化异步工作串行化。

起点均 NoBarrier；三对象避免总计时和阶段计时互相覆盖。
最后 only timer!=nullptr 时执行一次局部 PIPE_ALL；null timer 不执行 GM 写入、cycle 读取或新增 barrier。
ENABLE_TIMER 使用可覆写默认值与 `#if ENABLE_TIMER`；编译为 0 时 GdnKernelTimer 没有计时成员。
不添加 SyncAll、CrossCore flag、event 或 ready/free，不修改原有 workspace 生命周期和流水重叠。

## 接入与打包

四个 OpDef/L0/raw kernel 的 timer 均追加为最后输入；正式 L2 传 null，诊断 L2 传完整 buffer。
slice 地址由设备层按 kernel ID 计算，host 不创建子视图、不更改 data offset。
共享实现位于 `fla/ops/ascendc/common/kernel_utils/gdn_timer/`，使用现有 common copy/install 路径，
四个 kernel 包含 `kernel_utils/gdn_timer/AscendTimerV2_device.hpp`。
目标 op_kernel 的两个转发头兼容源码树和 OPP 扁平目录；host 包含目标转发头。
parser 读取共享 canonical C++ 头，以受限 AST 计算整数 constexpr 和枚举，避免双份布局常量。

## 时钟与输出解释

不引入新全核入口同步，也不预设 GetSystemCycle 的跨核/跨启动基准。
默认导出整数 cycle；INIT 是 issue、边界是 point，汇总不把 point 对的微小 duration 当成业务时间。
提供经目标平台确认的 cycles/us 后可导出 duration_us；同时确认 AIC/AIV 共同时钟后才生成 Perfetto。
trace 每个 kernel 单独取 origin，整数先减 origin 再转浮点；不能用于解释 kernel 间 launch 间隔或 host 总耗时。
计时结果用于阶段定位；正式性能标准继续使用 msprof op_summary 的 Task Duration。
需要在生产环境做原调用/诊断调用/ENABLE_TIMER=0 对照，确认精度及局部 drain/GM 写入的扰动。
