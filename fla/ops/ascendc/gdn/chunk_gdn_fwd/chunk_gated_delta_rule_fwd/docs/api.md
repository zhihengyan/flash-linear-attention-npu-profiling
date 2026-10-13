# GDN 计时诊断接口

本说明补充 main `90176b44` 的既有 ChunkGatedDeltaRuleFwd 接口。原数学语义、输入参数和十项返回值沿用原 README、ACLNN 头文件与 ATK 标杆；诊断接口是独立符号，不修改生产 C ABI 或 Stable schema。

`aclnnChunkGatedDeltaRuleFwdProfileGetWorkspaceSize` 接收原 GetWorkspaceSize 的全部参数，在 workspaceSize/executor 两个返回指针之前追加 `const aclTensor *timer`。`aclnnChunkGatedDeltaRuleFwdProfile(workspace, workspaceSize, executor, stream)` 沿用普通 launch。普通调用不分配 timer。

timer 必须是与 q 同设备的一维、连续、ND、INT64 张量，至少含 canonical 头 `common/kernel_utils/gdn_timer/AscendTimerV2.hpp` 的 TOTAL_BUFFER_SIZE 个元素。Python 采集端用该头推导大小，创建全零缓冲并验证 64B 对齐。每次调用单独创建缓冲，禁止多 stream 共享；调用者持有到 synchronize 后。直接 C 调用者负责同设备、对齐、清零及生命周期。host 不对 timer 调用 Contiguous，避免写入临时副本。

Python 私有 eager 入口为 `_aclnn_ctypes._profile_chunk_gated_delta_rule_fwd(q, k, v, g, beta, timer=timer, **原参数)`；它返回原十项 tuple，校验 dtype/rank/device/连续性，并标记 timer storage mutation。主要采集入口见 [timer_run.md](timer_run.md)，布局及测量含义见 [timer_flow.md](timer_flow.md)。诊断入口不注册为训练/图编译公开算子。

原始计数单位为 cycles。仅在明确核实 GetSystemCycle 频率时转换为 us；Perfetto 还需要确认 AIC/AIV 共同时钟。各 launch 时间原点独立。动态采样截断必须报告；没有对应窗口表示该核未执行或未覆盖，不等价于零计算耗时。
