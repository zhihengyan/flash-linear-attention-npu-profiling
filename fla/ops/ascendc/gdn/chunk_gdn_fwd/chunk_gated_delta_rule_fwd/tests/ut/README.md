# GDN 方案 A profiling

本工具对完整 GDN 前向的实际 kernel 主流程计时，覆盖 Prepare→H→O 和 arch22/arch35 Phase6。
固定槽为 KERNEL_TIMING、MAIN_INIT、MAIN_BODY、PHASE6_FRONT_DONE、
PHASE6_H_DISPATCH_DONE、PHASE6_HO_WAIT；本轮没有循环内动态事件。
流程图与源码映射见 `../../docs/timer_flow.md`，诊断契约见 `../../docs/api.md`。

## 生产环境使用

本次只完成源码接入和离线验证；下列设备命令尚未执行。
按 `docs/开发者指南.md` 在目标 Linux/Ascend 上完整构建并安装含本次源码的 wheel/OPP。
正式 API 签名和 Stable schema 保留，新的独立 ctypes 诊断入口要求新 host、OPP 和 Python 适配层一起安装。
确认实际加载的库路径及目标 SoC/CANN 版本，避免新旧产物混装。

以下路径从仓库根目录执行。示例 shape 只用于冒烟，不代表用户的模型性能 case。

```bash
UT=fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut
# A5 BF16 K=V=128、chunk=64、head ratio<=4 可能进入三个 kernel 路径。
python "$UT/profile_gdn.py" --device 0 --tokens 65 --warmup 5 --check-output --output-dir gdn_timer
# 默认特性下 V=256 进入 Phase6；仅在该组合受目标平台支持时执行。
python "$UT/profile_gdn.py" --device 0 --tokens 129 --value-dim 256 --output-dir gdn_timer_phase6
```

真实模型输入可以保存为 `torch.save({"q":q.cpu(),"k":k.cpu(),"v":v.cpu(),"g":g.cpu(),"beta":beta.cpu(),
"kwargs":{...}}, path)`，kwargs 使用原 wrapper 的选项，包括 initial_state、cu_seqlens、chunk_indices。

```bash
python "$UT/profile_gdn.py" --inputs model_inputs.pt --device 0 --check-output --output-dir model_timer
```

在已有 NPU Python 程序中可调用 `profile_gdn.capture(q,k,v,g,beta,output_dir=...,**原有选项)`；
它预热正式入口，独立 timer 只记录一次诊断调用，完成后同步并导出。
`--check-output` 比较正式/诊断调用的十项输出，要求逐元素完全一致；失败保留 raw，不能用放宽阈值隐藏差异。
该比较不是 CPU/六算子 ATK 精度验收。生产 ATK、确定性、内存测试沿用原 README。
诊断 helper 将 timer 的原位修改登记为 mutation，成功 launch 后更新其 version counter，失败时不更新；拒绝 timer 求梯度。
不要并发复用一个 timer，不要将本 eager 诊断入口放入 graph capture。

## 文件与时钟

| 文件 | 内容 |
| --- | --- |
| gdn_npu_time_raw.csv | 整个 timer 的 INT64 原始记录 |
| gdn_npu_time_parsed.csv | 每 kernel、每 AIC/AIV、每阶段 start/end/duration_cycles |
| timer_layout.json | 从 canonical C++ 头派生的布局、实际 blockDim/arch、计时语义 |
| capture_metadata.json | 输入/选项、库路径、软件版本、预热/采样次数及 raw/header 哈希 |
| gdn_npu_time.json | 显式确认频率及共同时钟后的 Perfetto trace |

默认只保留 cycle，不假定周期频率或跨核时间基准。
确认目标 GetSystemCycle 的 cycles/us 后加 `--cycles-per-us <已验证数值>`；
还须确认 AIC/AIV 使用共同时钟，再加 `--shared-clock` 才生成 Perfetto JSON，
用 https://ui.perfetto.dev/ 加载。也可离线转换已有 raw，无需 torch/CANN：

```bash
python "$UT/profile_gdn.py" --parse-only model_timer/gdn_npu_time_raw.csv --output-dir model_timer_parsed
# 用已验证的值替换 CYCLES_PER_US；--shared-clock 是调用方对平台事实的确认。
python "$UT/profile_gdn.py" --parse-only model_timer/gdn_npu_time_raw.csv \
  --output-dir model_timer_parsed --cycles-per-us "$CYCLES_PER_US" --shared-clock
```

KERNEL_TIMING 在 timer 绑定后起、末尾局部 PIPE_ALL drain 后止；会包含打点扰动。
MAIN_INIT 是初始化下发窗口；两个 Phase6 boundary 是瞬时点，不是 pipeline 完成时刻。
PHASE6_HO_WAIT 只包围既有串行 SyncAll，重叠路径不制造该事件。
总窗口与 BODY 共用末尾 timestamp，仅排空一次；null timer/ENABLE_TIMER=0 无新增计时 barrier。
trace 的每个 kernel 单独取 origin，不用于解释 kernel 间 launch 间隔或 host 总耗时。
打点数据不替代 msprof op_summary 的正式 Task Duration 指标。

## 离线验证与待设备验收

```bash
python -m unittest discover -s "$UT" -p test_timer_contract.py
python -m unittest discover -s ci/tests -p test_gdn_forward_output_contract.py
python tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py
```

本机计时套件 20 项、前向消费合同 2 项通过；原有 mutation 合同 7 项也通过。计时套件覆盖布局、两路由、核映射、坏记录拒绝、配对/时间窗/阶段顺序、
超限、缺核、不完整路由、混合等待分支、整数 trace 精度、诊断 ABI、旧十项输出、打包头路径，以及成功启动后 timer 版本更新和失败/grad 拒绝分支。
导出 fixture 是 synthetic，测试结束清理，不是设备结果。
原 ACLNN 合同脚本在本次修改前已失败：`prepare kernel 缺少非 exp2 模式的局部指数换底选择`，
本次只调整其 optional timer 调用正则，仍报告同一失败。
原 ctypes 完整套件在当前无 torch 环境有一项报错、一项设备用例跳过；三项 GDN 回归（训练/推理输出、norm/别名、预留 gate 参数拒绝）用 mock 单独运行通过，其中前两项也由新套件复用。

后续设备需验证 CANN 编译/实际加载、timer 错误输入、ENABLE_TIMER=0、逐核映射与 GM 边界；
覆盖两条前向路径、串行/重叠、短序列 idle AIC、tail/GVA、fixed/varlen、初态/末态，
完成 CPU/六算子双标杆、确定性和内存检测，并测量打点扰动、确认时钟后保留采集原始证据。
独立 Step 9 的源码审核见 `../../docs/timer_review.md`，不能替代这些设备检查。
