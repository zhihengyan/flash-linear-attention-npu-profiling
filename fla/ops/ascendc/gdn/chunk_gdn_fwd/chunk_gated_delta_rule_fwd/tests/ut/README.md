# GDN profiling：A5 / Ascend950PR / Docker

本文按当前 **Ascend950PR、已有 Docker 容器、容器内已打包 CANN 和 Python/ATK 依赖**的环境组织。
进入容器、编包安装后，运行一个入口脚本即可完成工程性能用例测试和逐核打点 CSV 输出。

`A2`、`A5` 是仓库使用的 Ascend 产品档位：

| 档位 | 本仓编包参数 |
| --- | --- |
| A2 / Ascend 910B | `FLA_NPU_SOC=ascend910b` |
| A5 / 当前 Ascend950PR 环境 | `FLA_NPU_SOC=ascend950` |

此前 A2 示例依据先前给出的 ascend910b 编包命令；当前环境应改为 **ascend950**。
下列操作在目标环境执行，本机未接入 Docker/CANN/NPU，不表示设备实测已经通过。

## 1. 进入已有容器：在宿主机执行

容器应已启动、已透传 NPU，且已挂载或复制本次源码。先查看名称，替换下面两个变量。
`REPO_IN_CONTAINER` 是容器内仓库根目录；示例路径请改成实际路径。

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
CONTAINER=你的已有容器名
REPO_IN_CONTAINER=/workspace/flash-linear-attention-npu-profiling
docker exec -it -w "$REPO_IN_CONTAINER" "$CONTAINER" bash
```

`docker exec` 是进入已运行容器，不会重新安装依赖。如果容器已停止，先 `docker start "$CONTAINER"`。
后续步骤全部在刚打开的容器 shell 执行。

## 2. 容器内编包并安装 A5 wheel

保持编包、安装、ATK 和采集使用同一个 Python 环境。若镜像依赖在专用 venv/conda 环境，
先按镜像约定激活它；CANN 自定义安装位置通过 `CANN_ENV` 指定，或修改 source 路径。

```bash
# 当前目录应为容器内仓库根目录，源码须包含本次更新。
source "${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
npu-smi info

# A5 / Ascend950PR：全量构建 Python + host + kernel + OPP。
FLA_NPU_SOC=ascend950 FLA_NPU_OPS= python scripts/build_wheel.py

# 用本轮构建输出的准确文件名替换占位符，或直接复制构建末尾打印的安装命令。
python -m pip install --force-reinstall --no-cache-dir --no-deps 'dist/<本轮准确wheel文件名>.whl'
```

如果此前按 A2 示例安装过 `flash-linear-attention-npu-a2`，先卸载该 A2 发行包，再安装本轮 A5 wheel：
`python -m pip uninstall -y flash-linear-attention-npu-a2`。两个档位共用 fla_npu 文件，避免同时保留。
构建细节见[开发者指南](../../../../../../../../docs/开发者指南.md)。镜像已含依赖，本流程不重复安装它们。

## 3. 安装后只运行这一个脚本

```bash
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py --device 0
```

该[入口脚本](../../../../../../../../tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py)自动执行：

1. 检查 ATK、NPU 环境、新安装 host/kernel/Python 产物及两个 Profile 符号。
2. 注册当前 wheel 的 OPP 路径，固定当前 Python 环境，以 `-soc=ascend950 -scope=performance` 调用工程 ATK。
3. 执行已有性能 JSON 的 case 0/1，在 DUT 初始化时保存真实输入，保持原性能调用循环。
4. 每条 case 用独立 Python 进程：预热 5 次，执行正式对照和一次诊断调用，导出 raw、逐核 CSV、metadata 和日志。

不需要手工设置 PYTHONPATH、GDN_PROFILE_INPUT_DIR 或 ATK_OUTPUT_ROOT。
脚本使用已激活的容器环境；若设置了 ATK_ENV，它必须与当前 Python 环境一致，脚本会检查这一点。
ATK 的 console script 也必须来自当前 Python 的 bin 目录且使用该解释器；脚本检查其启动路径并导入 executor 所需 ATK 模块。
子进程会清除旧 FLA_NPU_OPP_PATH 覆盖变量，使产物校验与实际调用都使用当前安装 wheel 的 OPP。
任何检查、ATK 测试、输入导出或采集失败都会返回非零并保留日志；成功时打印 `Profiling complete: <结果目录>`。

| case id | 模型 | B / Hk / Hv / T / K / V | dtype / chunk | 状态参数 |
| --- | --- | --- | --- | --- |
| 0 | 推理 | 1 / 16 / 32 / 11274 / 128 / 128 | BF16 / 64 | `cu_seqlens=[0,11274]`，确定性初态，输出 final_state |
| 1 | 训练 | 2 / 32 / 32 / 8192 / 128 / 128 | BF16 / 64 | 定长，无初态，无 final_state |

使用 `tests/atk/chunk_gated_delta_rule_fwd/atk_chunk_gated_delta_rule_fwd_perf.json`，不重新生成用例。
ATK 只跑 DUT，诊断阶段沿用保存的值、dtype、scale、chunk_size、初态、final_state、cu_seqlens 和 chunk_indices。
十项输出使用 `--check-output` 做正式/诊断 exact 对比；它不是 CPU/六算子双标杆精度验收。

可选参数：

```bash
# 只采推理 case；训练 case 用 --cases 1。
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py --device 0 --cases 0

# 自定义结果目录；目录须尚不存在，每轮保存首份输入并使用新 timer。
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py \
  --device 0 --output-dir /workspace/gdn_profile_run_01 --warmup 5
```

默认会清除旧的 case 范围覆盖，执行两条 case；`--cases` 可选 0、1，设备号由 `--device` 指定。

## 4. 结果在哪里，如何阅读

默认结果目录在仓库内已被 Git 忽略的位置：

```text
tests/atk/chunk_gated_delta_rule_fwd/atk_output/gdn_timer_<运行标识>/
  run_metadata.json           # 状态、ascend950、Python、case/输入哈希、源提交及 OPP 路径
  performance_cases.json     # 本轮固化的性能 JSON
  *.log                      # 环境/安装/符号检查和 ATK 日志
  inputs/case_0.pt, case_1.pt
  atk/perf/                  # ATK 正式性能报告与系统 profile
  timer/case_0/
    gdn_npu_time_raw.csv      # 完整 INT64 原始记录
    gdn_npu_time_parsed.csv   # 每 kernel/每核/每阶段的 start/end/duration_cycles
    timer_layout.json        # 布局、实际核数、架构和执行路径
    capture_metadata.json    # 输入、实际库、采样选项和十项输出对比结果
    profiling.log            # 阶段 min/median/max 统计
  timer/case_1/              # 同样的文件；只采单 case 时只生成对应目录
```

先看 `run_metadata.json` 的 status，再看各 case 的 metadata 和 profiling.log。
capture metadata 的 raw_summary 记录四个区段的非零 word/核行数；decode_status/decode_error 记录解析结果。
十项输出对比在解析之前完成，即使某段记录缺失，仍会保存对比结果和 raw。
同一 kernel、同一核类型内比较 KERNEL_TIMING / MAIN_BODY，观察负载差异；实际路由以 metadata 为准。
默认单位为 **cycles**，`Perfetto skipped` 是未确认时钟时的预期提示。

| 计时项 | 语义 |
| --- | --- |
| KERNEL_TIMING | 绑定 timer 后开始，kernel 末尾局部 PIPE_ALL drain 后结束，包含打点扰动 |
| MAIN_INIT / MAIN_BODY | 初始化下发窗口 / 主体窗口；INIT 可能包含异步初始化 |
| PHASE6_FRONT_DONE / PHASE6_H_DISPATCH_DONE | 瞬时下发边界，不作为业务 pipeline 完成耗时 |
| PHASE6_HO_WAIT | 既有串行 SyncAll 窗口；重叠路径不记录该事件 |

每个 kernel 单独取时间 origin，不能用 trace 推算不同 launch 的间隔或 host 总耗时。
正式 Task Duration 仍读取 ATK 系统 profile 的 op_summary。把整个结果目录带回即可分析。

## 5. 可选：输出 Perfetto

先核实 GetSystemCycle 的 cycles/us 及 AIC/AIV 共同时钟，再设置 CYCLES_PER_US 为已验证的正数。
可以在一次入口调用中同时生成 CSV 和 trace：

```bash
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py \
  --device 0 --cycles-per-us "${CYCLES_PER_US:?先设置已核实的cycles/us}" --shared-clock
```

也可以对第 3 步已有 raw 运行第二个脚本，离线转换，不重新执行 ATK 或 kernel：

```bash
RUN_DIR=/请替换为入口脚本打印的结果目录
UT=fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut
for case_id in 0 1; do
  python "$UT/profile_gdn.py" \
    --parse-only "$RUN_DIR/timer/case_${case_id}/gdn_npu_time_raw.csv" \
    --output-dir "$RUN_DIR/timer/case_${case_id}" \
    --cycles-per-us "${CYCLES_PER_US:?先设置已核实的cycles/us}" --shared-clock
done
```

只采单 case 时相应修改循环。新增 `gdn_npu_time.json`，CSV 新增 duration_us 列；在 https://ui.perfetto.dev/ 加载。
`--shared-clock` 是调用方对平台事实的确认，脚本不猜测频率或自动校准时钟。

## 排查与验证边界

### Prepare 全零 / incomplete Prepare/H/O capture

2026-10-12 用户在 A5 case 0 反馈：raw 长度 49,216，Prepare 全零，H/O 分别完整记录 16/28 个核组。
源码中的 Prepare 编译选项关闭自动 kernel-end DCCI，旧 timer 实现又缺少显式缓存写回。
本轮修复在计时窗口外定向写回 metadata 和每核 timer 行，保留完整性校验；旧 raw 缺失的时间戳无法恢复。

这包含 kernel 修改，必须同步源码后**重新编包安装**，仅更新 Python 不足以修复。
按第 2 节重建安装 A5 wheel 后，可以复用已保存的 case_0.pt，只重跑诊断，无需再执行 ATK 性能用例：
该独立 CLI 会在导入 torch_npu 前注册当前 Python 已安装 wheel 的 OPP，避免旧注册表影响重试。

```bash
# 仓库根目录；RUN_DIR 改为之前入口打印的那一轮目录。
RUN_DIR="$PWD/tests/atk/chunk_gated_delta_rule_fwd/atk_output/你的原运行目录名"
UT=fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut
python "$UT/profile_gdn.py" --inputs "$RUN_DIR/inputs/case_0.pt" \
  --device 0 --warmup 5 --check-output \
  --output-dir "$RUN_DIR/timer/case_0_retry_$(date +%Y%m%d_%H%M%S)"
```

成功应生成三个 kernel 的逐核 CSV；metadata 的 decode_status 为 complete，十项输出 exactly match。
若仍缺 Prepare，保留新的 raw/metadata/log，核对新 wheel/OPP 的实际加载与重建结果，不删除该完整性检查。
本机无 CANN/NPU，修复后的设备结果待复测；上述反馈只证明旧版本的 H/O 记录存在。

| 现象 | 处理 |
| --- | --- |
| 找不到 atk / NPU 或环境检查失败 | 检查容器 NPU 透传、CANN source 和当前 Python/ATK 环境 |
| ATK_ENV 与 Python 不一致 | 先激活正确环境，再在同一环境编包、安装和采集 |
| ATK launcher 不属于当前 Python | 在当前环境安装 ATK；不使用 PATH 中其他环境的 atk 或无法确认解释器的 shell launcher |
| 产物 md5 不一致 / 缺 Profile 符号 | 安装本轮 A5 wheel，核对实际加载路径，避免旧产物混装 |
| ATK 未导出 case_*.pt | 确认环境上的 executor 已更新，查看 atk_performance.log 和 worker 日志 |
| raw 无事件、缺核或坏布局 | 保留 raw/metadata/日志，核查新 kernel、ENABLE_TIMER 和实际路由 |
| 十项 exact 对比失败 | raw 会保留；按输出有效区定位差异，不直接放宽阈值形成通过结论 |

一键入口的环境和调度通过 12 项模拟合同，既有输入导出 5 项、输出合同 2 项、timer 合同 28 项通过。
这些不代表 Docker、真实 torch.save/load、CANN 编译、两条 A5 case 实测、精度、内存、时钟或扰动验收通过。
既有 ABI 合同失败 `prepare kernel 缺少非 exp2 模式的局部指数换底选择` 继续保留，未在本轮修复。
源码审核证据见[独立审核](../../docs/timer_review.md)，阶段和枚举见[打点流程](../../docs/timer_flow.md)，
接口见[诊断契约](../../docs/api.md)，工程测试规范见[ATK README](../../../../../../../../tests/atk/chunk_gated_delta_rule_fwd/README.md)。
