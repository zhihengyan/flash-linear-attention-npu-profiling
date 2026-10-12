# GDN 方案 A profiling

本工具对完整 GDN 前向的实际 kernel 主流程计时，覆盖 Prepare→H→O 和 arch22/arch35 Phase6。
固定槽为 KERNEL_TIMING、MAIN_INIT、MAIN_BODY、PHASE6_FRONT_DONE、
PHASE6_H_DISPATCH_DONE、PHASE6_HO_WAIT；本轮没有循环内动态事件。
流程图与源码映射见 `../../docs/timer_flow.md`，诊断契约见 `../../docs/api.md`。

## 安装 wheel 后：使用工程性能用例采集（A2）

本节对应已经完成的以下两步：

```bash
FLA_NPU_SOC=ascend910b python scripts/build_wheel.py
python -m pip install --force-reinstall --no-cache-dir --no-deps dist/<本轮准确wheel文件名>.whl
```

后续所有命令在目标 **Linux/Bash、仓库根目录、安装该 wheel 的同一个 Python/ATK 环境**中执行。
本次新增的 ATK executor 是仓库脚本，环境上的源码目录也必须同步这次修改；仅更新 wheel 不会更新它。
通用部署见[开发者指南](../../../../../../../../docs/开发者指南.md)，ATK 前置要求见
[ATK 说明](../../../../../../../../tests/atk/README.md)。本机无 CANN/NPU/ATK，下面设备命令尚未实测。

执行顺序：环境和加载检查 → ATK 性能测试并保存实际输入 → 同输入的独立诊断调用 → 逐核 CSV → 可选 Perfetto。
ATK 正式性能调用仍走原入口；Tik/Tok 采集在后续新 Python 进程中执行。

### 1. 准备变量、CANN 与 ATK 环境

如果需要激活 ATK 虚拟环境，应在该环境中安装 wheel；不要在安装后切换到没有该 wheel 的另一环境。
自定义 CANN 安装位置时替换下面 `source` 路径。当前示例用设备 0，修改 `NPU_DEVICE` 即可换卡。

```bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export FLA_NPU_SOC=ascend910b

# 当前工作目录必须是本次源码仓库根目录。
REPO_ROOT="$PWD"
OP_DIR="$REPO_ROOT/tests/atk/chunk_gated_delta_rule_fwd"
UT="$REPO_ROOT/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut"
NPU_DEVICE=0

# 每次运行使用新目录；该位置已被 Git 忽略，不提交采集数据。
RUN_DIR="$OP_DIR/atk_output/gdn_timer_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_DIR"
export GDN_PROFILE_INPUT_DIR="$RUN_DIR/inputs"
export ATK_OUTPUT_ROOT="$RUN_DIR/atk"

# 供 ATK worker 导入本算子的 gdn_reference / atk_role_contract 等模块。
# 不添加 torch_custom/fla_npu 源目录，实际执行已安装 wheel。
export PYTHONPATH="$OP_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 不导入 fla_npu 即定位已安装包；在任何 NPU Python 进程启动前注册新 OPP。
GDN_VENDOR_DIR="$(python - <<'PY'
from importlib.util import find_spec
from pathlib import Path
spec = find_spec("fla_npu")
if spec is None or spec.origin is None:
    raise SystemExit("当前 Python 环境未安装 fla_npu")
print(Path(spec.origin).resolve().parent / "opp/vendors/fla_npu_transformer")
PY
)"
export ASCEND_CUSTOM_OPP_PATH="$GDN_VENDOR_DIR:$GDN_VENDOR_DIR/op_api/lib:${ASCEND_CUSTOM_OPP_PATH:-}"

command -v python
command -v atk
atk --version 2>&1 | tee "$RUN_DIR/atk_version.txt"
npu-smi info | tee "$RUN_DIR/npu_info.txt"
python scripts/check_npu_env.py 2>&1 | tee "$RUN_DIR/environment.log"
python scripts/verify_libcust_opapi_md5.py --python 2>&1 | tee "$RUN_DIR/installed_artifacts.log"
```

ATK 版本要求由统一入口检查；当前仓库要求不低于 26.8.8。
`verify_libcust_opapi_md5.py` 应报告新 host 库、kernel 和 Python wrapper 一致；失败时先修复安装。
如已设置 `ATK_ENV`，统一入口会再次激活它，必须确保它就是上述已安装 wheel 的环境。

### 2. 确认诊断入口与用例来源

先检查实际加载路径和两个新增 Profile 符号。此处只加载/查询符号，不启动 GDN kernel。

```bash
python - <<'PY' 2>&1 | tee "$RUN_DIR/profile_symbols.log"
import fla_npu
from fla_npu.ops.ascendc import _aclnn_ctypes
print("fla_npu:", fla_npu.__file__)
print("adapter:", _aclnn_ctypes.__file__)
assert hasattr(_aclnn_ctypes, "_profile_chunk_gated_delta_rule_fwd"), "Python adapter is outdated"
libraries = fla_npu.load_ascendc_opapi_libraries()
for library in libraries:
    print("library:", library._name)
for name in ("aclnnChunkGatedDeltaRuleFwdProfileGetWorkspaceSize", "aclnnChunkGatedDeltaRuleFwdProfile"):
    assert any(hasattr(library, name) for library in libraries), "Missing symbol: " + name
print("Profile symbols: OK")
PY

if git rev-parse HEAD > "$RUN_DIR/source_commit.txt" 2>/dev/null; then
  cat "$RUN_DIR/source_commit.txt"
else
  printf '源码归档未附带 Git 信息；请补记对应提交号。\n' > "$RUN_DIR/source_commit.txt"
fi
cp "$OP_DIR/atk_chunk_gated_delta_rule_fwd_perf.json" "$RUN_DIR/performance_cases.json"
sha256sum "$RUN_DIR/performance_cases.json" > "$RUN_DIR/case_json_sha256.txt"
```

这里使用工程固化的性能 JSON，不重新生成随机用例或修改 shape/值域：

| case id | 原模型 | B / Hk / Hv / T / K / V | dtype / chunk | 状态与变长参数 |
| --- | --- | --- | --- | --- |
| 0 | 推理 | 1 / 16 / 32 / 11274 / 128 / 128 | BF16 / 64 | `cu_seqlens=[0,11274]`，确定性初态，输出 final_state |
| 1 | 训练 | 2 / 32 / 32 / 8192 / 128 / 128 | BF16 / 64 | 定长，无初态，无 final_state |

原模型 case 来自 A5；本节按你的 `ascend910b` 构建在 **A2 重放相同输入合同**，记录的是 A2 结果。
实际 kernel 路由从 `timer_layout.json` / `capture_metadata.json` 读取，不套用 A5 的性能结论。

### 3. 执行两条工程性能 case，同时导出实际输入

```bash
# 清空本命令的 case 范围覆盖，执行性能 JSON 的全部两条 case。
CASE_START= CASE_END= PERFORMANCE_START= PERFORMANCE_END= \
bash tests/atk/run_test_cpu.sh \
  -op=chunk_gated_delta_rule_fwd \
  -npu_device_id="$NPU_DEVICE" -soc=ascend910b -scope=performance \
  2>&1 | tee "$RUN_DIR/atk_performance.log"

# 导出成功时，ATK worker 日志包含 [gdn-profile-input] case=... path=...。
for case_id in 0 1; do
  test -s "$GDN_PROFILE_INPUT_DIR/case_${case_id}.pt"
done
sha256sum "$GDN_PROFILE_INPUT_DIR"/case_*.pt > "$RUN_DIR/input_sha256.txt"
```

产出 `inputs/case_0.pt` 和 `inputs/case_1.pt`。文件保存 ATK 已冻结的 q/k/v/g/beta，
以及 DUT 的 scale、chunk_size、初态、final_state 开关、cu_seqlens、chunk_indices。
dtype 与输入值沿用 executor，数据不重新抽样；额外保存 case id、scenario 和输入格式版本。

`GDN_PROFILE_INPUT_DIR` 是新增可选开关，必须为绝对路径，并由 ATK worker 继承。
只有 DUT 的 `init_by_input_data` 会保存，benchmark/golden 不保存，`__call__` 性能循环内不导出。
同目录同 case id 已有文件时保留首份，因此重跑要使用新的 RUN_DIR。未设置开关时不产生导出文件。
ATK 默认性能报告和系统 profile 位于 `$RUN_DIR/atk/perf/` 下；它们与下一步的逐核 Tik/Tok CSV 一起保留。
性能用例只跑 DUT，不运行 CPU golden 或六 ACLNN 双标杆。

如只采 case 0，可将上面的范围前缀改为 `CASE_START= CASE_END= PERFORMANCE_START=0 PERFORMANCE_END=1`，
并把本节文件检查和下一节采集循环都改为 `for case_id in 0`；end 是排他边界。

### 4. 对同一批实际输入采集 Tik/Tok，自动导出逐核 CSV

```bash
for case_id in 0 1; do
  case_output="$RUN_DIR/timer/case_${case_id}"
  mkdir -p "$case_output"
  python "$UT/profile_gdn.py" \
    --inputs "$GDN_PROFILE_INPUT_DIR/case_${case_id}.pt" \
    --device "$NPU_DEVICE" --warmup 5 --check-output \
    --output-dir "$case_output" \
    2>&1 | tee "$case_output/profiling.log"
done
printf '本轮结果目录：%s\n' "$RUN_DIR"
```

每条 case 使用新 Python 进程，先预热 5 次正式调用，再执行一次正式对照和一次诊断调用。
诊断 timer 每次新建并清零，同步后保存 raw、严格解析、打印逐核阶段统计。
`--check-output` 对十项输出进行 exact 对比，结果写入 `capture_metadata.json`；失败时 raw 保留。
这是正式/诊断调用对比，不能替代 ATK CPU/六算子精度验收。打点耗时包含观察扰动。

默认成功结果结构：

```text
gdn_timer_<时间>_<pid>/
  source_commit.txt / performance_cases.json / case_json_sha256.txt / input_sha256.txt
  atk_version.txt / npu_info.txt / environment.log / installed_artifacts.log / profile_symbols.log
  atk_performance.log
  inputs/
    case_0.pt
    case_1.pt
  atk/perf/                  # ATK 性能报告与系统采集文件，内部结构由 ATK 决定
  timer/
    case_0/
      gdn_npu_time_raw.csv
      gdn_npu_time_parsed.csv
      timer_layout.json
      capture_metadata.json
      profiling.log
    case_1/                 # 与 case_0 相同的五份文件
```

默认单位是 cycles，未确认时钟时打印 `Perfetto skipped` 属于预期行为；不会默认生成 trace JSON。

### 5. 可选：确认时钟后生成 Perfetto

确认目标 GetSystemCycle 的 cycles/us **以及 AIC/AIV 共同时钟**后，在同一 shell 设置
`CYCLES_PER_US` 为已核实的正数；下面不提供猜测值。转换重用 raw，不再次启动 kernel。

```bash
: "${CYCLES_PER_US:?请先设置已核实的 GetSystemCycle cycles/us}"
for case_id in 0 1; do
  python "$UT/profile_gdn.py" \
    --parse-only "$RUN_DIR/timer/case_${case_id}/gdn_npu_time_raw.csv" \
    --output-dir "$RUN_DIR/timer/case_${case_id}" \
    --cycles-per-us "$CYCLES_PER_US" --shared-clock
done
```

每条 case 新增 `gdn_npu_time.json`，CSV 新增 `duration_us` 列。在 https://ui.perfetto.dev/ 加载各自 JSON。
这里的 `--shared-clock` 是调用者对平台事实的确认；不是脚本自动校准。

### 6. 阅读结果与排查失败

先看 `capture_metadata.json` 的设备、库路径、输入/选项、实际 kernels 及 output_comparison，
再读 `profiling.log` 中按 kernel/stage/AIC 或 AIV 分类的 min/median/max cycles。
在同一 kernel、同一核类型内比较 KERNEL_TIMING / MAIN_BODY，找负载差异；
Phase6 的 PHASE6_HO_WAIT 测既有串行 SyncAll，重叠分支不记录该事件。
两个 DONE 是瞬时下发边界，不将其 duration 解释为业务阶段完成耗时。
每个 kernel 独立取时间 origin；不同 launch 的时间线不能用来推算端到端耗时。
正式 Task Duration 仍读 ATK 系统 profile 的 op_summary。

| 现象 | 处理 |
| --- | --- |
| `atk` 不存在/版本不满足 | 在安装 wheel 的环境准备符合仓库要求的 ATK 后重新运行 |
| 找不到 `gdn_reference` / `atk_role_contract` | 检查 OP_DIR 的绝对路径，以及 ATK worker 是否继承 PYTHONPATH |
| 没有 `inputs/case_*.pt` | 确认 executor 已同步、worker 继承 GDN_PROFILE_INPUT_DIR、DUT 已初始化，查看 ATK 日志 |
| `Missing symbol` 或 adapter outdated | 新 Python、host、OPP 不一致；重建安装并确认实际加载路径 |
| raw 无事件/缺核/坏布局 | 保留 raw、metadata、两种采集日志；检查新 kernel 是否安装及 ENABLE_TIMER/实际路由 |
| 十项输出 exact 对比失败 | 保留 raw 和失败 metadata，按输出有效区定位差异；不直接放宽阈值形成通过结论 |

需要重跑时先重新执行步骤 1 创建新 RUN_DIR；离线转换既有结果时把 RUN_DIR 设为原目录。
把完整 RUN_DIR 带回即可分析，尤其保留 inputs、case JSON/哈希、raw、metadata、日志和 ATK 报告。

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
python -m unittest discover -s ci/tests -p test_gdn_profile_input_export.py
python tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py
```

本机计时套件 20 项、前向消费合同 2 项通过；原有 mutation 合同 7 项也通过。计时套件覆盖布局、两路由、核映射、坏记录拒绝、配对/时间窗/阶段顺序、
超限、缺核、不完整路由、混合等待分支、整数 trace 精度、诊断 ABI、旧十项输出、打包头路径，以及成功启动后 timer 版本更新和失败/grad 拒绝分支。
导出 fixture 是 synthetic，测试结束清理，不是设备结果。
2026-10-12 新增 ATK 输入导出合同 5 项通过，覆盖默认关闭、非 DUT 隔离、实际值/dtype/变长/状态参数、
首份快照保留、绝对路径校验及初始化/性能循环隔离；前向输出 2 项和 timer 20 项重新运行通过。
当前 README 的 10 个 Bash 代码块通过 `bash -n`，相对链接和 Python 语法检查通过；设备命令未执行。
原 ACLNN 合同脚本在本次修改前已失败：`prepare kernel 缺少非 exp2 模式的局部指数换底选择`，
本次只调整其 optional timer 调用正则，仍报告同一失败。
原 ctypes 完整套件在当前无 torch 环境有一项报错、一项设备用例跳过；三项 GDN 回归（训练/推理输出、norm/别名、预留 gate 参数拒绝）用 mock 单独运行通过，其中前两项也由新套件复用。

后续设备需验证 CANN 编译/实际加载、timer 错误输入、ENABLE_TIMER=0、逐核映射与 GM 边界；
覆盖两条前向路径、串行/重叠、短序列 idle AIC、tail/GVA、fixed/varlen、初态/末态，
完成 CPU/六算子双标杆、确定性和内存检测，并测量打点扰动、确认时钟后保留采集原始证据。
独立 Step 9 的源码审核见 `../../docs/timer_review.md`，不能替代这些设备检查。
