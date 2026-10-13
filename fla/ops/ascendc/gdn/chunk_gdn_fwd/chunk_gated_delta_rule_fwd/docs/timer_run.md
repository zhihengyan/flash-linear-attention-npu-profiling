# GDN profiling 操作说明

流程及测量含义见 [timer_flow.md](timer_flow.md)，本轮源码/独立审核/设备状态见 [timer_report.md](timer_report.md)。本机是 Windows 源码工作区 `D:\Code\flash-linear-attention-npu`，未发现 Docker/CANN/NPU 工具。以下在已具备 NPU 的 Linux 环境执行；已有两个原性能 case 均为 A5，默认 SoC 为 ascend950。A2/A3 源码路径也已接入，实际编译/回归需分别使用 ascend910b/ascend910_93。

## 容器、环境与源码

宿主机使用 `docker ps --format 'table {{.Names}}\t{{.Status}}'` 确认现有 NPU 容器；将实际名字设为 NPU_CONTAINER 后执行 `docker exec -it "$NPU_CONTAINER" /bin/bash`。容器须已透传 NPU、含 ATK/CANN 和本分支源码。当前没有容器名及挂载路径证据，不编造环境名称。

容器内进入真实仓库根目录，沿镜像约定激活已有 Python/ATK 环境；构建、安装和测试固定使用同一个解释器。默认 CANN 路径按仓库指南：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
which python
python -m pip --version
which atk
atk --version
npu-smi info
git branch --show-current
```

确认当前分支包含本次改动、SoC 是 A5、ATK 不低于仓库要求的 26.8.8。镜像若使用其它 CANN 路径，沿实际 set_env.sh 激活。不要切换至不同的 Python 环境后安装。

## 构建并安装本轮准确产物

本次包含 Python wrapper 和独立 Profile ACLNN 符号，需要完整 wheel；清空遗留算子筛选，保持原构建脚本：

```bash
FLA_NPU_SOC=ascend950 FLA_NPU_OPS= python scripts/build_wheel.py
```

复制构建末尾输出的准确 wheel 绝对路径到 WHEEL_PATH，执行 `python -m pip install --force-reinstall --no-cache-dir --no-deps "$WHEEL_PATH"`。只安装本轮成功归档的文件，不按提交号或字典序选择旧 wheel；按构建脚本末尾的 staging/归档校验确认内容一致。然后重启测试进程。

## 安装后主入口

在仓库根目录运行：

```bash
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py --device 0
```

默认执行原 `atk_chunk_gated_delta_rule_fwd_perf.json` 的 case 0/1；可用 `--cases 0` 或 `--cases 1` 选择。脚本按当前 wheel 路径注册 OPP，再检查 ATK/Python、npu-smi、安装产物和 Profile 符号，运行原 ATK performance_device。executor 仅在性能循环外保存该 case 的真实输入值与全部参数；随后同输入普通/诊断调用进行十项输出精确对照，导出 raw、CSV、layout/capture/run metadata 与日志。原 CPU/六 ACLNN 比较规则和性能统计不变。本入口是采集工具，完整精度/确定性/内存验收仍执行 ATK 原 all。

默认结果在 `tests/atk/chunk_gated_delta_rule_fwd/atk_output/gdn_timer_<时间戳>_<随机标识>/`（仓库已忽略 atk_output）。也可用 `--output-dir` 指定一个尚不存在的目录。查看 run_metadata.json 的 status、原性能报告及实际成功用例，再看 timer/case_N 的 capture_metadata.json、timer_layout.json、gdn_npu_time_parsed.csv。采样 actual/sampled/truncated 字段给出每核每阶段计数；出现截断只能分析前 32 个调用。

复用已归档 wheel 时，追加 `--reference-wheel "$WHEEL_PATH"`。预检直接比较已安装 fla_npu 与指定 wheel 内全部运行文件的 SHA256，包括 Python、op_api 动态库、kernel .o 及配置；缺文件、内容不一致或额外遗留的 .so/.o 会中止采集。结果记录在 installed_artifacts.log 和 run_metadata.json 的 reference_wheel 中，包含 wheel 绝对路径、SHA256、distribution/version 和检查数量。这项检查在性能循环外完成；无需保留该 wheel 对应的 build/ staging。未指定此参数时仍使用原 build/ 与当前 Python 源码的安装检查。

原始数据留在内部环境时，使用 tests/ut/summarize_timer.py 生成小型 brief.txt 及阶段/逐核汇总，首轮只需粘贴摘要。命令、指标含义与分析顺序见 [timer_analysis.md](timer_analysis.md)。

成功标志是原性能用例确实执行、十项普通/诊断输出一致、实际路由所有 launch 有有效 header 与每核总窗口、CSV 严格校验完成，并最终打印 `Profiling complete:`。空 raw、缺 kernel、缺配对、负 duration、越界等会报失败且保留 raw/日志，不能只根据子进程 exit 0 宣告设备成功。

核实 GetSystemCycle 频率和 AIC/AIV 共同时钟后，可在同一主入口追加 `--cycles-per-us 实际频率 --shared-clock`，生成 gdn_npu_time.json，使用 [Perfetto](https://ui.perfetto.dev/) 加载。默认只输出 cycles，不猜频率；各 kernel 独立原点，窗口可重叠。

## Stage6 优化：先跑 B 与严格 A/B

**先采一次 B 可以沿用上面的编译→安装→主入口流程。** A 是未优化版本（codex/gdn-profiling，实验起点 `809bce6887639e831caa856b8ac3a197a6a94294`），B 是 codex/gdn-stage6-overlap 的候选代码。每次运行主入口先采普通调用 msprof 性能，再用当次保存的同输入做 timer 诊断；主要收益比较取 run_metadata.json 中普通调用的 Task Duration(us)，不是 timer cycles。

### 先跑优化后的 B1

1. 在内部 Linux 仓库确认已同步本次 B 的代码和文档，再按前述环境步骤激活同一个 Python/ATK/CANN。检查 Prepare Arch35 的 Stage6 已有 prefetchK 条件。构建 A/B 前确认各自 kernel 改动已提交并记录 commit；source_commit 只记录采集入口工作区的 HEAD，不包含未提交 diff，也不能用来识别切换安装的 wheel。
2. 构建完整 B wheel，记录构建输出的准确路径，强制安装，并结束旧测试 Python 进程。沿用前述主入口的安装产物检查，不只检查 libcust_opapi.so；这次变更是 kernel，需确认 kernel .o 也来自 B。
3. 先做改动路径的精度检查，再进行性能采集。完整设备验收见 [validation.md](validation.md)；可用原 Prepare 精度矩阵的 G2/V128 对齐和尾块作前置检查：

```bash
# 在 B 源码和准确安装的 B wheel 下，仓库根目录运行
ATK_GM_INIT_MODE=off CASE_START=24 CASE_END=48 \
  bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=0 -scope=accuracy -soc=ascend950
ATK_GM_INIT_MODE=off CASE_START=216 CASE_END=240 \
  bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=0 -scope=accuracy -soc=ascend950
```

这两个区间来自当前 Prepare README 的 24 组合法 flag 映射，CASE_END 不含右端；属于前置检查，不替代完整 Prepare/GDN 精度、MSS、确定性和 Example/ST。

编译安装和 B1 采集命令如下。B_WHEEL 必须填写刚才构建输出的准确绝对路径：

```bash
git branch --show-current
git rev-parse HEAD
git status --short
FLA_NPU_SOC=ascend950 FLA_NPU_OPS= python scripts/build_wheel.py

# 将 B_WHEEL 设为本轮构建输出的准确 wheel 绝对路径
python -m pip install --force-reinstall --no-cache-dir --no-deps "$B_WHEEL"

python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py \
  --device 0 --cases 0 --warmup 5 \
  --reference-wheel "$B_WHEEL" \
  --output-dir tests/atk/chunk_gated_delta_rule_fwd/atk_output/ab_stage6/B1
```

`ab_stage6/B1` 在执行前必须不存在，不提前 mkdir 这个叶目录；脚本自行创建并拒绝覆盖旧结果。B2/B3 使用不同目录名。该命令的 warmup=5 只控制随后 timer 重放的预热，普通 ATK 性能的预热/次数配置继续沿用原入口，A/B 必须相同。

主入口中的 `ten outputs exactly match uninstrumented call` 是 **B 普通调用与 B 诊断调用** 的一致性，不能替代 B 对 CPU golden 的精度检查，也不表示 B 与 A 的数值已经比较。若 B1 沿用未固定输入的原性能 JSON，它先作为新 B 采集，与历史 A0 比较只作为趋势参考。

### 严格 A1→B1→A2 对照

不要只将历史 A0 与新 B1 的均值相减就宣告收益。使用相同设备、Python/ATK/CANN、性能配置、用例和输入，重新运行 A1→B1→A2，并重复批次。**A 和 B 各编译一次、归档完整 wheel 后，可以只切换安装 wheel，始终使用当前分支这一份主入口；不用每轮切分支或重新编译。** 两个 wheel 均须包含本次独立 Profile 接口和相同 timer 布局，使用相同 SoC 和构建选项；A 是上面的固定基线，B 是已提交优化版本。分别保存到 checkout 外的 A/B 目录并记录构建 commit 与 wheel SHA256，避免后一次构建覆盖前一次同名文件。

当前主入口没有“从已有 case_0.pt 重放 ATK 普通性能”的参数；`profile_gdn.py --inputs` 仅重放普通/诊断对照和 timer，不生成原 ATK 普通性能报告。因此复现性能需要 ATK 自身固定输入。建议先从内部原始运行保存的 performance_cases.json 建立一个 A/B 共用的测试文件，保持原 shape、dtype、属性和值域；default_seed 已固定时保留，null 时为 case 0/1 选择一次固定整数，并在 A/B 中始终使用这一份配置。原 Prepare 冻结用例也使用 default_seed。

下面仅改变随机种子和文件排版，不改变原用例其它内容。在 Linux 运行前，将 GDN_ORIGINAL_RUN 设为**内部完整原始运行目录**（包含 performance_cases.json），不是当前本机只含几个摘要的 atk_output 目录。共用文件保存于 checkout 外，测试期间始终使用这一份：

```bash
# GDN_ORIGINAL_RUN 指向内部原始完整运行；这里只准备一份固定配置
python - "$GDN_ORIGINAL_RUN/performance_cases.json" /tmp/gdn_stage6_ab_cases.json <<'PY'
import json, sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
if target.exists():
    raise SystemExit("共用配置已存在；复用原文件，不覆盖或重新选种子")
cases = json.loads(source.read_text(encoding="utf-8-sig"))
for case in cases:
    if case.get("default_seed") is None:
        case["default_seed"] = 20261013 + int(case["id"])
target.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
```

归档两个 wheel 后，保留含 `--reference-wheel` 参数的当前采集脚本和相同测试配置。先完成对应版本的精度前置检查，再依次安装/采集 A1、B1、A2。固定 seed 后的 A1 是新基线，不宣称它与历史 A0 的输入值相同。

| 阶段 | 安装的归档 wheel | 主入口输出目录（必须新建） |
| --- | --- | --- |
| A1 | A_WHEEL，构建自固定基线 `809bce6887639e831caa856b8ac3a197a6a94294` | `atk_output/ab_stage6/A1` |
| B1 | B_WHEEL，构建自已提交的 Stage6 优化版本 | `atk_output/ab_stage6/B1`；如果先跑 B 已用此名，改用 B2 |
| A2 | 与 A1 完全相同的 A_WHEEL | `atk_output/ab_stage6/A2` |

```bash
# 在同一采集工作区只复制一次；A_WHEEL/B_WHEEL 填写归档文件的准确绝对路径
cp /tmp/gdn_stage6_ab_cases.json tests/atk/chunk_gated_delta_rule_fwd/atk_chunk_gated_delta_rule_fwd_perf.json

python -m pip install --force-reinstall --no-cache-dir --no-deps "$A_WHEEL"
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py \
  --device 0 --cases 0 --warmup 5 --reference-wheel "$A_WHEEL" \
  --output-dir tests/atk/chunk_gated_delta_rule_fwd/atk_output/ab_stage6/A1

python -m pip install --force-reinstall --no-cache-dir --no-deps "$B_WHEEL"
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py \
  --device 0 --cases 0 --warmup 5 --reference-wheel "$B_WHEEL" \
  --output-dir tests/atk/chunk_gated_delta_rule_fwd/atk_output/ab_stage6/B1

python -m pip install --force-reinstall --no-cache-dir --no-deps "$A_WHEEL"
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py \
  --device 0 --cases 0 --warmup 5 --reference-wheel "$A_WHEEL" \
  --output-dir tests/atk/chunk_gated_delta_rule_fwd/atk_output/ab_stage6/A2
```

每条采集命令都启动新的 Python 进程；不要在已加载 A 动态库的交互进程中安装 B 后继续测试。`--reference-wheel` 仅指定校验基准，不执行安装，上面的 pip 步骤必须成功。

比较前核对每轮 metadata 的 case_json_sha256、device、性能记录数量及 reference_wheel：A1/A2 的 wheel SHA256 应相同，B 应不同；配置哈希应一致。同一采集工作区的 source_commit 可以三轮相同，source_commit_role=profiling_checkout 表示它是入口工作区版本；被测版本以归档构建 commit 和通过校验的 wheel SHA256 识别，不能只看可能相同的 wheel version 字符串。脚本会重新保存 inputs/case_0.pt：输入文件 SHA256 相同即可证明文件一致；哈希不同时可能是序列化差异或输入变化，须在内部加载快照，逐项比较 q/k/v/g/beta、initial_state 和 kwargs 的 dtype、shape、值及选项，不能直接当相同输入。若固定 seed 后实际输入不一致，先修正输入冻结再比较性能。

收益计算以每次普通调用七项 kernel Task Duration(us) 的时间和为主，分别保留 Prepare/H/O 分布。先检查 A1 与 A2 的波动，再判断 B 是否在多批次中稳定下降；不相加各项 p95，不用诊断窗口算收益。只传出 A1/B/A2 的 run_metadata.json 即可做首轮性能比较；B 的 brief/阶段表用于解释流水变化，raw 和大 CSV 留在内部。

## 可选重放与离线解析

次入口为 `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/profile_gdn.py`，只接受原 case 保存的输入或原 raw：

```bash
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/profile_gdn.py --inputs "$INPUT_SNAPSHOT" --device 0 --check-output --output-dir "$REPLAY_RESULT"
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/profile_gdn.py --parse-only "$RAW_CSV" --output-dir "$PARSED_RESULT"
```

独立重放同样在导入 torch_npu 前注册当前 wheel OPP。离线解析不需要 torch/CANN，但必须使用与 raw 对应的布局 v2 源码；不能解析旧布局数据。真实频率参数也适用于该入口。

## 回归与开关对照

先运行源码检查与离线单元测试：

```bash
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/check_timer_source.py
python -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p test_timer_contract.py -v
```

设备回归：按 tests/atk/README.md 准备包内 OPP 后分别执行四个受影响算子 chunk_gated_delta_rule_fwd、chunk_gated_delta_rule_fwd_prepare、chunk_fwd_h、chunk_fwd_o 的 `bash tests/atk/run_test_cpu.sh -op=算子名 -npu_device_id=0 -soc=ascend950 -scope=all`；随后运行 `python examples/flash_gated_delta_rule.py`。A2/A3 按各自支持范围和 ATK README 执行，不把未选中的 key 记为覆盖。

ENABLE_TIMER 默认 1，只有非 null timer 生效。彻底关闭的编译对照可将 canonical 头默认值改成 0、重新构建完整 wheel 并运行普通验收，或在四个算子的 add_ops_compile_options 中临时传入 `-DENABLE_TIMER=0`（本轮未添加新的公共构建选项）。关闭后 Profile 采集应因空数据失败；开销比较使用相同 case/版本/预热/次数及 msprof Task Duration(us)，不要拿诊断窗代替性能验收口径。
