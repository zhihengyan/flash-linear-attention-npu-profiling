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

## 本轮 Prepare Stage3–5 定位

分支为 `codex/gdn-prepare-stage3-5-timer`，直接基于 `codex/gdn-profiling` 的809bce68。具体原操作与观测含义见 [诊断设计](timer_prepare_stage35_design.md)。先按上面的完整编译和安装步骤更新到本轮v3 wheel，再运行：

```bash
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py \
  --device 0 --cases 0 --reference-wheel "$WHEEL_PATH" \
  --prepare-stage35 --timer-repeats 2 \
  --output-dir tests/atk/chunk_gated_delta_rule_fwd/atk_output_stage35
```

输出目录必须尚不存在。命令自动完成原普通msprof性能采集和输入保存，再对每核pack ordinal的0/16/32/48窗口各采8pack，逐窗口coarse/detail成对采集；重复2轮，共16次诊断调用，均在普通性能循环外。每次做同输入十输出精确对照、严格解析并自动生成摘要。只有 --prepare-stage35 而不加repeats则8次；不加该开关时仍仅1次粗采集。可用 --prepare-pack-starts 16 32 缩小区间。每次timer约117.38MiB，原始文件留内部。

结果位置：

- `run_metadata.json`：普通Task Duration(us)、准确wheel哈希、输入snapshot哈希和采样计划。
- `timer/case_0/stage35_analysis/brief.txt`：各窗口细分阶段与粗/细扰动对照。
- 同目录 `perturbation.csv`、`window_stage_summary.csv`、`summary.json`：较小的成对扰动/阶段汇总。
- `task_summary.csv`、`pack_summary.csv`：按work ID和本核pack关联，留内部定点查看。
- `timer/case_0/{coarse,detail}_pack{0,16,32,48}_r{0,1}/analysis/`：每次brief/core/stage/longest；父目录保留raw、parsed、layout、capture metadata和日志。

优先提供run_metadata.json、stage35_analysis/brief.txt和perturbation.csv；不足时再补window_stage_summary.csv。无需传raw。若粗/细窗口变化明显或不一致，先降低粒度/补pipe信息；工具不把任何窗口当launch耗时占比或收益上限。这里的扰动比较仅覆盖固定结束戳之前，Finish之后的count/tag发布与DCCI不在窗口内。

粗/细同v3对照也不能排除整个计时实现改变编译资源。可安装保存好的旧A wheel，用同一新入口单采普通性能：

```bash
python -m pip install --force-reinstall --no-cache-dir --no-deps "$A_WHEEL"
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py \
  --device 0 --cases 0 --reference-wheel "$A_WHEEL" --ordinary-only \
  --output-dir tests/atk/chunk_gated_delta_rule_fwd/atk_output_stage35_A
```

安装后开启新测试进程。ordinary-only仍运行普通ATK/msprof、保存输入和wheel身份，跳过Profile符号和timer，可使用原v2 A包。新v3诊断主入口也先采普通性能；据同条件重复结果核对普通调用是否退化，历史A0只作背景。

读取顺序：先看扰动与采样有效性，再比较两个稳定期区段的S3_KKT_WAIT/S3_POST_KKT与S4_INPUT_WAIT，之后看S3_LEAF_SOLVE、Y完成等待/回读、A transfer及S7_A_READY_WAIT。后者在原Stage6输入等待之后，只显示此时剩余的A依赖。PREPARE_PACK包含上一pack的AIV写回；最后当前pack写回在循环外，不是本pack输出完成时延。issue窗含原内部等待且可嵌套，不能累加为纯计算或全算子时间。本轮先定位，确认候选后再另开优化分支。

## 安装后通用主入口

在仓库根目录运行：

```bash
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py --device 0
```

默认执行原 `atk_chunk_gated_delta_rule_fwd_perf.json` 的 case 0/1；可用 `--cases 0` 或 `--cases 1` 选择。脚本按当前 wheel 路径注册 OPP，再检查 ATK/Python、npu-smi、安装产物和 Profile 符号，运行原 ATK performance_device。executor 仅在性能循环外保存该 case 的真实输入值与全部参数；随后同输入普通/诊断调用进行十项输出精确对照，导出 raw、CSV、layout/capture/run metadata 与日志。原 CPU/六 ACLNN 比较规则和性能统计不变。本入口是采集工具，完整精度/确定性/内存验收仍执行 ATK 原 all。

默认结果在 `tests/atk/chunk_gated_delta_rule_fwd/atk_output/gdn_timer_<时间戳>_<随机标识>/`（仓库已忽略 atk_output）。也可用 `--output-dir` 指定一个尚不存在的目录。查看 run_metadata.json 的 status、原性能报告及实际成功用例，再看 timer/case_N 的 capture_metadata.json、timer_layout.json、gdn_npu_time_parsed.csv。actual/sampled/truncated给出计数；Prepare为选定pack窗口，H/O/Phase6为前32调用，不外推未记录的部分。

原始数据留在内部环境时，使用 tests/ut/summarize_timer.py 生成小型 brief.txt 及阶段/逐核汇总，首轮只需粘贴摘要。命令、指标含义与分析顺序见 [timer_analysis.md](timer_analysis.md)。

成功标志是原性能用例确实执行、十项普通/诊断输出一致、实际路由所有 launch 有有效 header 与每核总窗口、CSV 严格校验完成，并最终打印 `Profiling complete:`。空 raw、缺 kernel、缺配对、负 duration、越界等会报失败且保留 raw/日志，不能只根据子进程 exit 0 宣告设备成功。

核实 GetSystemCycle 频率和 AIC/AIV 共同时钟后，可在同一主入口追加 `--cycles-per-us 实际频率 --shared-clock`，生成 gdn_npu_time.json，使用 [Perfetto](https://ui.perfetto.dev/) 加载。默认只输出 cycles，不猜频率；各 kernel 独立原点，窗口可重叠。

## 可选重放与离线解析

次入口为 `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/profile_gdn.py`，只接受原 case 保存的输入或原 raw：

```bash
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/profile_gdn.py --inputs "$INPUT_SNAPSHOT" --device 0 --check-output --output-dir "$REPLAY_RESULT"
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/profile_gdn.py --parse-only "$RAW_CSV" --output-dir "$PARSED_RESULT"
```

独立重放同样在导入 torch_npu 前注册当前 wheel OPP。本分支离线解析仅支持布局v3；原v2 raw使用gdn-profiling分支。独立重放可追加 --prepare-detail --prepare-pack-start 16 --prepare-pack-count 8；重新解析不能补回旧raw未记录的数据。真实频率参数也适用于该入口。

## 回归与开关对照

先运行源码检查与离线单元测试：

```bash
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/check_timer_source.py
python -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p test_timer_contract.py -v
```

设备回归：按 tests/atk/README.md 准备包内 OPP 后分别执行四个受影响算子 chunk_gated_delta_rule_fwd、chunk_gated_delta_rule_fwd_prepare、chunk_fwd_h、chunk_fwd_o 的 `bash tests/atk/run_test_cpu.sh -op=算子名 -npu_device_id=0 -soc=ascend950 -scope=all`；随后运行 `python examples/flash_gated_delta_rule.py`。A2/A3 按各自支持范围和 ATK README 执行，不把未选中的 key 记为覆盖。

ENABLE_TIMER 默认 1，只有非 null timer 生效。彻底关闭的编译对照可将 canonical 头默认值改成 0、重新构建完整 wheel 并运行普通验收，或在四个算子的 add_ops_compile_options 中临时传入 `-DENABLE_TIMER=0`（本轮未添加新的公共构建选项）。关闭后 Profile 采集应因空数据失败；开销比较使用相同 case/版本/预热/次数及 msprof Task Duration(us)，不要拿诊断窗代替性能验收口径。
