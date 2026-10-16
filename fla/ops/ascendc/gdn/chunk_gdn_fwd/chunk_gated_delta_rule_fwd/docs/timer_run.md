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

当前Stage3实验分支移植了可选归档wheel校验：`--reference-wheel "$WHEEL_PATH"`。默认仍比较当前build staging；切换归档K32/B wheel且保留另一版本build时，应安装指定wheel并在同一份采集脚本上添加该参数。脚本直接核对安装产物与wheel中的`.so`/`.o`/Python/配置，支持wheel根目录及`.data/purelib`、`.data/platlib`布局；不匹配仍报错。metadata记录wheel SHA256，`source_commit_role=profiling_checkout`说明source_commit仅标识采集脚本checkout，各wheel的源码/构建日志仍须单独保存。当前实验状态和完整A/B命令见 [validation.md](validation.md)。

默认结果在 `tests/atk/chunk_gated_delta_rule_fwd/atk_output/gdn_timer_<时间戳>_<随机标识>/`（仓库已忽略 atk_output）。也可用 `--output-dir` 指定一个尚不存在的目录。查看 run_metadata.json 的 status、原性能报告及实际成功用例，再看 timer/case_N 的 capture_metadata.json、timer_layout.json、gdn_npu_time_parsed.csv。采样 actual/sampled/truncated 字段给出每核每阶段计数；出现截断只能分析前 32 个调用。

原始数据留在内部环境时，使用 tests/ut/summarize_timer.py 生成小型 brief.txt 及阶段/逐核汇总，首轮只需粘贴摘要。命令、指标含义与分析顺序见 [timer_analysis.md](timer_analysis.md)。

成功标志是原性能用例确实执行、十项普通/诊断输出一致、实际路由所有 launch 有有效 header 与每核总窗口、CSV 严格校验完成，并最终打印 `Profiling complete:`。空 raw、缺 kernel、缺配对、负 duration、越界等会报失败且保留 raw/日志，不能只根据子进程 exit 0 宣告设备成功。

核实 GetSystemCycle 频率和 AIC/AIV 共同时钟后，可在同一主入口追加 `--cycles-per-us 实际频率 --shared-clock`，生成 gdn_npu_time.json，使用 [Perfetto](https://ui.perfetto.dev/) 加载。默认只输出 cycles，不猜频率；各 kernel 独立原点，窗口可重叠。

## 可选重放与离线解析

次入口为 `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/profile_gdn.py`，只接受原 case 保存的输入或原 raw：

```bash
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/profile_gdn.py --inputs "$INPUT_SNAPSHOT" --device 0 --check-output --output-dir "$REPLAY_RESULT"
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/profile_gdn.py --parse-only "$RAW_CSV" --output-dir "$PARSED_RESULT"
```

独立重放同样在导入 torch_npu 前注册当前 wheel OPP。离线解析不需要 torch/CANN，但必须使用与 raw 对应的布局 v2 源码；不能解析旧布局数据。真实频率参数也适用于该入口。

## 回归与开关对照

原计时接入分支的 `check_timer_source.py` 比对固定main的同步文本，不能用于当前性能候选的同步等价判定（K32增加专用helper，Stage3实验增加一对V_MTE3）。当前候选使用以下包含历史K32、当前Stage3、wheel校验及timer/摘要的离线单元测试；原检查器未放宽：

```bash
python -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p 'test_*.py' -v
```

设备回归：按 tests/atk/README.md 准备包内 OPP 后分别执行四个受影响算子 chunk_gated_delta_rule_fwd、chunk_gated_delta_rule_fwd_prepare、chunk_fwd_h、chunk_fwd_o 的 `bash tests/atk/run_test_cpu.sh -op=算子名 -npu_device_id=0 -soc=ascend950 -scope=all`；随后运行 `python examples/flash_gated_delta_rule.py`。A2/A3 按各自支持范围和 ATK README 执行，不把未选中的 key 记为覆盖。

ENABLE_TIMER 默认 1，只有非 null timer 生效。彻底关闭的编译对照可将 canonical 头默认值改成 0、重新构建完整 wheel 并运行普通验收，或在四个算子的 add_ops_compile_options 中临时传入 `-DENABLE_TIMER=0`（本轮未添加新的公共构建选项）。关闭后 Profile 采集应因空数据失败；开销比较使用相同 case/版本/预热/次数及 msprof Task Duration(us)，不要拿诊断窗代替性能验收口径。
