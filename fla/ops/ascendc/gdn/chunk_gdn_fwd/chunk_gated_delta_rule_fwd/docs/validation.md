# Prepare K32 开发期验证记录

初始日期：2026-10-14；实测分析更新：2026-10-15。候选分支 `codex/gdn-prepare-k32`，从 `codex/gdn-profiling` / `809bce6887639e831caa856b8ac3a197a6a94294` 创建。方案见 [prepare_k32_design.md](prepare_k32_design.md)。当前完成候选设计、编码、离线检查和用户提供的三轮普通性能分析；用户报告本轮精度测试全部通过。相对历史A0七kernel合计均值下降5.623%，新鲜A基线下的5%收益及完整ATK正式验收仍待完成。本记录保留至设备验收完成，届时按阶段05收敛结果至对应 ATK README。

## 性能追踪

| 版本 | 主变量 | 证据 | 判断 |
| --- | --- | --- | --- |
| A0，历史基线 | 原 K64 求解，未采用 Stage6/H 实验 | 普通调用20组，七 kernel均值1708.3914us；Prepare715.08995us | 预算参考，需同环境新鲜 A |
| msopprof_r0 | 原始实现 Prepare replay | 708.430969us；AIC Cube77.56251%、MTE1 23.2370% | 支持尝试计算/加载减量，不能替代普通性能基线 |
| B1/B2/B3，K32 候选 | 三处 FP32 MMAD 使用有效半区 K | 用户报告精度通过；各20组普通调用，七项均值1613.28590/1612.17790/1611.52665us，合并1612.33015us；Prepare620.64430us | 相对历史A0下降5.623%，Prepare下降13.208%；建议保留候选并补新鲜A对照 |

三轮CSV与各自metadata的140行名称/耗时完全匹配，四轮输入hash及case JSON hash相同。B数据对应优化提交 `dd707730627b29b1862fd335dfbc76e7f0e2666d`。收益归因、逐kernel/逐轮分布、5%余量和数据边界见 [timer_analysis_k32_b1_b3_2026-10-15.md](timer_analysis_k32_b1_b3_2026-10-15.md)。本批未带出构建/安装/频率日志及精度case报告，不据此宣布完整设备验收完成；case1等其它支持场景尚无本候选的新结果。

普通性能固定统计 Cast、三次 Transpose、Prepare、H、O 七个 kernel 的 `op_summary.csv` / `Task Duration(us)`；以每次调用七行之和计算分布，再比较重复运行的均值、中位数和波动。不得把 timer scope sum、pipe占比或 msopprof replay 当性能验收值。首轮仅 case0；case1及其它支持场景在正式接受前回归。

## 已完成的离线检查

Windows 源码环境无 CANN、NPU及 AscendC 编译器。Python3.12.14 / NumPy2.3.5，执行：

```bash
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/test_prepare_k32.py
python -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p test_timer_contract.py -v
```

- K32 检查4项通过：从实际 C++ 描述符/模板调用读取配置，用独立 NZ 地址模型核对非对称输入的源半区、转置、步长及 L0A/B 写满8KiB、未触及后8KiB。
- 9种有效行数1/10/16/31/32/33/48/63/64 × 4种种子 × 3种下三角幅度，共108组 FP32 模型：Y与原 K64公式一致，A与原公式及 FP64 独立逆矩阵对照通过；覆盖两块 leaf 和尾块补齐。
- 负例证实模型能发现转置 dstStride=2、错误半区和错误源步长。
- 与809基线对照：原通用 FP32/BF16/WU helper、Stage4预填、Dump、其它Stage的代码 token保持；两处修改函数的原event顺序、timer scope保持。
- 原 timer/采集数据契约14项通过；`git diff --check` 通过。

上述是布局/代数和源码检查，不是实际 LoadData/MMAD 指令仿真，不验证硬件归约舍入、MSS、动态同步或设备精度。108组模型不是正式 ATK case，不替代既有 CPU golden。

原 `check_timer_source.py` 面向计时接入与固定 main 的文件内同步调用文本比较。新专用 helper 增加一份 MTE1_M Set/Wait 的源码文本，原三处调用运行时仍执行相同同步；因此该脚本的旧“文件内序列相等”规则不适用于本性能候选。不要把这个差异解释成设备同步通过，也不要放宽原计时分支的检查；本轮使用上面的候选检查及后续 MSS/确定性证据。计时清单的行号已随候选更新。

## 设备执行顺序

沿 [timer_run.md](timer_run.md) 激活同一 CANN/Python/ATK 环境。以下在 Linux 仓库根目录、B源码下执行；每行或 Bash 数组整体复制，避免续行被终端拆开。

### 1. 构建、安装、确认本轮二进制

```bash
git switch codex/gdn-prepare-k32
git status --short
FLA_NPU_SOC=ascend950 FLA_NPU_OPS= python scripts/build_wheel.py
```

将构建末尾打印的准确 wheel 路径设为 `WHEEL_PATH`，然后：

```bash
python -m pip install --force-reinstall --no-cache-dir --no-deps "$WHEEL_PATH"
python scripts/verify_libcust_opapi_md5.py --python
```

使用全量 wheel 是为了沿用此前 A/B 的交付方式，包含生产和 Profile 符号。必须保存wheel、源码commit、构建日志及 `build/` 校验对应关系；新进程运行测试。共享实现被独立 Prepare 和前向调用链消费，两者都须验证。host/ABI、H/O并未修改，不把历史设备验收当本轮验收。

### 2. Prepare 开发期精度筛查

先测 G2 对齐/尾块，以及 A 不写GM两种路径。按 README 的12条小批执行，`CASE_END` 不含右端：

```bash
set -e
for first in 24 36 216 228 1224 1236 1416 1428; do
  ATK_GM_INIT_MODE=off CASE_START="$first" CASE_END="$((first + 12))" bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=0 -scope=accuracy -soc=ascend950
done
```

这些96条覆盖该 shape的24组合法flag，但不是全支持范围验收。继续回归 G1/G3/G4、V256、B>1、部分pack、varlen/近chunk尾块、FP32/BF16 gate；完整2416条、MSS和确定性按现有 Prepare README执行，生成器/精度标准不改。

需要的逐 Stage 证据：Stage3生产的leaf/N布局，Stage4的FP32 Y，Stage5的FP32 A及BF16 A，最终w/u。离线模型已覆盖Y/A代数，但当前公开 Prepare输出只能直接比较A、w/u等已有输出，不公开Y。设备逐Stage采数若需要内部Y，按精度定位文档使用失败case的调试导出/快照路径；不能用timer窗口代替数值。出现失败先定位，不继续做收益判断。

### 3. 首轮 B 普通性能和计时摘要

精度筛查通过后，用原主入口采 case0，至少三次独立进程；此处使用从基线继承的参数，不需要 Stage3–5诊断分支：

```bash
set -e
for repeat in 1 2 3; do
  result="tests/atk/chunk_gated_delta_rule_fwd/atk_output_k32/B_${repeat}_$(date +%Y%m%d_%H%M%S)"
  python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py --device 0 --cases 0 --output-dir "$result"
done
```

主入口先执行普通 ATK performance_device，再采布局v2计时数据；`--warmup` 默认5控制诊断重放，不重设普通 ATK的采样配置。检查普通报告实际组数与 A 一致（历史A0为20组），保存全部原始报告。每轮至少带出 `run_metadata.json` 和普通 `op_summary.csv`；timer摘要按 [timer_analysis.md](timer_analysis.md) 生成，辅助确认 Prepare窗口变化。

本候选分支的基线采集器没有 `--reference-wheel`、`--prepare-stage35`、`--timer-repeats` 或 `--ordinary-only`。若继续使用以前已验证的新版采集器，应固定同一版本用于A/B，记录采集器commit与实际wheel；不要把Stage35诊断内核或其它优化合并进本候选。

### 4. 新鲜 A/B/A 与接受条件

A使用809基线，B使用本候选；同设备、软件、输入、普通预热/次数、频率和空闲条件，交错或至少补测A1并复测B，建议每组不少于三次独立运行。核对case JSON和实际输入hash；不一致时先固定ATK种子或使用同一冻结输入重测，不作为严格A/B。历史 A0仅参考。

原主入口安装检查默认对照当前工作目录 `build/`，直接把A wheel装进保留B build的目录会产生不匹配。A可以在匹配A源码及build的独立目录编译/安装/运行同一份基线采集脚本；B回到对应B目录。也可复用已经验证过的新版 `--reference-wheel` 采集器来核对每份wheel，但该参数不属于本分支。

先比较各组普通七kernel合计分布，判断节省是否稳定≥5%；同时核对Prepare独立时间及H/O没有抵消。若需要继续定位，再在B以同样命令采msopprof，预期观察Cube/MTE1绝对活跃时间下降；占比可能因总窗变短上升，不要求比例必降。低于5%或处于波动内则本候选不满足用户目标，不因理论MAC下降37.5%而保留。

### 5. 正式采用前的待验项

- 当前代码CANN构建成功及 `.so`/全部`.o`/Python加载证据；实际Prepare key0与受影响前向路由选择证据。
- Prepare完整精度、MSS、确定性；前向原500条双标杆分片和MSS/确定性、Example/ST；不能把96条筛查称全量通过。
- 两个原前向模型case的普通msprof性能，以及Prepare原六个perf case的支持场景回归；case0稳定≥5%是本轮优先目标。
- 逐Stage数值、BF16转换后的A/w/u、普通/Profile输出对照及原精度标准，数值失败按精度定位路由返回设计。
- 完成后记录真实版本/环境/原始报告，用户决定采用候选后再将结果收敛至正式设计/ATK README，删除本开发期记录。
