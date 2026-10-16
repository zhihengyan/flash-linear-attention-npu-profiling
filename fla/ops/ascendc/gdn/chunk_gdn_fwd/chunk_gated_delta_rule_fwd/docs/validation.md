# 当前实验：Stage5 A 直接供 L1（2026-10-16）

当前分支 `codex/gdn-stage5-a-direct-l1`，直接基于K32 `d0cb628df7c8c23c0342392dfaf4ec910aba9947`。前轮已放弃的Stage3实验与全部分析文档已提交至其原分支，提交为 `21d9998a`。本分支不继承Stage3/N32/Block32的kernel改动。方案见 [prepare_stage5_a_direct_l1_design.md](prepare_stage5_a_direct_l1_design.md)，以下为本轮待设备验证记录，后文K32结果仅为历史基线。

## 本轮范围与性能追踪

| 项目 | 实现或证据 | 状态 |
| --- | --- | --- |
| K32 A | 最近普通case0七项1610.88620us，Prepare620.28335us；K32用户已报告精度通过 | 业务基线；本候选仍需同条件新鲜对照 |
| Stage5 A直接供L1 B | 完整chunk/outputA=true保留GM A写出，以原DumpAL0cToL1替换GM→L1回读；Stage7统一等FIX_MTE1 | 已接入普通/Profile共享Prepare实现，无实验开关；设备结果待采 |
| 减量与代价 | case0 5632tasks少读44MiB GM地址数据，新增5632次8KiB Fixpipe | 不等于HBM流量或已测净收益；PIPE_FIX竞争、L0C占用和Stage6隐藏程度待测 |
| 参考收益 | 约2.7%=净省43.4939274us，如全来自Prepare则7.011945% | 用户未设3%/2.7%硬门槛；不因理论字节减少宣称达标 |

生产kernel仅修改Prepare arch35主头的Stage5_Dump及ProcessAic中对应的A-ready等待，另有注释更新。K32 MMAD、VCS、Stage1–4/6–7计算、内存分配、Y workspace、host/ABI、入口及H/O保持。tail/outputA=true原FIX_MTE2对保留，outputA=false原单次L1 Fixpipe行为保持；完整chunk不再发布A的MTE2_MTE1事件，消费者相应改等FIX_MTE1。Stage4 Y的MTE2_MTE1及pack起止FIX_MTE1(0) credit保持，防止漏消费或重复credit。两次输出均在FIX_M credit之前，L0C不得提前复用。

采用阶段03 V2、CHUNK_INDEPENDENT V1参考（来源 `a78efa9512dfa64c20b2a58a75e84e88a054d694`）；工程参考为K32自身的直接输出路径。原task映射和Vector→Cube数据流保持，不声称完成R01/R06迁移。按照用户已有授权跳过逐Stage设备验证的编码门槛，完成整链接入；正式精度/MSS/确定性/Example/ST仍须原标准验收。

## 离线检查与本机限制

Windows本机没有CANN/NPU/AscendC编译器，Python3.12.14 / NumPy2.3.5。已执行：

```bash
python -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p 'test_*.py' -v
```

50项全部通过（8.961s）：当前Stage5六项、冻结历史K32四项、timer/摘要25项、归档wheel校验15项。补充混合full/tail连续pack模型后，定向六项复测通过（2.429s）。Stage5检查从真实C++读取分支、Fixpipe字段和事件：

- FP32 L0C NZ16→BF16 L1 NZ16与独立GM ND→NZ编码对照，非对称数据、RNE边界、±0/有限值、valid1..64有效GM范围、canary及错误stride/转置负例。
- full/tail与outputA开关的实际分支trace；tail/a0与K32一致，GM输出参数与offset保持，两次输出先于FIX_M释放。
- G1/2/3/4对应pack3/4、partial pack1..capacity、连续五pack与混合full/tail的A/Y事件token模型；旧A回读wait负例能发现未配对等待。
- 其余源码逐token对照K32，包括全部helper、Stage7/V128/256分支、普通/Profile入口与host；timer scope名称、布局保持，清单行号更新。

以上不是硬件指令仿真或实际调度记录。token模型不能证明动态pipe时序；V128/256的依据是Stage7源码与资源保持，并非设备case已运行。编译、真实BF16舍入、硬件同步、普通/Profile输出对照、正式CPU精度、MSS、确定性与性能均待设备验证；不沿用前轮已放弃实验的设备结果。

采集脚本仅移植前轮已验证的可选归档wheel核对功能和15项测试（从 `21d9998a` 取对应文件）。默认命令不变，默认校验安装产物与本轮build；`--reference-wheel`严格核对安装文件与指定wheel，支持根目录/purelib/platlib布局。输入生成器、case、普通预热/采样配置和精度阈值均未更改。

## 设备验证：沿用原入口

在原Linux CANN/Python/ATK环境的仓库根目录，确认本分支源码，构建后将末尾打印的准确wheel路径设为 `WHEEL_PATH`：

```bash
git switch codex/gdn-stage5-a-direct-l1
git status --short
FLA_NPU_SOC=ascend950 FLA_NPU_OPS= python scripts/build_wheel.py
python -m pip install --force-reinstall --no-cache-dir --no-deps "$WHEEL_PATH"
python scripts/verify_libcust_opapi_md5.py --python
```

保持用户此前wheel交付方式，不变更构建或安装机制。必须保存实际wheel、源码commit及构建/安装校验日志，测试使用新Python进程。该共享实现由独立Prepare和前向普通/Profile调用消费，均需要验证。

先做Prepare精度筛查，可沿用后文96条G2 full/tail与outputA开关筛查；采用前仍需Prepare原2416条及MSS/确定性，覆盖G1/2/3/4、V128/256、varlen、partial pack、有效长度32/64边界和同核连续pack。前向原500条双标杆分片入口：

```bash
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 0
```

精度通过后性能仍用原命令（默认case0/1）：

```bash
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py --device 0
```

空载卡为device1时所有精度/性能命令一致改成1，A/B使用同一张卡。聚焦case0可加`--cases 0`，建议K32/B/K32交错且各至少3轮独立进程；普通20组采样与诊断warmup5的含义保持。每轮带回run_metadata.json、普通op_summary.csv与精度结果；timer/case_0数据或摘要按需补充。固定每次七项Task Duration合计为主指标，附Prepare/H/O变化与输入/case哈希，timer/pipe只辅助定位。

在同一采集checkout轮流安装归档K32和B wheel时，指定当前实际安装的那份：

```bash
python -m pip install --force-reinstall --no-cache-dir --no-deps "$WHEEL_PATH"
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py --device 0 --cases 0 --reference-wheel "$WHEEL_PATH"
```

默认模式要求安装产物与本轮build匹配；reference模式要求与指定wheel的.so/.o/Python/配置一致，任何不匹配都应处理，不能跳过校验。metadata的source_commit是采集checkout，归档wheel自身的源码/构建身份仍需独立保存。完整设备命令及解释见 [timer_run.md](timer_run.md)。

如果额外Fixpipe抵消收益，或原回读已经被Stage6隐藏，停止此单变量候选；精度失败按原定位路由处理，不放宽阈值。阶段05正式完成前保留本开发期记录。

---

# 历史：Prepare K32 开发期验证记录

初始日期：2026-10-14；实测分析更新：2026-10-15。候选分支 `codex/gdn-prepare-k32`，从 `codex/gdn-profiling` / `809bce6887639e831caa856b8ac3a197a6a94294` 创建。方案见 [prepare_k32_design.md](prepare_k32_design.md)。当前完成候选设计、编码、离线检查、三轮B及三轮同日A的普通性能分析；用户报告本轮精度测试全部通过。case0七kernel合计均值相对同日A下降5.424%，达到≥5%均值目标；完整支持范围的ATK正式验收仍待完成。本记录保留至设备验收完成，届时按阶段05收敛结果至对应 ATK README。

## 性能追踪

| 版本 | 主变量 | 证据 | 判断 |
| --- | --- | --- | --- |
| A0，历史基线 | 原 K64 求解，未采用 Stage6/H 实验 | 普通调用20组，七 kernel均值1708.3914us；Prepare715.08995us | 保留历史预算参考；当前以同日A作为主对照 |
| msopprof_r0 | 原始实现 Prepare replay | 708.430969us；AIC Cube77.56251%、MTE1 23.2370% | 支持尝试计算/加载减量，不能替代普通性能基线 |
| 同日A1/A2/A3 | 用户安装原始A0 wheel复测 | 各20组，七项均值1705.45645/1704.92515/1704.02030us，合并1704.80063us；Prepare714.91452us | 新鲜A对照已完成；相对历史A0快0.2102% |
| B1/B2/B3，K32 候选 | 三处 FP32 MMAD 使用有效半区 K | 用户报告精度通过；各20组普通调用，七项均值1613.28590/1612.17790/1611.52665us，合并1612.33015us；Prepare620.64430us | 相对同日A下降5.424%，Prepare下降13.186%；已测最保守轮均值组合下降5.325%，建议保留 |

首批B三轮CSV与各自metadata的140行名称/耗时完全匹配，含历史A0的四轮输入hash及case JSON hash相同。B数据对应优化提交 `dd707730627b29b1862fd335dfbc76e7f0e2666d`。历史对比的收益归因、逐kernel/逐轮分布和数据边界见 [timer_analysis_k32_b1_b3_2026-10-15.md](timer_analysis_k32_b1_b3_2026-10-15.md)。尚未带出构建/安装/频率日志及精度case报告，不据此宣布完整设备验收完成；case1等其它支持场景尚无本候选的新结果。

最新同日复核见 [timer_analysis_k32_same_day_ab_2026-10-15.md](timer_analysis_k32_same_day_ab_2026-10-15.md)：新增三份A CSV也与各自metadata完全匹配，全部七轮输入hash及case JSON hash相同；A/B各60组，净节省92.47048us，5%余量7.23045us。p95降幅4.956%、B逐调用波动较大，当前达标结论针对既定均值指标。新A的采集checkout提交为3b0664fc，本地无该对象，A身份按用户明确说明的原始A0 wheel记录，wheel版本/安装校验信息仍需随正式归档保留。新鲜A复核项已完成，下一步聚焦其它支持场景回归。

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
