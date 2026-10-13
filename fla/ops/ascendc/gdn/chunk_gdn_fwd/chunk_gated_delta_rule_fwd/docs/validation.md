# Stage6 优化开发期验证记录

当前状态：实验候选编码完成，离线检查已运行，阶段 05 设备验收未完成。本文件保留到正式验收结束；只有通过后才把有效结果归档并删除，不以离线检查代替设备结论。

## 任务、基线与参考

- 编码分支：codex/gdn-stage6-overlap；父分支 codex/gdn-profiling，起点 809bce6887639e831caa856b8ac3a197a6a94294，工作树创建时干净。
- 用户授权：从当前分支新建分支并实施已经讨论的 Prepare Stage6 候选。只修改 K 输入搬运下发位置，实验条件、公式、空间与同步证明见 [design.md](design.md)。
- 性能输入：用户 case 0 的 brief、三个汇总 CSV、profiling.log 和 run_metadata.json；基线数值、输入及用例文件哈希见 [case 0 分析](timer_analysis_case0_2026-10-13.md)。数据是历史优化前证据，不能当作本分支设备验证。
- 参考：仓库 CHUNK_DEPENDENT V1，来源 a14961e9dfe8c8264768d67fb447101873da4ecf，仅复用原跨 Stage ready/free 和状态依赖分析方法；本轮只改 chunk 内 Prepare 输入准备，不改 H 状态递推。
- 受影响入口：独立 ChunkGatedDeltaRuleFwdPrepare Arch35；公开 ChunkGatedDeltaRuleFwd 的 A5 Prepare/H/O 路由。其它 SoC Prepare Arch22 和融合 Phase6 不改。输入/标杆见独立 Prepare README、其 scripts/cpu_golden.py，以及 GDN 的 gdn_reference.py。

## 性能追踪

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 用户 case 0：B1/Hk16/Hv32/T11274/K128/V128/BT64，varlen，初态/终态 | 基线 | Prepare 为最大单项，WU_PREPARE 采样 median=3074.5 cycles；采样非完整硬件耗时 | 原 K Copy 在 Vb 上传下发后 | 用户报告普通/诊断十项一致；正式 CPU 精度未提供 | 普通 Prepare/H/O mean=715.090/588.803/359.376 us；七项 mean=1708.391 us，20 条/项、单批 | 未实施优化 | 采集 source_commit=29422b8f；本地与内部 case JSON 哈希不同，设备 A/B 使用原冻结用例及输入 |
| 同一用户 case 0 | 待实测 | 提前 K Copy 可能隐藏于 V 的处理期间；可隐藏时长未知 | V128/G2/Hv%4=0 时，K Copy 移至 V-ready Set 与 Wait 之间；其它路径旧顺序 | 待设备验证 | 待 A→B→A 重复 msprof | 待硬件 overlap/窗口核对 | 不能预填收益；相同输入、环境、预热、次数、已加载版本；收益超波动才采用 |
| 原模型 case 1 及非候选 Prepare 分支 | 待实测 | 新条件判断有标量开销，原序列应继续执行 | 条件不满足时 K Copy 留在旧位置 | 待设备验证 | 待回归 | 待查回退 | 不以候选未启用推断零性能影响 |

## 实现追溯与本机结果

| 设计项 | 代码位置 / 验证 | 当前结果 |
| --- | --- | --- |
| 提前 K Copy 条件与唯一一次搬运 | Prepare Arch35 Stage6_AivOne 的 prefetchK 分支 | 已编码；还原新增条件/提前 Copy 后与父分支 Stage6 文本一致，仅本函数改变；每个分支仅读一次 K，设备待验 |
| K/V 及系数地址不重叠、旧 Stage3 别名释放 | common.h 固定区间、InitOnChip、ProcessAiv；design.md 地址表 | 已按当前源码核对；设备 race/MSS 未运行 |
| 原 Set/Wait/通知顺序和计数 | check_timer_source.py，运行时 BASELINE 指定父分支起点 809bce68 | 通过；修改文件原同步调用序列一致，生产声明/签名一致；不证明硬件同步 |
| timer 点位、枚举和源码导航 | 195 个 scope，28 个动态类型、6 个固定项；timer_flow.md | 更新 9 处后续源码行号，清单与源码对应检查通过 |
| GVA2 pack 的 source K 所有权 | 基于现有 PackWorkId/OwnerTaskIdx 定义，对 Hv=4..256（步长4）、连续3 chunk 枚举 | 24960 个 task 映射均由同一 AIV 拥有 source-K，且不跨 chunk；仅为静态映射检查，实际任务记录待设备补充 |
| decoder/汇总与采集证据合同 | unittest discover，本目录 tests/ut 的 test_*.py | 25 项通过，保护原计时布局和诊断解释；不能替代 CANN 精度 |
| GDN 生产 wrapper 定向合同 | test_aclnn_ctypes_abi.py 中三项 GDN 测试 | 3 项通过：训练/推理返回、norm 输出/别名、非法保留参数 |
| 构建/安装与真实加载 | 当前 Windows 未发现 CANN 编译器、Docker、bash、npu-smi；Python 无 torch/numpy | 无法在本机执行设备验证，不记通过 |

本轮还有两个已确认的既有校验限制，未改其脚本或扩展 Stage6 任务范围：

- check_timer_source.py 默认固定 main `90176b44` 时，会对父分支已有的两个新 timer 转发头调用 git show；这些头不在旧 main，因此脚本退出。已在父分支文件清单复现。本轮将模块 BASELINE 显式设为 `809bce6887639e831caa856b8ac3a197a6a94294` 后检查通过，未将默认命令记为通过。
- `python tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py` 的 direct BSND output 文本模式检查失败。将它所读取的所有源码换为父分支版本后，同样失败信息复现；不是本轮 Stage6 改动引入。参数合同解析与本轮三个生产 wrapper 定向测试通过，但未将完整 ABI 脚本记为通过。

离线复现命令（仓库根目录）：

```bash
python -c "import sys; sys.path.insert(0, 'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut'); import check_timer_source as c; c.BASELINE='809bce6887639e831caa856b8ac3a197a6a94294'; c.main()"
python -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p 'test_*.py' -v
python torch_custom/fla_npu/test/test_aclnn_ctypes_abi.py AclnnCtypesAbiTest.test_gdn_training_and_inference_output_contract AclnnCtypesAbiTest.test_gdn_norm_outputs_and_input_aliases AclnnCtypesAbiTest.test_gdn_reserved_gate_arguments_rejected_before_launch -v
```

## 设备验证计划

先按 [timer_run.md](timer_run.md) 的已存在环境/构建/准确 wheel 安装流程切换父分支 A 与候选分支 B，确认实际加载版本。冻结内部 performance_cases.json 和输入快照，沿用原 ATK 环境及性能配置；不重生成随机输入替代本次快照。正常精度/性能调用关闭额外执行记录，不传 timer。

1. **Stage6 到整算子精度。** 独立 Prepare 对原 CPU golden 比较 W/U（以及 A、gate、hats/rstd 等既有输出），再跑 GDN 原双标杆入口与 Example/ST。数值序列未变，仍需在设备验证，不以源码分析替代。
2. **候选与回退路径。** 使用原 Prepare 精简 162 条或完整 2416 条冻结矩阵：G2/V128/Hv%4=0 的对齐、尾块、varlen、L2Norm 开关、exp2 开关、beta/gate dtype、outputA 两态；以及 G1/3/4、V256、Hv 不满 pack。原精度基线/容差不修改。精简集只能作为前置验证，正式验收按 tests/atk/README.md。
3. **复用与同步。** 覆盖同核至少连续多个 pack、db0/db1 切换、下一 pack 复用、尾任务与空闲核；按阶段 04 用独立测试记录确认实际任务/slot/ready/free 过程，记录不得增加同步。正常性能测试关闭该记录。本轮未新增或采集设备执行记录。
4. **内存与确定性。** 独立 Prepare 与公开 GDN 的既有 MSS、determinism 用例；无越界、未初始化读取或 race。新增局部分支需真实命中证据，可用测试记录及编译后指令确认，不能仅凭模型 shape 推断。
5. **普通性能 A/B。** case 0 比较每次公开调用七项 kernel 的 Task Duration(us) 和，保留 Prepare/H/O/转换项各自分布；A→B→A 交替重复若干批，结果可重复且超过噪声才采用。case 1 与独立 Prepare 原性能集检查回退。诊断 WU_PREPARE 仅用于解释，确认硬件 MTE2/V/MTE3 重叠时使用实际 pipe profiling，不把 issue scope 当利用率。

原入口示例（仓库根目录，先完成准确构建安装）：

```bash
# Prepare 精度 / MSS / 确定性
ATK_GM_INIT_MODE=off bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=0 -scope=accuracy -soc=ascend950
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=0 -scope=mssanitizer -soc=ascend950
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=0 -scope=determinism -soc=ascend950
# GDN 正式精度和模型性能/计时原入口
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 0
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py --device 0 --cases 0
```

最后一条是原采集入口，会运行普通性能并导出新的输入快照/诊断数据；正式冻结输入 A/B 须另外确认内部 ATK 复用的是本次已保存输入和用例，不能仅凭命令相同认为输入值相同。具体安装路径、CPU 对比、执行记录及 A/B 结果在内部设备完成后补到本记录。当前只交付实验代码与验证计划，不填写正式验收或收益通过。
