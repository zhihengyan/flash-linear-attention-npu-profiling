# chunk_gated_delta_rule_fwd ATK 验证

本目录验证公开接口 `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd`。融合算子在 A2/A3
使用私有 `arch22` 实现，在 A5 使用私有 `arch35` 实现；A3 当前只具备注册和编译验收环境。

## 支持范围

- `q/k`：`[B,Hk,T,128]`，BF16/FP16。
- `v`：`[B,Hv,T,V]`，BF16/FP16，`V=128/256`，`Hv % Hk == 0`。
- `g/beta`：`[B,T,Hv]`；`g` 为 FP32，`beta` 与 q/k/v 同 dtype。
- `chunk_size`：64 或 128。
- 支持定长、变长、GVA、可选初始状态和可选最终状态。
- SoC：A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`)。

## 精度标杆

精度使用 ATK 原生 `cv_fused_double_benchmark`：

1. NPU DUT：`chunk_gated_delta_rule_fwd`；
2. NPU benchmark：公开算子链 `chunk_local_cumsum`、`chunk_scaled_dot_kkt`、`solve_tri`、
   `recompute_w_u_fwd`、`chunk_gated_delta_rule_fwd_h`、`chunk_fwd_o`；
3. CPU golden：相同冻结输入上的 FP64 recurrence。

公开 `solve_tri` 来自 main 已合入的 PR 398；融合 kernel 不引用该公开实现，只有双标杆链路调用它。
比较阈值为最大相对误差比例 5、平均相对误差比例 1.5、均方根误差比例 1.5。

## 用例

- `atk_chunk_gated_delta_rule_fwd.json`：既有泛化 500 条冻结矩阵，五种场景各 100 条，
  覆盖 BF16/FP16、MHA/GVA、V128/V256、chunk 64/128、定长/变长及状态组合。
- `scripts/cases/legacy500_adapted.json`：既有 BF16/MHA 历史 500 条回归矩阵，不作为默认入口。
- `atk_chunk_gated_delta_rule_fwd_perf.json`：A5 两条模型 case：
  - 推理：`B=1,Hk=16,Hv=32,T=11274,K=V=128,chunk=64`，变长并输出最终状态；
  - 训练：`B=2,Hk=Hv=32,T=8192,K=V=128,chunk=64`，定长无状态输出。
- `atk_chunk_gated_delta_rule_fwd_mss.json`：从冻结矩阵抽取 6 条精简用例，覆盖 V128/V256、
  chunk 64/128、定长/变长、FP16/BF16 和状态输出。

A5 模型 shape 分别来源于 `推理model.csv` 和 `训练model.csv`，原文件 SHA256 为
`a8f21a5ddc23b824b2b5ccc625d33db95dd9441b33f2b0c0e3e313e72aeaa363` 与
`87e9bb1027c44eaf8cc2f5fc4d24256b22e0ff00c16162b5fb11aaaf24aca8ca`。

## TilingKey 覆盖

| TilingKey | 选择条件 | 普通/边界用例 | SoC | 实际选择证据 |
| --- | --- | --- | --- | --- |
| 1 | `V=128`，未进入 A5 推理模型特化 | MSS 0、2、4 | A2/A3/A5 | 本 PR 硬件门禁补录 |
| 2 | `V=256` | MSS 1、3、5 | A2/A3/A5 | 本 PR 硬件门禁补录 |
| 301 | A5 性能 case 0 的推理模式，初态为 BF16/FP32 | MSS 尚未覆盖 | A5 | 原模型 shape 已实测命中；MSS 待补 |

## 执行

先执行不依赖 NPU/ATK 的 ACLNN ABI 合同，确认公开参数顺序、ctypes 类型和默认路径映射：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py
```

公开 `aclnnChunkGatedDeltaRuleFwd` 保留完整扩展 ABI。当前 Phase6 默认路径使用
`layout=BNSD`、`useExp2=false`、`allowNegEigval=false`、`stateVFirst=false`，且
`aLog/dtBias` 与扩展中间输出为空；`finalStateOutOptional` 是否为空决定是否输出 final state。
尚未实现的扩展组合会显式返回参数错误，不会静默忽略。

正式 500 条双标杆精度使用可恢复分片入口；默认每 25 条启动一个 fresh ATK 进程，避免
六算子 benchmark 长进程状态累积：

```bash
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 0
```

冒烟或单分片可直接使用三节点入口（默认 `-mt 5`）：

```bash
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_double_benchmark.sh 0
```

复跑历史矩阵：

```bash
GDN_ATK_CASE_JSON="$PWD/tests/atk/chunk_gated_delta_rule_fwd/scripts/cases/legacy500_adapted.json" \
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 0
```

性能、确定性和内存检测仍使用仓内统一入口：

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=0 -scope=mssanitizer
```

正式结论必须记录代码 commit、ATK/CANN 版本、SoC、实际加载的 OPP、case JSON 哈希和原始报告。

## 十项返回接口回归

前向新增 q_hat、k_hat、q_rstd、k_rstd，追加在原六项之后。
关闭核内 L2Norm 时 hats 为输入对象别名，rstd 为 None；开启时 rstd 固定为 FP32 `[B,Hk,T]`。

安装包含本次修改的完整 wheel 后运行：

```bash
python torch_custom/fla_npu/test/test_aclnn_ctypes_abi.py
python torch_custom/fla_npu/test/test_aclnn_ctypes_abi.py NormOutputsTest --npu
```

开发回归结果（基于 main `e22e0bf4` 的接口修改）：

- Ascend 950 完整 wheel 构建、安装通过。
- ctypes 单元测试 10 项通过；ACLNN ABI 参数数量与顺序检查通过。
- 16 组设备回归通过：BNSD/BSND/NTD/TND × L2Norm 开关 × 定长/变长，T=65、Hk=2、Hv=4，包含尾块及 GVA。
- 开启归一化时四项输出与独立 prepare 调用逐元素完全一致；关闭时验证 Python 对象别名。
- 原始输入加核内归一化与显式复用 hats 的前向 O/final_state 逐元素完全一致。
- 全部组合的反向均同步完成，输出与梯度有限值检查通过。

以上是接口回归，不是 ATK 全量精度、梯度精度、确定性、内存或性能验收。
本轮未执行这些正式验收项目，不能据此宣称完整 ATK 验收通过。

## 内核 profiling 采集

从仓库根目录运行 `python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py --device 0`，沿用本目录原两个 A5 性能 case、输入生成器和执行器；输入快照、同输入普通/诊断对照与 raw/CSV 导出均在原性能循环外。预检及实际 kernel 报告校验、构建安装步骤、结果阅读见 [timer_run.md](../../../fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/docs/timer_run.md)，本次状态见 [timer_report.md](../../../fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/docs/timer_report.md)。

安装归档 A/B wheel 后，在同一采集分支追加 `--reference-wheel "$WHEEL_PATH"`，按指定 wheel 校验真实安装文件并记录 SHA256，避免与另一版本的 `build/` staging 比较；不用重新编译或跳过产物校验。该参数只校验，须先自行安装对应 wheel。2026-10-16，N32 分支从既有 `87431f0b` 移入此能力，10项无设备合同检查通过；普通 case、输入和采样配置未改。

同日补充 root / `.data/purelib/` / `.data/platlib/` 归档成员到安装路径的映射，修复完整 wheel 被误报缺少 op_api/kernel 的情况；15项归档校验检查通过，仍拒绝内容不符、实际缺失和安装路径冲突。

2026-10-12：计时源码接入后，14 项离线 timer/采集证据合同测试与 3 项既有 GDN 生产 wrapper 契约定向回归通过；原数学标杆、值域、比较阈值和 case JSON 未改动。当前 Windows 环境没有 CANN/NPU/torch，本轮未进行正式 ATK、Example/ST 或真实 profiling 验收，不能沿用本 README 历史设备结论作为本轮验证。设备待验项集中在 timer_report。
