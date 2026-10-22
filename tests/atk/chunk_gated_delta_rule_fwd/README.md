# chunk_gated_delta_rule_fwd ATK 验证

本目录验证公开接口 `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd`。融合算子在 A2/A3
使用私有 `arch22` 实现，在 A5 使用私有 `arch35` 实现；A3 当前只具备注册和编译验收环境。

## 支持范围

- `q/k`：`[B,Hk,T,128]`，BF16/FP16。
- `v`：`[B,Hv,T,V]`，BF16/FP16，`V=128/256`，`Hv % Hk == 0`。
- `g/beta`：`[B,T,Hv]`；`g` 为 FP32，`beta` 与 q/k/v 同 dtype。
- ATK 的 `g` 取值范围固定为 `[-1, 0]`，表示不为正的逐 token 门控增量；三份正式 JSON、
  历史矩阵 JSON、YAML 和生成器保持一致。输入整理不会对正值做截断或改号。
- `chunk_size`：64 或 128。
- 支持定长、变长、GVA、可选初始状态和可选最终状态。
- SoC：A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`)。

## 精度标杆

精度使用 ATK `mixed_tolerance_bm`，在同一份同步输入上直接比较两路 NPU 输出：

1. NPU DUT：`chunk_gated_delta_rule_fwd`；
2. NPU benchmark：公开算子链 `chunk_local_cumsum`、`chunk_scaled_dot_kkt`、`solve_tri`、
   `recompute_w_u_fwd`、`chunk_gated_delta_rule_fwd_h`、`chunk_fwd_o`。

公开 `solve_tri` 来自 main 已合入的 PR 398；融合 kernel 不引用该公开实现，只有六算子对照路径调用它。
CPU FP64 recurrence 保留为独立排障参考，不参与这两个脚本入口的比较。阈值使用当前
ATK 中显式注册的 `mixed_tolerance_bm` 默认配置，不沿用旧的误差比阈值；结果不能等同于
原三路双标杆验收。正式精度结论仍需按仓内精度规则完成新比较标准的值域校准。

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

正式 500 条用例使用可恢复分片入口；默认每 25 条启动一个 fresh ATK 进程，避免
六算子 benchmark 长进程状态累积：

```bash
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 0
```

6 条冒烟直接指定 MSS JSON，使用两个 NPU 节点（默认 `-mt 5`）：

```bash
GDN_ATK_CASE_JSON="$PWD/tests/atk/chunk_gated_delta_rule_fwd/atk_chunk_gated_delta_rule_fwd_mss.json" \
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_double_benchmark.sh 0
```

`DN_ATK_CASE_JSON` 兼容为 `GDN_ATK_CASE_JSON` 的别名；两者同时设置时必须一致。
保留 `run_double_benchmark.sh` 文件名以兼容既有命令，实际执行的是双路 NPU 的
`mixed_tolerance_bm` 精度比较。角色合同每 case 只要求 `dut` 和 `benchmark` 各一次，
6 条冒烟应有 12 条角色记录。数值精度、执行失败和用例缺失仍会使脚本失败。

切换比较标准后请使用新的结果目录；旧标准的分片 `summary.json` 不允许续跑复用。
`g` 值域从 `[-1, 1]` 修正为 `[-1, 0]` 后，旧输入、输出和分片结果同样不能复用；
请从新目录重新执行，不能将原值域的结果作为本轮验收依据。
报告按列名解析，兼容有无“错误信息匹配用例个数”的 summary；没有逐 case 精度结果时
不能判定通过。分片汇总保留失败状态，不会将失败分片汇总成通过。

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

不依赖 ATK/NPU 的脚本合同回归：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd/scripts/test_benchmark_contract.py
# 额外用模拟 ATK 走完 6 条冒烟、默认 20 个分片共 500 条及断点复用
GDN_ATK_TEST_BASH="$(command -v bash)" \
python3 tests/atk/chunk_gated_delta_rule_fwd/scripts/test_benchmark_contract.py
```

本地合同回归与模拟 ATK 入口只验证配置、角色和报告处理；实际 NPU 精度、混合容差值域
校准、性能、确定性及内存检查均需在目标 ATK/CANN 环境完成后归档。

2026-10-22 比较器迁移的本地回归：10 项通过，包含 6 条冒烟、默认 20 个分片的 500 条用例、断点复用，
以及角色缺失和精度失败的拒绝通过检查。目标环境 ATK/NPU 实测及新标准的值域校准待完成，
尚未形成正式精度验收结论。

后续 `g` 值域修正已覆盖全部 4 份 JSON 的 1008 条用例、YAML 和生成器；静态配置与
模拟 ATK 入口共 11 项回归通过。使用 ATK 接口替身执行生成器的全部 500 条配置，确认
`g` 始终取 `[-1, 0]`；目标硬件上的值域校准和精度验收仍待完成。

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
