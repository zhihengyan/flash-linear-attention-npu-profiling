# O raw QK 共享：开发验证记录

日期：2026-10-16；分支 `codex/gdn-o-shared-qk`，直接基于 K32
`d0cb628df7c8c23c0342392dfaf4ec910aba9947`。切分支前，H 结果文档已在原分支提交为
`f020cbb8`，本分支不包含 H right-ready 或其它未采用候选的 kernel。
设计见 [design.md](design.md)，接口冻结见 [api.md](api.md)。

## 性能追踪

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GDN case0，B=1/HK=16/HV=32/T=11274/K=V=128/BT=64，A5、BF16、FP32 g、varlen、FP32 初态及 final_state | K32 基线 | O 重复 raw QK；MAC 96.50610 us、MTE2 331.40270 us，字段可重叠 | 保留 K32，taskGroupSize=4 | 用户已报告 K32 精度全部通过 | 七项 1610.88620 us，O 358.29505 us | 原始数据同级 k32 目录 | 已确认基线，不补测原始 A |
| 同 case0 | 候选，待设备实测 | 同组同 hk 的 raw 相同 | 只消除重复 QK MMAD/L1->L0；保留 raw FP32 L0C 并逐 head Fixpipe | 未执行 | 未执行 | 预期 QK 实际调用 5664->2832；QH/AV/raw 分发次数不变 | 完整块总 MAC -12.5% 不是时延保证；不叠加组大小实验 |
| 原性能 case1 | 待设备实测 | 共享调用和其它原模型覆盖 | 保留冻结 JSON 的完整 shape/dtype/属性 | 未执行 | 未执行 | 未知 | 不用自选小 shape 替代 |

原始数据路径为 `D:/Code/results/atk_output_1016/原始/`；K32 为同级 `k32/`。
H right-ready 失败结果在同级 `h_right_ready/`，不作为本候选或基线的实现。
主指标仍是每次 Cast、Transpose(g)、Transpose(beta)、Prepare、H、O、Transpose(gCumsum)
七行 Task Duration(us) 的和，保留全部普通样本。约 2.7% 只是参考；没有收益保证。
采集器 --warmup=5 是诊断预热，普通 ATK 配置和 20 条样本口径保持。
采用须同卡、同环境、同输入、同软件配置的 K32/B/K32 交错、多个独立进程及原始证据。

## 实现追溯与离线检查

| 设计项 | 位置 / 验证 | 状态 |
| --- | --- | --- |
| 首/末共享 consumer，chunk/group/hk 边界 | Process 的 loadQK/lastQKConsumer | 已实现；实际源码谓词的离线覆盖通过 |
| raw L0C 生命周期与另槽 QH | ProcessStage2Head、qkCSlot | 已接入共享入口；未新增空间，设备时序待验证 |
| 单次真实 raw M_FIX 与最后一次 FIX_M free | 原 event ID、末 consumer 归还 | 静态范围与 credit 模型通过 |
| Q 的最后 QH L1 consumer 释放 | lastQKConsumer 发布 qkEvent free | 已实现；提前归还负例可检出 |
| 全部逐 head raw/QH Fixpipe、AIV ready | AIV 数值计算/owner/flag 数保持 | raw 分发、QH/AV/ready 模型次数保持；Vector 源码 token 对照通过 |
| G1/2/3/4、partial group、跨 HK/跨 task、tail、slot 回绕、错误释放及通知过早复用 | 离线数值/事件模型与源码范围检查 | 8 项 O 共享 UT 通过，不代表硬件验收 |
| Prepare/H、host/ABI、CPU 标杆、JSON、普通/Profile 共用入口 | 固定 K32 diff 检查 | 核对通过；未修改公开和诊断调用层 |
| 设备编译、原精度、MSS、确定性、共享调用及性能 | 以下原环境命令 | 未执行 |

采用阶段03 V2 和 CHUNK_INDEPENDENT V1（来源 a78efa9512dfa64c20b2a58a75e84e88a054d694）；
本轮静态设计审查限定于共享 QK、地址所有权和同步，历史 R01/R06 映射差异见设计，未改写
为已完成 V2 迁移。用户已授权整链接入并跳过逐 Stage 设备验证编码门槛；本地 Windows
没有 CANN/NPU/AscendC 编译器、torch 或 NumPy，只能做标准库离线检查。

本地 Python 3.14.7 已执行：

```bash
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/test_o_shared_qk.py
# 8 tests OK
python -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p test_timer_contract.py
# 14 tests OK
git diff --check
# 通过
```

O 模型从当前 C++ 读取首/末 consumer 和 QH C 槽选择，使用独立输入身份集合检查生命周期，
遍历 G1..4、G/2G/5G 个 head、3/4 head 组、五个 chunk、三个 core 及两个初始 C 槽。
G3 配 group4 等组合是防御性边界模型，非新增 host 支持范围。case0 的 28 核/177 chunk
模型得到 QK=2832、QH=AV=raw_copy=ready=5664；真实设备执行次数尚未采集。
G1 的槽位轨迹与原始 QK/QH/AV 次序相同。数值复用模型使用 BF16 输入/输出和独立 head
gate/H，覆盖有效长度 1/3/8；它只验证复用语义，不模拟 CANN MMAD 累加或设备指数近似。
提前归还 raw/Q 的 credit、QH 写入 raw 槽、无新 MMAD 却重复等 M_FIX、遗漏 producer 的
负例能被检出。另从 QK/QH/AV 源码提取 MTE1_M 的 ID，检查其复用受同一 A/B 槽的
M_MTE1 free 保护；恢复按 C 槽编号的负例能检出去重后输入通知可能过早复用。
模型按串行资源状态核对，不代替真实异步 pipe timeline、DMA 或动态调度。

实际 QK 数学及 layout 语句与 K32 token 对照相同，QH 改变 C 槽选择和 Q free 条件；
QK/QH/AV 的 MTE1_M 通知统一跟随 A 槽号。Stage2 的输入预取/上一组等待、
除此之外的 Cube 方法、Vector、common、Prepare/H、host、
Python 适配、ATK JSON/标杆均与 K32 核对保持。全部 195 个既有 timer scope 清单行号已核对，
未新增打点，O_QK 实际调用数按去重后的工作计。历史 check_timer_source.py 固定比较
插桩前 main 的同步全等，不适用于本轮有意改变生命周期；保留原检查，不加绕过白名单。
完整 ABI 路径合同在 K32 上存在已记录的 direct BSND output 正则失败，本轮未重写该检查，
也不把源码未改等同于完整 ABI 合同验收。

## 设备验证入口

继续使用原 Linux CANN/ATK 环境和既有 wheel 流程。WHEEL_PATH 设为当前 checkout
实际构建出的准确 wheel，DEVICE 统一使用空载卡；安装后重启 Python 进程：

```bash
FLA_NPU_SOC=ascend950 FLA_NPU_OPS= python scripts/build_wheel.py
python -m pip install --force-reinstall --no-cache-dir --no-deps "$WHEEL_PATH"
python scripts/verify_libcust_opapi_md5.py --python
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id="$DEVICE" -scope=accuracy
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh "$DEVICE"
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py --device "$DEVICE" --cases 0
```

也可按阶段04构建 `chunk_fwd_o` 单算子 run 包、--install，并用
`python scripts/verify_libcust_opapi_md5.py --run-package <实际 run 包>` 全量核对；不绕过产物
一致性校验。从 K32 继承的采集器尚不支持 --reference-wheel，不把归档包与另一候选 staging
混比；轮换归档包时使用交接已验证的严格 reference 工具。

先执行原 CPU/六 ACLNN 双标杆，再测性能。完整接受前执行单算子 all（原 accuracy、
performance、determinism、mssanitizer）以及 GDN case0/1、Example/ST 和共享调用回归。
现有单算子冻结 ATK 资产不覆盖所有 GVA/tail/布局组合，须从原 GDN 矩阵逐项确认
G1/2/3/4、tail、dense/varlen、gate dtype、exp/exp2、state_v_first 的 A5 实际选择证据，
缺项按原值域与阈值补齐后再正式接受。test_npu_fwd_o_generalization.py 当前默认调用
BNSD 兼容路径，不能将其通过当作本轮 A5 共享路径覆盖。不能凭用例文件存在宣布覆盖。
进入 kernel 后连续 60 秒不返回按阶段04处理超时并清理。构建、精度、性能或同步失败均
保留真实结果，不把本地模型、普通/Profile 一致或性能报告的 complete 当作精度通过。
阶段05未完成前保留本文，不提前写入正式验收结论。
