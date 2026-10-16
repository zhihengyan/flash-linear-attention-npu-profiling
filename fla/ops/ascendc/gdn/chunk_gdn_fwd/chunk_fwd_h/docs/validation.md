# H right 优先交付：开发验证记录

日期：2026-10-16。分支 `codex/gdn-h-right-ready`，直接基于 K32
`d0cb628df7c8c23c0342392dfaf4ec910aba9947`。原 Stage5 分支的四份改动已先提交为
`c72e3632`，本分支不含 Stage5、N32、Block32、Gather 或历史双 head lookahead kernel。
方案见 [design.md 第 10 节](design.md#10-h-right-优先交付候选2026-10-16)。

## 范围与性能追踪

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GDN case0：B=1,HK=16,HV=32,T=11274,K=V=128,BT=64；A5/BF16、g FP32、varlen、FP32 初态及 final_state | 基线 K32 | H right/H 反复经 GM 交接；H MTE2 451.47465 us，MAC 148.78900 us，仅作定位线索 | K32；保持 16 AIC、每 AIV 一个 head | 用户已报告 K32 精度全部通过 | 10/16 七项 1610.88620 us；H 586.94120 us | 尚无该交接的完整硬件 timeline | 沿用已确认基线，不补测原始 A |
| 同 case0 | 待实测 B | Stage1 原 V_new MTE3 排在 right/RIGHT_READY 之前 | 仅改为 right -> RIGHT_READY -> V_new；其余 free/数学/资源保持 | 未执行 | 未执行；约 2.7% 为参考、非硬门槛 | 需确认 right 消费是否提前且未转移等量阻塞到 H 写回 | 源码明确存在串行边，暴露时长与净收益未知 |
| 原性能 case1，完整 shape/dtype/属性沿用冻结 JSON | 待实测 | 共享调用回归 | 不替换或缩小模型 shape | 未执行 | 未执行 | 未知 | 没有本轮结果，不沿用其它版本结论 |

普通主指标：每次 Cast、Transpose(g)、Transpose(beta)、Prepare、H、O、
Transpose(gCumsum) 七行 `Task Duration(us)` 的和，保留全部 20 个同进程样本；
H 单项用于归因。原 ATK 预热/采样配置、输入生成、case、CPU golden、阈值保持。
采集器 `--warmup=5` 仅为诊断预热，不代表普通性能预热次数。K32/B/K32 要在同卡、同环境
交错各跑多个独立进程，保存每次原始 CSV、run_metadata.json、安装产物身份与设备负载信息。
不得把 pipe 字段或 NoBarrier scope sum 当净收益，也不承诺 2.7%。

## 设计评审与实现追溯

采用阶段03 `V2`，按本轮受影响范围审查地址/生命周期、ready 计数、零初态、tail 和终止路径；
沿用历史 head-owner 映射，不声称完成 R01/R20 结构迁移。阶段04参考角色
`CHUNK_DEPENDENT V1`，提炼来源 `a14961e9dfe8c8264768d67fb447101873da4ecf`；
采用其顺序递推、真实生产 pipe ready、最后消费者 free 和最终收口方法，不复制其窗口布局。
具体工程依据为 K32 的 ChunkFwdH 自身。

| 设计项 | 代码位置 / 验证方法 | 状态 |
| --- | --- | --- |
| FP32 scalar-g、activeHeadCount<=2 选择 | arch35/chunk_fwd_h_vec.h 的 UseEarlyRightReady | 已实现，适用域离线检查通过 |
| right MTE3 完成后通知，随后写 V_new | ConsumeStage1Head；WorkFree/IoFree 不前移 | 已接入，原 VF 与状态/释放代码 token 对照通过 |
| 单 head 和普通多 head 调度不重复 ready | ProcessSingleHeadPipeline、RunStage1 | 已实现，所测路径每代恰好一次 ready |
| 3/4 head 聚合通知及共享 UB WAR | 保持原聚合分支；离线依赖图负例检查 | 离线通过；不是硬件调度证明 |
| 数值与公开/诊断入口 | VF/Cube、host/ABI、golden/JSON 不变；共享 ChunkFwdH 实现 | 源码范围核对通过；设备输出尚未比较 |
| 静态检查与同步模型 | 标准库离线 UT；不能替代编译或硬件调度证据 | 7 项 H 交接 UT、14 项 timer UT 通过 |
| 真实 CANN 编译、精度、MSS、确定性、Example/ST、普通性能 | 以下设备命令及原 ATK README | 未执行 |

新候选的 Stage1 VF 和 Stage3 VF 均保持原数学及舍入顺序，未改变数值标准，
不重新校准输入。每个 head/chunk 的 right 通知恰好一次，terminal/no-final 为零；
WorkFree 仍保护公开 V_new 源 bank，IoFree 和 H0 的 MTE3->V fence 保持。
只允许 D 覆盖已完成 MTE3 消费的 right 槽，不允许四 head 路径提前发布。

## 本地与设备验证

当前为 Windows 源码环境，Python 3.14.7；没有 CANN/NPU/AscendC 编译器、torch 或 NumPy。
本地只做标准库源码/依赖模型验证，不能登记设备编译、精度或性能通过。
按用户本次及交接授权，直接接入普通/Profile 共用实现，不保留私有 probe 前置门槛。

本机执行结果：

```bash
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/test_h_right_handoff.py
# 7 tests OK
python -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p test_timer_contract.py
# 14 tests OK
git diff --check
# 通过
```

H 检查从实际写回分支提取顺序，覆盖 FP32/BF16、g/gk、activeHeadCount=1..4、
WRITE_RIGHT=true/false；以 1/2/5/177 chunks 检查终止通知数。独立偏序模型验证 right
最后一次 MTE3 消费先于 D 覆盖、V_new MTE3 完成先于下一次 input bank 写入；提前发布
ready、提前 WorkFree、四 head 提前 ready 的负例均能检出缺失依赖。该模型不模拟全部
硬件 pipe、DMA 分段、CANN 排程或算术，不能用它推断真实流水重叠/收益。

全部 195 个现有 timer scope 的清单行号已逐项核对，本轮未新增打点。历史
check_timer_source.py 的同步全等检查固定比较 main `90176b44`，不适用于有意改变
Stage1 通知位置的本轮候选；保留其原检查，不用白名单绕过。

额外执行 `python tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py` 时，完整脚本在
`direct BSND output` 路径正则处失败。通过替换 Path.read_text 为固定 K32 git 对象读取，
对同一脚本重放得到相同失败；失败前读入的 header、实现、ctypes 三文件均与 K32 相同。
公开 ACLNN 参数表和 ctypes 类型表独立检查通过。这是继承的路径检查失败，本轮不修改
它，也不声称完整 ABI 合同通过。Prepare/O、H host、全部 ATK 用例/标杆和 Python 适配
与 K32 的 `git diff --quiet` 核对通过。

设备上沿用原 Linux CANN/ATK 环境及用户既有 wheel 验证方式；以下 `WHEEL_PATH` 必须是
该 checkout 实际构建出的准确 wheel，`DEVICE` 统一选择空载卡，不能混用不同卡的 A/B：

```bash
FLA_NPU_SOC=ascend950 FLA_NPU_OPS= python scripts/build_wheel.py
python -m pip install --force-reinstall --no-cache-dir --no-deps "$WHEEL_PATH"
python scripts/verify_libcust_opapi_md5.py --python
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -npu_device_id="$DEVICE" -scope=accuracy
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh "$DEVICE"
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py --device "$DEVICE" --cases 0
```

每轮重启 Python 进程。当前从 K32 继承的采集器只支持 build-staging 校验，不传
`--reference-wheel`；如轮换归档 wheel，须使用交接已验证的严格 reference 校验工具，不能
把安装 K32 对比候选 staging 的 mismatch 绕过。仅修改 H kernel 也可按阶段04单算子 run
包路线构建 `chunk_fwd_h` 并 `--install`；使用
`python scripts/verify_libcust_opapi_md5.py --run-package <实际 run 包>` 完整核对后再测。

正式采用前，按 tests/atk/chunk_fwd_h/README.md 完整执行 all（200 accuracy、44 performance、
32-key determinism/MSS），并完成 GDN 原双标杆矩阵、case0/1、Example/ST 及共享调用回归。
单用例进入 NPU kernel 后 60 秒不返回按阶段04终止并清理，不跨失败继续采集。
基线精度报告不证明候选通过，普通/Profile 一致也不替代 CPU 标杆验收。
阶段05未完成前保留本文，不将待验内容写成正式验收结果。
