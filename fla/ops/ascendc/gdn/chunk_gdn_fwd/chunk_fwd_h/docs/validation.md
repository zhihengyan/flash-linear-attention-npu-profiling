# H 双 head lookahead 开发期验证记录

2026-10-13：候选设计、编码及离线检查完成；设备编译、精度、实际分支选择、确定性、内存与性能待执行。阶段 04 设备闭环和阶段 05 正式验收尚未完成，保留本文件，验收后按流程归档并删除。

分支 codex/gdn-h-dual-head-lookahead，起点 codex/gdn-profiling 的 809bce6887639e831caa856b8ac3a197a6a94294。使用独立 worktree，未合入 Stage6 overlap。用户已授权候选设计与编码，正式采用要求整体至少 5% 收益。

## 方案与性能追踪

地址、数学、生命周期和事件证明见 [design.md 第 10 节](design.md#10-a5-双-head-跨-chunk-输入-lookahead)。唯一主要变量为独立输入的跨 chunk 预取距离：AIC 两份 W 和共享 K 采用两组联合所有权；两个 AIV 各复用原 U/g 双 bank。H/right 的状态依赖、算术和 ready/free 保留。新路径限 A5、FP32 state、scalar-g、完整 GVA2 pair。

参考 CHUNK_DEPENDENT V1，来源 a14961e9dfe8c8264768d67fb447101873da4ecf（docs/agents/reference/04-operator-development/chunk-dependent-development.md）。复用递推、状态驻留、lookahead 和事件闭环方法。本轮保留既有按整个 head 分配 AIV 的映射，未迁移到新 R01 的单 head 两 AIV 横切映射；不宣称完整 V2 映射已实现或设备性能已验收。

case 0：Ascend950PR_957c、torch 2.10.0+cpu；q/k BF16 [1,16,11274,128]，v BF16 [1,32,11274,128]，g FP32 [1,11274,32]，beta BF16 [1,11274,32]；BT64、BNSD、varlen、FP32 initial_state [1,32,128,128]、输出终态、scale=0.0883883461356163。普通 ATK 每 launch 20 条记录；timer warmup=5、capture=1。H 实际处理 Prepare 生成的 W/U/g。

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GDN case 0 | A0 / 基线 | H mean 588.8026 us，占七项时间和 34.47%；双 head 原路径无独立输入 lookahead | 保留递推链与 16 AIC / 32 AIV | 已有普通/诊断十输出一致，不是 CPU 验收 | Prepare 715.08995、H 588.8026、O 359.37605、七项时间和 1708.3914 us | 输入搬运暴露延迟未证实 | 旧 A0 无 reference_wheel，需新 A 复测补齐二进制身份 |
| GDN case 0 | H1 / 待实测 | next W/K 与当前 KV/state、next U/g 与当前 Vector 工作重叠 | 只改独立输入预取距离与所有权 | 51 项离线检查通过，设备待测 | B 未测；整体目标 ≤1622.97183 us；其余项不变时 H ≤503.38303 us | 未测 | 需节省 85.41957 us，即 H 约下降 14.51%；80.5 KiB 输入量不等于可消除延迟 |
| case 1、H 支持场景、共享调用 | H1 / 回归待测 | 非完整 pair、其它 GVA、gk、BF16 state 和其它 SoC 回旧路径 | 保留旧实现 | 待测 | 基线及 B 待补 | 未测 | case 0 不能替代其它场景验收 |

七项口径为普通调用 Cast、Transpose1、Transpose2、Prepare、H、O、Transpose3 的 Task Duration(us) 之和；旧 A0 按顺序推断分组，缺 task ID/时间戳，不能称为 host 端到端墙钟时间。新采集逐轮记录条目、kernel 名、调用归属、分布和环境。普通预热/循环沿用原 ATK 配置并归档；--warmup 只控制后续 timer 诊断。

## 实现与证据追溯

| 对象 | 代码位置 | 离线证据 | 设备待验 |
| --- | --- | --- | --- |
| 相同 selector | chunk_fwd_h_utils.h 的 FwdHUsePairedHeadLookahead、两端 ProcessWorkUnit | 256 组 head/GVA 范围；新增场景按 C=28 推导 eligible unit | 真正 key、blockDim、head range 和执行/编译特化证据，不能只用 shape 推断 |
| W/K 双 bank | arch35/chunk_fwd_h_cube.h 的 PrefetchPairedHeadInputs、ProcessPairedHeadPipeline | 共享 K 一次搬入、两次消费，末消费者后归还 | 首末块、tail、177 chunks、MSS |
| group credit | 同文件；四 helper 的 MANAGE_INPUT_EVENTS=false 仅新路径使用 | 252 组队列调度无遗留 credit；漏末块 free、错 bank、漏 H wait 的反例能失败 | flag 真实语义、跨 unit 复用、无设备死锁 |
| U/g 双 bank及常驻 state | arch35/chunk_fwd_h_vec.h 复用 ProcessSingleHeadPipeline | VF、Stage1/3、初态及单 head pipeline 与 A 源码一致 | h/v_new/final_state CPU 标杆、bank 和跨 unit |
| 旧分支 | 四 W/K helper 默认 MANAGE_INPUT_EVENTS=true | 去除新增条件包装后函数体与 A 一致；未改资源常量/host/接口/timer 布局 | 原模板全量精度、性能、确定性、MSS、A2/A3 |
| 测量身份 | profile_performance.py --reference-wheel | 回移已有 wheel SHA 校验，10 条对应测试 | 安装全部 runtime/OPP 文件与归档 wheel 一致、构建源码记录 |

离线模型抽象 MTE2/MTE1/M/FIX 和两 AIV 的 MTE2/V/MTE3 队列，检查输入代际、递推顺序和 credit；不模拟 MMAD 数值、真实时延、flag 硬件实现、完整 UB 别名或 ROUND 握手，不能替代设备同步验证。跨 unit 有源码与调度覆盖证据，正式检查需执行 244/245 及其其它 seed。

timer ID/布局/打点位置保留，但原 H_WH/H_KV scope 内的部分 W/K 等待移到联合调度处。scope 下降不能代表收益，最终只用普通 msprof 的 H 及七项时间和判定。

## 离线结果

Windows Python 3.14.7，无 torch/ATK/CANN/NPU。

- H 检查 12/12：252 组随机队列调度（chunks=1/2/3/17/177，全部初态/终态组合，各 12 seeds；另有 12 组全局 Cube 活跃但局部短序列无 WH/KV 的场景）、反例、selector、源码对照、冻结用例检查。
- GDN profiling/timer 检查 35/35，其中 wheel reference 10 条。
- GDN ctypes 输出契约及反向签名检查 4/4。
- 48 个新逻辑精度场景 × 3 固定 executor seed，共 144 条；accuracy 总数 344。MSS/确定性追加 8 条，总数 40；performance 保留 44 条。
- 生成器与 JSON 一致。原 accuracy 0–199、performance 0–43、MSS 0–31 的完整内容、seed 和比较标准与 A 相同。
- CANN 编译、NPU 精度/同步、确定性、MSS 和性能均未执行。

~~~bash
python -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/tests/ut -p 'test_*.py' -v
python -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p 'test_*.py' -v
~~~

## 内部设备运行顺序

在 NPU Linux 仓库根目录、同一 Python/ATK/CANN、device 0、ascend950 执行。环境和完整构建见 [GDN timer_run.md](../../chunk_gated_delta_rule_fwd/docs/timer_run.md) 与开发者指南。

常规 kernel 调试按阶段 04 使用单算子 run 包：bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=chunk_fwd_h；安装准确 build_out run 包后执行 python scripts/verify_libcust_opapi_md5.py --run-package <本轮run包>，全部 OK 后开始精度。run 包覆盖旧 wheel 内 OPP 后，旧归档 wheel 不能代表当前 B。

用户选择归档 A/B wheel 实验。A 从指定基线、B 从本分支准确源码分别完整构建：FLA_NPU_SOC=ascend950 FLA_NPU_OPS= python scripts/build_wheel.py。归档构建末尾明确打印的 wheel，记录源码提交及未提交 diff，不按文件排序猜版本。同一份本分支 profiling 入口运行 A/B。

~~~bash
# B_WHEEL 设为准确归档 B wheel 的绝对路径，安装后使用 fresh Python 进程。
python -m pip install --force-reinstall --no-cache-dir --no-deps "$B_WHEEL"

# 首轮最小精度：T1/T64 全初态/终态组合，先验证有初态的 WH。
ATK_GM_INIT_MODE=off CASE_START=232 CASE_END=240 \
  bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -npu_device_id=0 -scope=accuracy -soc=ascend950

# 新 48 个逻辑场景的全部三个 seed（CASE_END 不包含）。
ATK_GM_INIT_MODE=off CASE_START=200 CASE_END=344 \
  bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -npu_device_id=0 -scope=accuracy -soc=ascend950
ATK_GM_INIT_MODE=off CASE_START=32 CASE_END=40 \
  bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -npu_device_id=0 -scope=determinism -soc=ascend950
ATK_GM_INIT_MODE=off CASE_START=32 CASE_END=40 \
  bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -npu_device_id=0 -scope=mssanitizer -soc=ascend950
~~~

off 沿用本轮 A5 静态 tensor 开发检查条件，不能记为完整 GM 初始化/NaN 验收；正式检测按 ATK README 与本环境要求执行。新场景沿用 input/state scale=0.05、gate step scale=0.02、mixed_tolerance_bm；CPU 标杆与阈值未改。新增覆盖的值域区分能力待按精度文档，用同一输入对照正确参考、全零、错误 K head、遗漏更新，保存 CT v0.9.1 指标与图，校准完成再正式验收。

Stage0–3 闭环从 WH/P 开始，再比较 v_new/right、D、h/final_state。整体输出通过不算逐 Stage 中间值通过；首错未知时按精度文档 Stage 中间结果采集/CT 路线临时采数，本候选不加公开 debug ABI。进入 kernel 后单 case 连续 60 s 未返回判开发超时，记录 Stage/key/blockDim，终止、清理设备后定位；不继续性能测试。执行记录/中间采数关闭后再测普通性能。

精度和同步通过后，安装 B wheel 并执行：

~~~bash
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py \
  --device 0 --cases 0 --reference-wheel "$B_WHEEL" \
  --output-dir "$PWD/tests/atk/chunk_gated_delta_rule_fwd/atk_output/h_lookahead_B1"
~~~

此入口内部先跑普通 ATK msprof，再导出同输入 timer，无额外普通采集步骤。输出目录须不存在。随后安装准确 A wheel，使用同一入口采 A1，再交错 B2/A2；若先有新 A1，可 A1/B1/A2。保存 wheel SHA、源码构建记录、installed_artifacts.log、run_metadata.json、输入快照 hash、普通 op_summary 和轮次环境。旧 A0 仅用于预算，不独自证明稳定收益。

比较同条件 H、Prepare/O/辅助项、七项总和的 mean/median/p95 和交错 A 波动。目标整体 ≥5%；其余项不变时 H 需下降约 14.51%。未达标、精度失败或稳定回退则记录实际结果并放弃；不能按 timer stage 好看采用。若要求 host 墙钟收益，另测相同输入墙钟延迟。

## 共享调用与正式验收

实际 GDN prepared route 调用独立 l0op::ChunkFwdH；旧 chunk_gated_delta_rule_fwd_h 和内部融合 H 不受本次 kernel 修改。共享调用还包括 standalone H、GDN backward 重计算、KDA forward V2 与 KDA backward V2 的 H 重计算。KDA prepared gk 回旧分支，仍回归编译与公开入口。

待办：H 全量 344 accuracy / 44 performance / 40 determinism/MSS、新 8 模板实际 key/分支选择；GDN case0/1 性能和 500 条双标杆矩阵；GDN backward 重计算；KDA fwd/bwd V2；必要 Example/ST、A2/A3 原路径编译和精度。GDN 正式精度使用 bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 0；其它算子按各自 ATK README 正式全量验证。未测试项不记 PASS。

结论：候选设计与编码可供内部环境编译和 A/B，设备验收及 ≥5% 性能目标均未验证。
