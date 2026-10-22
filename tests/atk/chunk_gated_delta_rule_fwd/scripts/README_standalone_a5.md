# A5 GDN 独立 Python 测试

本目录新增两个入口，运行时不导入或调用 ATK，不需要安装 ATK：

| 脚本 | 用途 | 默认用例 |
| --- | --- | --- |
| `accuracy_a5.py` | 融合 GDN 与 CPU FP64 golden 逐输出比较 | `../atk_chunk_gated_delta_rule_fwd.json`，500 条 |
| `perf_a5_msopprof.py` | 按 case 启动 CANN msopprof，汇总设备任务耗时 | `../atk_chunk_gated_delta_rule_fwd_perf.json`，2 条 |

脚本运行于 Linux 昇腾 A5（Ascend 950）主机。下文命令均从仓库根目录执行。
这两个文件按本仓测试脚本存放规则放在算子的 `scripts/` 中；目录名称中的 `atk` 不代表依赖
ATK 运行时。精度标杆直接复用上级目录仅依赖 PyTorch 的 `gdn_reference.py`，两个脚本也会互相
复用输入准备函数，因此请保留仓库目录结构。不会导入 `executor_*.py` 或 `gen_*.py`。

## 1. 准备 A5 环境

1. 激活安装了匹配版本 `torch`、`torch_npu` 和 `fla_npu` 的 Python 环境，加载 CANN：

   ```bash
   source /path/to/your/venv/bin/activate
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   npu-smi info
   python scripts/check_npu_env.py
   python -c "import torch, torch_npu, fla_npu; print(torch.__version__, torch_npu.__version__, fla_npu.__file__)"
   ```

   路径替换为主机实际安装位置。`torch`/`torch_npu` 必须配对，其他依赖按
   [开发者指南](../../../../docs/开发者指南.md) 准备。

2. 安装当前代码对应的 **A5 wheel**。已有匹配的完整 wheel 时可直接使用；需要从源码构建时：

   ```bash
   FLA_NPU_SOC=ascend950 python scripts/build_wheel.py
   python -m pip install --force-reinstall --no-cache-dir --no-deps dist/<本次构建的准确wheel文件名>.whl
   ```

   如果环境使用外置 custom OPP，另外加载该 OPP 的环境：

   ```bash
   source /path/to/vendors/fla_npu_transformer/bin/set_env.bash
   ```

   使用 wheel 内嵌 OPP 时依照 wheel 的 runtime 加载机制即可。确认当前 Python 加载的 wheel
   与目标 OPP 来自同一次构建；记录实际 OPP 路径和 CANN 版本。脚本要求当前十项返回接口，
   不兼容旧六项返回的 wheel。

3. 性能测试还需确认 msopprof：

   ```bash
   command -v msopprof
   msopprof --version
   msopprof --help
   ```

   若未加入 PATH，后续用 `--msopprof /实际CANN路径/tools/msopprof/bin/msopprof` 指定。
   本脚本使用 `--application`、`--kernel-name`、`--launch-count`、`--warm-up`、
   `--aic-metrics` 和 `--kill`，应以目标 CANN 的帮助为准。
   参数及 CSV 含义参考 [Ascend msopprof 官方指南](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md)
   和 [性能数据说明](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_performance_data.md)。

`--device` 表示当前进程看到的逻辑设备号，默认 0；设置可见设备环境变量后，应使用映射后的
逻辑编号。脚本检查设备名称包含 `950`。

## 2. 精度测试

### 输入与比较范围

- 直接读取 JSON 的 q/k/v/g/beta shape、dtype、范围、scale、chunk_size 和序列元数据。
  QKV 使用 BNSD，g/beta 使用 `[B,T,Hv]`。随机数在 CPU FP32 中均匀生成后转换到指定 dtype。
  这会保留 JSON 值域，但不保证与 ATK 内部随机数生成器逐元素一致。
- q/k/v 范围为 `[-0.05,0.05]`；g/beta 范围为 `[-1,1]`。
  不额外归一化 q/k，也不对 g/beta 做 sigmoid、log、截断或值域收缩。
- 沿用 CPU 标杆的确定性初态：`state_initial_final`、`state_initial_only` 使用非零 FP32 初态；
  `state_zero_final` 使用零 FP32 初态；`dense`、`varlen` 不传初态。
- 按原 executor 设置输出最终状态的场景；生成 canonical chunk indices；变长序列各自重置状态。
- 比较 `o`、`g_cumsum`、`A` 和请求时的 `final_state`。CPU 在 FP64 内计算，随后沿用既有
  标杆的公开输出 dtype；将 CPU 的 o 从 BNSD 转成 DUT 的 BSND。
  A 只比较每个真实 chunk 的有效方阵，无定义的尾列不参与指标和判定。
- 显式设置 `layout=BNSD`、`use_exp2=False`、核内 L2Norm/门控/beta sigmoid 关闭、
  `disable_recompute=True`、`state_v_first=False`，与本次参考的 ATK executor 默认路径一致。
  同时检查关闭的可选输出和 q_hat/k_hat 别名契约。

### 选择容差

原 JSON 的 `5 / 1.5 / 1.5` 来自三路双标杆判据，本脚本采用直接 CPU 比较：

```text
对每个有效元素：abs(NPU - golden) <= atol + rtol * abs(golden)
所有请求输出均通过，且没有 NaN/Inf，case 才 PASS。
```

**两种判据不等价，不能将双标杆比值填入 rtol/atol。** 仓内这组 JSON 没有提供可直接沿用的
混合容差，因此脚本要求显式传入 `--rtol`、`--atol`，不内置未经校准的通过阈值。
正式使用应按已经确认的混合容差执行；需要校准时遵循
[精度对比与定位](../../../../docs/agents/reference/精度对比与定位.md)，确认正确输出通过，
且全零、错误 scale、错误 head/chunk 映射等代表性错误失败。

下面 `0.02 / 1e-6` **仅用于演示命令和首次观察误差，未在 A5 上校准，不是验收标准**。
请将其替换为项目已经确认的值；BF16/FP16 或不同输出需要不同标准时，应分别执行或据报告
逐输出复核，不能因观察到失败而临时放宽阈值。

### 执行步骤

1. 列出用例，不需要 NPU，也不需要 PyTorch：

   ```bash
   python tests/atk/chunk_gated_delta_rule_fwd/scripts/accuracy_a5.py --list-cases
   ```

2. 先单条验证环境与返回接口，再覆盖五种场景：

   ```bash
   python tests/atk/chunk_gated_delta_rule_fwd/scripts/accuracy_a5.py \
     --device 0 --case-ids 0 --rtol 0.02 --atol 1e-6 \
     --output-dir output/gdn_accuracy_smoke

   python tests/atk/chunk_gated_delta_rule_fwd/scripts/accuracy_a5.py \
     --device 0 --case-ids 0,100,200,300,400 --rtol 0.02 --atol 1e-6 \
     --output-dir output/gdn_accuracy_scenarios
   ```

3. 用确认后的容差执行全量；以下环境变量应先设置为确认值。每个 case 跑 3 个固定种子：

   ```bash
   python tests/atk/chunk_gated_delta_rule_fwd/scripts/accuracy_a5.py \
     --device 0 --rtol "${GDN_RTOL:?请先设置已确认的GDN_RTOL}" \
     --atol "${GDN_ATOL:?请先设置已确认的GDN_ATOL}" \
     --seed-base 20260827 --seed-count 3 --cpu-threads 1 \
     --output-dir output/gdn_accuracy_full
   ```

   默认每条跑 1 个种子。JSON 的 `default_seed` 非空时优先使用，否则首个种子为
   `seed_base + case_id`；`seed-count` 指定后续连续种子数。可用 `--dtype bf16` 或
   `--dtype fp16` 分别执行。500 条 × 3 种子的 CPU FP64 recurrence 会花较长时间。

4. 查看 `accuracy.json`：每个输出都有有效元素数、失败元素数、最大/平均绝对误差、RMSE、
   relative L2、实际容差及 PASS/FAIL。默认失败保存
   `case_<id>_seed_<seed>.pt`，包含冻结输入、初态、DUT 和 golden；`--save all` 保存全部，
   `--save none` 关闭。用例不通过返回码为 1，全部通过为 0。运行异常会保留错误报告并停止，
   设备错误后请在新进程重跑。

   ```bash
   python -c "import torch; d=torch.load('output/gdn_accuracy_smoke/case_0_seed_20260827.pt', map_location='cpu', weights_only=False); print(d.keys())"
   ```

   上述文件只有在该 case 失败或使用 `--save all` 时存在。重新运行请指定新的 output-dir，
   避免旧 dump 与本轮报告混淆。输出比较异常前尚未取得的数据无法保存；错误信息见报告。

## 3. msopprof 性能测试

### 模型用例

| id | 场景 | B / Hk / Hv / T / K / V | dtype / chunk | 序列与状态 |
| --- | --- | --- | --- | --- |
| 0 | 推理前向 | 1 / 16 / 32 / 11274 / 128 / 128 | BF16 / 64 | `cu_seqlens=[0,11274]`，非零 FP32 初态，输出最终状态 |
| 1 | 训练前向 | 2 / 32 / 32 / 8192 / 128 / 128 | BF16 / 64 | 定长，不传初态，不输出最终状态 |

两条均沿用 executor 的 `disable_recompute=True`，会导出 g_cumsum/A；这里“推理”指 JSON 的
模型用例名称，不会擅自关闭中间输出。此处只测前向，不执行反向或优化器。

### 执行步骤

1. 检查 shape 和即将调用的命令；这两条命令仅使用 Python 标准库：

   ```bash
   python tests/atk/chunk_gated_delta_rule_fwd/scripts/perf_a5_msopprof.py --list-cases
   python tests/atk/chunk_gated_delta_rule_fwd/scripts/perf_a5_msopprof.py --dry-run
   ```

2. 先采集单条确认工具、过滤器和 CSV，再执行两条：

   ```bash
   python tests/atk/chunk_gated_delta_rule_fwd/scripts/perf_a5_msopprof.py \
     --device 0 --case-ids 0 --iterations 1 --warm-up 10 \
     --output-dir output/gdn_perf_smoke

   python tests/atk/chunk_gated_delta_rule_fwd/scripts/perf_a5_msopprof.py \
     --device 0 --iterations 10 --warm-up 10 --aic-metrics BasicInfo \
     --output-dir output/gdn_perf
   ```

   每个 case 启动一个新的 msopprof/worker 进程，输入只生成、搬到 NPU 一次。worker 下发
   `iterations` 次融合 GDN，每次同步；msopprof 内部负责预热，Python 不另行下发预热算子。
   通过 kernel 名过滤辅助操作，默认前缀为
   `chunk_gated_delta_rule_fwd*|ChunkGatedDeltaRuleFwd*`。预热次数指 msopprof 参数，
   不混入最终的样本数。默认 10 次采样不代表 10 次独立进程或实际应用缓存状态。

3. 需要分析瓶颈时，另跑 `Default` 指标，保留延迟测试原始报告用于对比：

   ```bash
   python tests/atk/chunk_gated_delta_rule_fwd/scripts/perf_a5_msopprof.py \
     --device 0 --case-ids 0 --iterations 1 --warm-up 10 --aic-metrics Default \
     --output-dir output/gdn_perf_detail
   ```

4. 查看 `output-dir/<UTC时间戳>/performance.csv` 和 `performance.json`：分别记录每条模型的
   sample_count、mean/median/min/max/stdev，单位均为 **us**。每个样本保留原始 CSV、行号和
   原始字段，完整 profiler 输出在 `case_<id>/profile/`，工具日志在 `msopprof.log`。
   `worker.json` 记录实际设备、输入、随机种子、wheel 路径和输出有限值检查；有限值检查不代表
   精度通过。外层 `environment.json`、`command.json` 保存工具版本、命令和源码/JSON 哈希。

   统计值来自目标 `OpBasicInfo*.csv` 的 **`Task Duration(us)`**，该字段包含设备调度、执行及
   结束响应，并非纯 kernel 指令执行时间，也不包含 Python 生成输入和 CPU golden 的耗时。
   不汇总 AIC/AIV 分核时间，也不将 Python wall time 作为算子耗时。

5. 若报告没有目标样本或样本数与 `iterations` 不同，脚本返回失败，不生成假零耗时。
   先查 `msopprof.log` 和实际 `OpBasicInfo*.csv`；如果只有名称后缀不同，可用
   `--kernel-name '<实际融合kernel前缀>*'` 调整。不要改成无过滤采集，否则可能采到辅助算子。
   CSV 缺少预期字段时脚本明确报错，需要按实际 CANN 导出格式核对，不能靠列号猜测。

性能对比时固定卡、代码/wheel/OPP、CANN、输入、种子、输出开关、预热/采样参数和其他负载。
当前 JSON 未定义性能达标阈值，脚本提供采集成功与耗时统计，不宣称达到某个性能目标。

## 4. 本次验证边界

脚本的标准库路径、用例解析、命令生成和 CSV 统计可在无 NPU 主机检查。A5 精度、容差校准和
msopprof 实测必须在上述目标环境执行。本脚本的 CPU 混合容差结果不能标记为 ATK 双标杆验收；
本次也不覆盖内存检测、确定性或反向精度。
