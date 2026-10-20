# main 与 main-k32 的 GDN 前向验证步骤

在 **Linux + Ascend A5（ascend950）** 机器上执行，工作目录为项目根目录。K32 改动只在 A5 Prepare 内核中生效。

本次只验证公开接口 `chunk_gated_delta_rule_fwd` 的端到端精度和性能，暂不单独验证 Prepare。仓内两条 A5 模型性能用例都会调用 `Prepare → H → O`，因此包含 K32 优化的收益，也包含其余阶段和 Cast/Transpose 等辅助 kernel 的开销。

两个分支使用同一张卡、同一个 Python/CANN/ATK 环境，依次运行，测试期间不要让其他任务占卡。先激活已有的 NPU Python 环境，环境中需有匹配的 torch、torch_npu、triton-ascend、ATK 和 openpyxl；ATK 不低于 26.8.8。

仅使用以下仓内原始用例，不运行生成器，不改 JSON、输入值域或精度阈值：

| 验证项 | 用例文件 | 当前数量 |
| --- | --- | ---: |
| GDN 前向精度 | `tests/atk/chunk_gated_delta_rule_fwd/atk_chunk_gated_delta_rule_fwd.json` | 500 |
| GDN 前向性能 | 同目录 `atk_chunk_gated_delta_rule_fwd_perf.json` | 2 |

GDN 的两条性能用例分别是仓内的推理模型和训练模型。另执行 `ci/example_st_cases.json` 中已启用的 Example/ST 用例。

## 1. 准备环境，只做一次

下面各段命令在同一个 Bash 终端中依次执行。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -eo pipefail

which python python3 atk
python scripts/check_npu_env.py
python -c 'import openpyxl'
atk --version
npu-smi info

export DEVICE=0
export VERIFY_ROOT="$(pwd)/../verify-k32-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$VERIFY_ROOT"
export PYTHONPATH="$PWD:$PWD/tests/atk/chunk_gated_delta_rule_fwd${PYTHONPATH:+:$PYTHONPATH}"

# 清除旧的构建筛选、环境切换和用例筛选设置。
unset FLA_NPU_OPS FLA_NPU_BUILD_ARGS FLA_NPU_OPP_PATH
unset ATK_ENV FLA_NPU_ENV FLA_NPU_OPP_ENV
unset CASE_START CASE_END ACCURACY_START ACCURACY_END
unset PERFORMANCE_START PERFORMANCE_END
unset GDN_ATK_CASE_JSON GDN_ATK_MATRIX_START GDN_ATK_RESULT_DIR

# 确认两个分支的测试代码和原始用例一致；正常时没有输出。
git diff --exit-code refs/heads/main refs/heads/main-k32 -- tests/atk ci/example_st_cases.json

atk --version > "$VERIFY_ROOT/atk-version.txt"
python -m pip freeze > "$VERIFY_ROOT/packages.txt"
npu-smi info > "$VERIFY_ROOT/npu-info.txt"
cat /usr/local/Ascend/ascend-toolkit/latest/opp/version.info > "$VERIFY_ROOT/cann-version.txt"
```

`DEVICE` 是当前环境可见的逻辑设备号，按实际情况修改。先确认 `python`、`python3`、`atk` 属于同一个已激活环境。

## 2. 编译、安装当前分支

先用 `BRANCH=main` 执行第 2～4 步。完成后，把下面首行改成 `BRANCH=main-k32`，再完整执行第 2～4 步。

两个分支都全量编译并重装 wheel，确保 Prepare、前向调用链和双标杆所需算子全部来自当前分支。

```bash
BRANCH=main
git switch "$BRANCH"
OUT="$VERIFY_ROOT/$BRANCH"
mkdir -p "$OUT"
git rev-parse HEAD | tee "$OUT/commit.txt"

sha256sum \
  tests/atk/chunk_gated_delta_rule_fwd/atk_chunk_gated_delta_rule_fwd.json \
  tests/atk/chunk_gated_delta_rule_fwd/atk_chunk_gated_delta_rule_fwd_perf.json \
  ci/example_st_cases.json > "$OUT/cases.sha256"

FLA_NPU_SOC=ascend950 python scripts/build_wheel.py \
  --wheel-dir "$OUT/wheel" 2>&1 | tee "$OUT/build.log"

WHEELS=("$OUT"/wheel/*.whl)
test "${#WHEELS[@]}" -eq 1
test -f "${WHEELS[0]}"
python -m pip install --force-reinstall --no-cache-dir --no-deps \
  "${WHEELS[0]}" 2>&1 | tee "$OUT/install.log"

OPP_ENV=$(python -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("fla_npu").origin).parent / "opp/vendors/fla_npu_transformer/bin/set_env.bash")')
source "$OPP_ENV"
python scripts/verify_libcust_opapi_md5.py --python \
  2>&1 | tee "$OUT/installed-md5.log"
```

接口库、kernel `.o` 和 Python 文件的校验都应显示 `[OK]`。若出现 `[FAIL]` 或跳过 kernel 校验，先处理安装问题，再测试。每次安装后都使用新启动的测试进程，下面的命令会自动启动新进程。

## 3. 精度验证

### GDN 前向：完整 500 条

使用仓内双标杆分片入口，每批默认 25 条。该入口同时比较 NPU DUT、六算子 NPU benchmark 和 CPU FP64 golden。

```bash
GDN_ATK_CASE_JSON="$PWD/tests/atk/chunk_gated_delta_rule_fwd/atk_chunk_gated_delta_rule_fwd.json" \
GDN_ATK_MATRIX_ROOT="$OUT/fwd-accuracy" \
GDN_ATK_SHARD_SIZE=25 GDN_ATK_MATRIX_START=0 \
  bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh "$DEVICE" \
  2>&1 | tee "$OUT/fwd-accuracy.log"

python -m json.tool "$OUT/fwd-accuracy/aggregate_summary.json"
```

汇总必须为 `complete: true`、`passed: true`，并且 `observed_cases` 为 500。不要用普通单标杆入口替代这组双标杆测试。

### Example/ST：仓内已启用用例

```bash
python ci/run_example_st_cases.py \
  --device "$DEVICE" --cases-file ci/example_st_cases.json \
  --accuracy-report-file "$OUT/example-st.json" \
  2>&1 | tee "$OUT/example-st.log"
```

上述精度测试全部通过后，再执行性能测试。只跑了部分用例或出现失败，不能记为精度通过。

## 4. 性能验证

GDN 前向的 2 条模型性能用例都完整运行，重复 3 轮。预热与内部循环沿用同一 ATK 版本的默认设置，不修改用例。

```bash
for ROUND in 1 2 3; do
  ATK_OUTPUT_ROOT="$OUT/performance-$ROUND/chunk_gated_delta_rule_fwd" \
    bash tests/atk/run_test_cpu.sh \
      -op=chunk_gated_delta_rule_fwd \
      -npu_device_id="$DEVICE" -soc=ascend950 -scope=performance \
    2>&1 | tee "$OUT/performance-$ROUND-fwd.log"
done
```

这里使用 ATK 的外部性能采集，不需要在源码中加入 Tik/Tok 或 timer 张量。

## 5. 对比结果

确认两个分支都完成第 2～4 步，然后执行：

```bash
diff -u "$VERIFY_ROOT/main/cases.sha256" "$VERIFY_ROOT/main-k32/cases.sha256"
find "$VERIFY_ROOT" -type f \( -name '*.xlsx' -o -name 'op_summary*.csv' \)
```

哈希对比应没有差异。按相同算子、相同 case 分别比较，不能把不同 case 的时延混在一起。

- 精度：两个分支都必须通过完整 GDN 前向和 Example/ST 测试，原始精度规则不变。
- 性能：按仓内性能口径，查看 ATK 保存的 profile 中 `op_summary*.csv` 的 `Task Duration(us)`，对一次 GDN DUT 调用的全部 kernel 时延求和，包括 Prepare、H、O 及其 Cast/Transpose 等辅助 kernel。用这个总值比较端到端收益。
- 同一 case 先取每轮有效采样的平均值，再比较 3 轮结果及波动。不要使用 Python 执行时间或整段测试命令的耗时判断收益。
- 时延下降比例：`(main时延 - main-k32时延) / main时延 × 100%`。正数表示 K32 更快。

若差异接近运行波动，可切回 `main`，重装第 2 步保留的 main wheel，再按第 4 步复测。结果写到新的 `OUT="$VERIFY_ROOT/main-repeat"` 目录，避免覆盖第一次结果。

以上是 GDN 前向的精度和性能对比步骤；本轮暂不进行 Prepare 单独验证、确定性和内存检测。
