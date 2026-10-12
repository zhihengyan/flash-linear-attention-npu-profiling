# op-timer-ascendc

给**已有 CANN 算子**（AscendC kernel + aclnn host + PyTorch 适配层）接入 **AscendTimerV2** 内核计时能力的 Claude Code Skill。

它在 kernel 内用 `Tik`/`Tok` 打点、写入 GM 上的 int64 缓冲，host 侧创建/校验 timer 张量，Python UT 导出解析成**每核每阶段 CSV**，并生成 **perfetto trace** 可视化——用于定位 kernel 内各阶段耗时、AIC/AIV 核间负载均衡与等待气泡。

> 本 skill 只做**观测**：不改算子数值语义、不破坏原有同步与流水结构，也不做算子本身的性能优化（那是 `op-perf` 的事）。

## 安装

把本仓库克隆到任一 skill 目录下即可（二选一）：

```bash
# 用户级（对所有项目生效）
git clone <本仓库URL> ~/.claude/skills/op-timer-ascendc

# 项目级（仅当前项目）
git clone <本仓库URL> <项目根>/.claude/skills/op-timer-ascendc
```

克隆后目录应为 `<skills 根>/op-timer-ascendc/SKILL.md`。Claude Code 会自动加载 `SKILL.md` 的 frontmatter。

## 何时触发

内核计时、kernel 计时、AscendTimer、Tik、Tok、打点、perfetto、核间等待、等待气泡、负载均衡、kernel 内部耗时。

## 前置条件

- 目标算子已存在：AscendC kernel + host/调用适配；沿用 `_torch_adpt.h`、ctypes、Stable ABI 或 C++ 等现有路径。
- 目标算子能编译并能跑通一个最小调用（否则无法验证计时）。
- 计时框架文件（`assets/` 四件）尚未在目标工程：不存在则由本 skill 拷入，已存在则复用。

## 前置补充（Step 0）

先确认芯片（默认推荐 A5/ascend950）、单个公开算子入口、Docker/CANN/Python 环境和工程原性能用例。保留下面 Step 1–9；新入口与原测试语言一致，主入口自动完成采集。A5 优先检查缓存发布，所有相关平台检查 OPP 初始化和加载一致性。

## 工作流（Step 1 → Step 9）

| 步骤 | 内容 |
|------|------|
| 1 | 文件准备与前置确认（拷入 `assets/` 四文件） |
| 2 | host 侧接入（`Input("timer")` + 适配层张量校验 + `timerAddr` 入参） |
| 3 | **画原始流程图 → 按每个关键原理步骤直接规划打点 → 核对覆盖 → 改枚举** |
| 4 | kernel 主流程初始化（`Init` + 原有同步 + `TikNoBarrier`） |
| 5 | 固定项打点（按完成锚点选 Tok 模式） |
| 6 | 动态项打点（循环内，Overwrite/Accumulate 与 barrier 选型） |
| 7 | 子模块打点 + 停测管理 |
| 8 | 原工程同语言主入口 + UT 导出/可视化/验证 + 集中运行说明 |
| 9 | **收尾独立校验**：由**独立 agent** 对照原始流程图核对打点位置、Tik/Tok 配对、枚举/命名/清单/图四方一致，问题如实上报 |

Step 3 与 Step 9 是本 skill 的重点纪律：

- **Step 3**：原始图以 host 为主线，直接覆盖每个关键原理步骤并标注实际 Tik/Tok，不提供不同粒度的候选方案要求用户选择。例如 GDN 根据实际代码覆盖 KKT、下三角求逆/求解、W/U、H、O，并区分下发/等待/完成窗口。
- **Step 9**：打点完成后，由**不参与实现的独立子 agent** 拿着流程图与落地代码交叉校验，主判据包括「每个 `Tik` 是否有对应 `Tok` 收尾」「打点位置与流程图是否完全匹配」；任何不一致、缺失配对、图文不符都必须**如实列出**（含证据）。

## 目录结构

```
op-timer-ascendc/
├── SKILL.md                 # skill 主入口（frontmatter + 完整工作流）
├── assets/                  # 可分发模板（拷进目标算子工程）
│   ├── AscendTimerV2.hpp            # → 目标 op_kernel/
│   ├── AscendTimerV2_device.hpp     # → 目标 op_kernel/
│   ├── parse_timer_csv.py           # → 目标 tests/ut/
│   └── trace_parser.py              # → 目标 tests/ut/
└── references/              # 本 skill 私有参考
    ├── migration-guide.md                        # 计时原理 + 三侧迁移步骤 + 核对清单
    └── worked-example-fused-attention-op.md   # 12 步完整落地案例（算子/内部符号已匿名化）
```

## 产出

- 打点后的 kernel / host / UT 代码（目标算子工程）
- `rank_*_npu_time_parsed.csv`（每核每阶段 start/end/duration）
- `*_npu_time.json` / `all_rank_trace.json`（perfetto trace）
- `docs/timer_flow.md`（原始流程图 + 关键步骤打点/清单/锚点）
- 同目录 `timer_run.md`（Docker、构建、本轮准确产物安装、一个主要入口）和 `timer_report.md`（结论/审核/实测状态）
- 独立校验报告（Step 9）
- 计时结论（各阶段耗时表 + 核间负载均衡 / 等待气泡）

模板局部补丁验证：`python scripts/validate_assets.py`。C++ host stub 与离线解析不能代替 CANN/NPU 实测。

## 许可

[MIT](LICENSE) © 2026 Send_Tanks
