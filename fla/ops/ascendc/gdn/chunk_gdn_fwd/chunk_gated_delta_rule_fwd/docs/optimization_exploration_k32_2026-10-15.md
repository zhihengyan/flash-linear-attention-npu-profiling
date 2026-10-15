# K32 之后的 GDN 优化路径探索

后续实现：用户明确选择主候选，并授权暂时跳过逐Stage设备验证门禁。在当前`codex/gdn-prepare-block32`分支、干净K32基线`f27e968b`上完成整链接入，见[Block32详设](prepare_block32_design.md)和[验证记录](validation.md)。下文“未修改生产kernel”等为探索阶段状态，不代表当前工作树；实际性能尚待设备测量。

日期：2026-10-15；方案规则 V2，当前处于阶段 03。本文是候选筛选和进入详设前的工程分析，不是已通过整体评审的实现方案。公开接口、CPU golden、输入生成、容差和支持范围保持当前约定。

本轮结论：**优先细化 Prepare 的 32×32 分块求解与完整 A 重组**。它从 K32 已有的归约裁剪继续消除结构零计算，有比局部搬运时序更大的静态预算。用户已明确：不强制相对K32再降5%，理论七项总耗时收益低于3%的方向直接放弃。按此门槛，单独消除格式转换排除；N32仅条件保留，分块求解继续详设；H的BF16 H/right中转为独立备用。没有修改生产kernel，没有将任何新候选记为采用。

## 1. 接手状态与证据

最初主目录 `D:/Code/flash-linear-attention-npu` 在 `codex/gdn-stage6-overlap` / `dddd7bce`，工作树干净。用户指定的交接文件位于本地 `codex/gdn-prepare-k32` / `d0cb628d`，不在最初检出的 Stage6 分支中。已通过 Git 历史读取完整交接，并只读核对已有 K32 工作树干净。

在主目录创建 `codex/gdn-k32-next-path`，起点为 `d0cb628df7c8c23c0342392dfaf4ec910aba9947`；原 Stage6、K32 分支和已有工作树保留。生产 Prepare 源码仍为优化提交 `dd707730627b29b1862fd335dfbc76e7f0e2666d`。没有叠加 Stage6 提前 K、H 双 head lookahead 或 Stage3–5 诊断分支。

流程已按根 AGENTS、阶段 03/04/05 和 ATK 路由核对；重新读取当前接口、设计、验证记录、Prepare/H/O 的相关实现和标杆/测试。现有 `design.md` 主要记录诊断增量，不能据此声称整个业务实现已完成 V2 全面迁移。本轮没有进入阶段 04。

只读重算 `D:/Code/results` 下同日三轮 A、三轮 K32 B；各 140 行，逐行核对 metadata 中的名称/耗时、顺序与七项调用分组，确认输入、case JSON、设备和 Python 身份字段一致。六份原始报告未改写，文件 SHA256 写入本机忽略目录中的筛选结果。没有重新要求已完成的新鲜 A 复测。

| 指标，普通 Task Duration(us) | 同日 A，60 次调用 | K32，60 次调用 |
| --- | ---: | ---: |
| 七项逐调用求和后的均值 | 1704.80063 | 1612.33015 |
| Prepare 均值 | 714.91452 | 620.64430 |
| H 均值 | 586.75883 | 588.33007 |
| O 均值 | 357.92022 | 358.23867 |

K32 相对同日 A 的既定均值收益为 **5.424%**，继续保留。这里是七项设备任务时长之和，非墙钟端到端时延；各 60 次来自各三轮采集。p95 降幅 4.956%、未达到 5% 的已有边界不变，详见[同日复核](timer_analysis_k32_same_day_ab_2026-10-15.md)。

用户报告 K32 精度测试全过；本机仍没有完整 case/seed 报告、case1、MSS、确定性和 Example/ST 的本候选全套结果。因此它是后续探索基线，尚不归档为阶段 05 全部完成。本轮没有改变这些已知待验项。

## 2. 新预算与当前 pipe 线索

原始 A 的 ≥5% 均值目标已达成。用户本轮明确采用**相对K32的理论增量3%筛选线**，不再要求相对K32至少5%；同时保留相对原始A的累计收益。3%用来决定方向是否值得投入，不把粗略估计冒充性能预测或实测验收结果：

- 3%对应节省48.36990us，等价七项均值1563.96025us。
- 若收益只来自Prepare，等价下降7.793%；只来自H，等价8.222%；只来自O，等价13.502%。
- 按扣除新增搬运、转换、同步和流水损失后的七项总收益筛选，不能用某个Stage的MAC下降或timer窗下降3%代替。
- 只有在乐观减量仍低于3%，或已证明可实施净收益低于3%时，才可据此直接放弃。区间跨过3%的候选应标明缺失证据，不能把较小情景称为理论上限。

一个可直接排除的独立方向：若Prepare/H/O不变，仅消除当前Cast和三次Transpose，理想节省也只有45.11712us、**2.798%**，低于3%。不为单独去掉这些转换安排实验。若另一个方案同时改变消费者布局并减少其内部搬运，应按新的完整改动范围重新预算，不能套用这个有限范围的上限。

现有 K32 普通 CSV 的每项 60 条 pipe 字段均值如下。它们是报告字段的平均值，pipe 可重叠，不能相加、相减得到关键路径，也不是某 Stage 的独占时长。

| kernel | AIC MAC | AIC MTE1 | AIC MTE2 | AIC Fixpipe | AIV Vector | AIV MTE3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prepare | 390.684 | 161.292 | 90.221 | 165.737 | 348.959 | 269.250 |
| H | 148.553 | 81.629 | 451.986 | 189.827 | 140.777 | 100.130 |
| O | 96.545 | 75.542 | 330.541 | 105.196 | 80.319 | 57.813 |

Prepare 仍有计算减量理由，但 Vector 和交接成本不能忽略。H/O 的 MTE2 字段较大，支持核对中转、格式和加载流水；不证明其全部可消除。旧 K64 的细 timer 窗不用于给 K32 的 Stage 排名。本机没有 K32 的完整硬件 timeline，不声称已确定唯一关键路径。

## 3. 主候选：只求左下块，重组完整 A

### 3.1 代数与唯一主要变量

沿用 Stage3 已有的两个 FP32 leaf 逆矩阵，不修改 Vector leaf 求解算法。设：

```text
I + L = [[I32 + L00,        0],
         [       L10, I32 + L11]]
R00 = inverse(I32 + L00)       # Stage3 已有
R11 = inverse(I32 + L11)       # Stage3 已有
X   = R11 @ (-L10)            # FP32 [32,32] @ [32,32]
C10 = X @ R00                # FP32 [32,32] @ [32,32]
A   = [[R00,   0],
       [C10, R11]]
```

当前 K32 仍执行 `I@I` 预填、`Left@N`、`I@Left`、`Y@Right`。上述形式只保留最终 A 左下块真正需要的两次乘法，并直接复用已求得的对角块。唯一主要变量为 **Prepare Stage4/5 的求解及结果重组数据流**；主候选不同时改变 Stage1/2/6/7 算法、CG/pack 映射、H/O 或输入精度。

这不是把 W/U 改成不产生 A 的递推：公开前向默认disable_recompute=True会导出GM A，因此必须保留完整BF16 A输出；H实际消费W/U，不直接读取A（整链接线时按host实参修正原描述）。tail仍按原补齐方式生成完整A，并只向公开GM A写有效行；`output_a=false`仍向W/U提供完整L1 A。

### 3.2 计算与搬运预算

| 求解路径 | 每 task FP32 MAC | 相对 K32 减量 |
| --- | ---: | ---: |
| 原始 K64，四次 64³ | 1048576 | 历史参考 |
| K32，64³ + 3×64²×32 | 655360 | 基线 |
| 本候选，两次 32³ | 65536 | **90%** |

case0 原任务数为 5664，pack=4、1416 packs，28 AIC 各处理 50/51 packs；本候选不改该分配。每 task 少 589824 MAC，合计少 3340763136 MAC。比例仅针对 FP32 求解部分，不是整个 Prepare 或七项收益。

为隔离风险，首版仍保留每 task 独立的 GM 中间槽，不直接把 FP32 ChannelSplit 的目的地改成 L1。将 X 以 32×32、C0=8 的合法形态中转，再供第二次 MMAD 使用；descriptor 和嵌入槽 stride 需详设确认。有效 X GM 写/读各 4KiB，代替 Y 各 16KiB，逻辑中转量减少 24KiB/task（全 case0 132.75MiB）。保留原 16KiB 槽及 workspace 总分配，首版不做 host 缩容。

第二次 MMAD 的 C10 计划 Fixpipe 到所属 AIV 的已释放 UB，AIV 将原 FP32 leaf、C10 拼成 BF16 A，并按原输出约定写 GM/L1。相较现有输出路径，新增 C10 的 4KiB UB 写入、AIV 拼接/VF、UB→L1 的 8KiB A 及同步；GM A 输出仍需完整有效行。不能将减少的 FP32 计算和中转字节全算作净收益。

仅作投入筛选：前次 K64→K32 少 393216 MAC/task、Prepare 少 94.27022us；若**假设**所有减量有相同边际暴露率，新候选的计算部分对应141.40533us，即七项总量的8.770%。相对3%筛选线，新增拼接、布局、同步及流水损失的条件余量约93.03542us。

但当前`Stage4_PrefillI`明确在Stage3 ready等待之前下发，目的就是隐藏I@I。新候选删除的589824 MAC中有262144来自预填；若预填已经完全隐藏，其余计算减量只有327680 MAC，按同一粗略比例对应**78.55851us，即4.872%**。相对3%仍有**30.18861us**的条件余量，需覆盖新增拼接、布局、同步和流水损失。减少X/Y中转可能另外贡献收益，也可能被已有重叠隐藏。

这两个情景不是上下界或预测，4.872%更不是保证收益；前次边际暴露率可能不再成立。新门槛下该方向值得继续详设，无需先证明能达5%。详设须量化新增代价，并继续核对K32预填/X中转的关键依赖；若可实施净收益低于3%，直接停止，不因90%的局部MAC减量继续投入。

### 3.3 可用地址与必须闭合的依赖

以下是合法容量方向，不是可直接编码的最终资源表。所有区间单位 KiB，沿用现有绝对地址；task/db/owner 对应关系不变。

| 对象 | 拟用现有区域 | 首次写入、最后读取与复用条件 |
| --- | --- | --- |
| X 的 L1/GM 槽 | L1 `[16t,16(t+1))`；GM 每 AIC 原四个 16KiB Y 槽 | 首次 MMAD/Fixpipe 生产；第二次 MTE1 末次读取；不减槽数、不提前复用 |
| N、R00、R11 | 原 NegL、LeafRight、LeafLeft 槽 | Stage3 生产；X/C10 读取相应象限；NegL 与下一 pack 的 K 别名关系仍受 S4 free 保护 |
| C10 接收 | AIV db0 `[10,14)`；db1 `[50,54)`，原 LPacked 的前 4KiB | 原 LPacked 已被 VCS 消费且 Stage3 ready 发布后，才允许 AIC Fixpipe 覆盖；拼接 VF 末次读取 |
| 两个 leaf | 原 ResVcs db0 `[18,34)`、db1 `[58,74)`，实际 packed 32×64 FP32 为 8KiB | 保留至 A 拼接 VF 完成；不能先进入覆盖该区的 Stage6 |
| A 拼接 scratch | 原 LFull db0 `[34,50)`、db1 `[74,90)`，BF16 完整 A 为 8KiB | -L 上传结束后可覆盖；V→MTE3 完成且 GM/L1 写完后，才可供 Stage6 复用 |
| L1 A | 原 `[256+8t,264+8t)`，BF16 NZ C0=16 | AIV 唯一整 tile 写者；Stage7 MTE1 消费；与上一 pack Stage7 的复用仍需 ready/free 证明 |
| L0A/B/C | 原物理 banks 和每 task 的 64KiB L0C stride | 32×32 块不超过原容量；实际描述符、事件和覆盖范围逐项验证 |

不新增 UB/L1 分配；最高地址保持原规划（V128 UB 236/248KiB、L1 496/512KiB）。这不能替代生命周期审查。特别不能在发出 Stage3 ready 之前将 LPacked 当结果缓冲，也不能先跑 Stage6 覆盖保留的 leaf。

候选调度骨架：

```text
AIV：原 Stage1 → Stage3，完成两 task leaf/N 上传，逐 task 发布 S3 ready
AIC：原 Stage2 → 等 S3 ready → X MMAD → X GM/L1 中转 → C10 MMAD
     → C10 Fixpipe 到该 task 所属 AIV 的 LPacked 空闲区 → 发布 C10 ready
AIV：逐 task 等 C10 ready → 单次 VF 拼接/转 BF16 A
     → 写 GM A（若要求）及 L1 A → 保证相关 MTE3 已完成
     → 原 Stage6 → 发布原 S6 ready（同时保证 A 已就绪）
AIC：等 S6 ready → 原 Stage7；之后沿原规则释放/下一 pack
```

C10 ready 可研究按 phase 复用 0..3 的 KKT flag：同一 task 先消费 KKT ready，再生产/消费 C10 ready；不能只添加一次 Set 或使用已知无效的 12..15。S4 的跨 pack free 与新 C10 ready 必须分别闭环。AIV 的 wait 必须阻塞实际消费 pipe；标量时间戳不能证明 PIPE_V/FIX/MTE 等待完成。

下一步详设还必须证明：初始 credit、两个 AIV 的 +16 映射、G1/2/3/4 和 partial pack、尾块、同核连续 pack、结果 UB 的生产/消费/释放，以及 L1 A 的跨 pack 复用。现有 flag 是否可安全复用、所有事件数量及各 pipe 顺序尚未完成整体评审，**不能据此进入 kernel 编码**。

### 3.4 数值门禁与离线结果

FP32 leaf 算法、两次矩阵乘的有效乘积顺序和最后 BF16 cast 目标不变；去掉的是结构零对应的输出区与单位阵乘法。若 `|Lij|≤l`、`|Rij|≤r=max(1,l(1+l)^30)`，则 `|Xij|≤32rl`、`|C10ij|≤1024r²l`，与现有 K32 的有效左下路径同阶；没有引入新的除法/指数或更早的 BF16 cast。此条件界不是新增接口输入约束，原真实值域仍需按既定精度路由验证。

使用独立 NumPy 模型，1..64 的全部有效长度 × 4 seed × 三种严格下三角幅度（0/0.02/0.2），共 **768 组**：

- 分块 A 对 K32 FP32 模型最大绝对差 `1.0430813e-7`；对独立 FP64 solve 最大绝对差 `8.8496761e-8`。
- 无效行/列按补齐单位阵处理，覆盖 32 两侧边界；非对称 leaf 的负例能发现漏转置。
- 软件 BF16 RNE 后，分块模型与 K32 共 **7 个 A 元素不同**；两个独立 128 列 RHS 模拟 W/U 的 FP32 结果最大绝对差 `1.9073486e-6`。不声称位级一致，不据此改变 ATK 容差。
- 完整 BF16 A 的四个逻辑象限经 NZ C0=16 地址模型覆盖全部 4096 元素、256 个互不重叠的 32B 块。这里只验证逻辑拼接所有权，**没有验证 LoadData/Fixpipe descriptor、Vector 指令或动态同步**。

这些随机下三角是设计筛查输入，未替代、校准或扩展正式 ATK 生成器。FP32 归约/舍入、BF16 转换指令、非有限输入的行为及原精度标准均需设备实测；不宣称对任意 NaN/Inf 等价。

## 4. 备用候选及停止依据

| 路径 | 代码事实和理论变化 | 当前取舍 |
| --- | --- | --- |
| 保守 N32 | Stage5 只需 Y 左 32 列。将 Stage4 I@I / Left@N 的 N 改为32；Stage5 保持完整 I@Left 初始化，仅向 A 左半区累加 Y_left@R00。MAC 655360→393216（-40%），Y 有效中转减半 | 条件保留。按前次边际为62.847us（3.898%），若删除的预填已隐藏则31.423us（1.949%），尚未扣代价，区间跨3%。若预填已隐藏且中转没有额外暴露收益则放弃；否则须先证明净节省可达48.370us。当前不优先安排设备实验 |
| Stage3 条带上传 | 每 task 8次 -L、8次 leaf UB→L1；K32 未减少这些调用 | 旧 timer 标量窗不能给出可节省时长；缺合法批量格式和scratch证明，暂不编码 |
| Stage5 保留 GM A、额外直供 L1 | full/outputA 路径原先 GM 写 A 后回读；tail/a0 已有 L0C→L1 BF16 | 需付额外 Fixpipe，单纯少8KiB读不证明够48.370us；尚无达到3%的净收益预算，不安排编码；不能删除公开中间输出约定的GM A |
| H BF16 H/right 直接供 L1 | `LoadStage0H` 从 GM 读 BF16 H；`LoadRight` 从 vUpdateWorkspace 读 BF16 right。AIV 已生产这两份数据 | **独立第二优先路径**。保留 O 所需 GM H；只消除 H 内部回读/私有中转。不是消除已常驻的 FP32 state。须证明 UB→L1 的物理转置/NZ、H/right 共享槽 free、tail 和 H0/末 chunk。不能与 Prepare 主候选合测 |
| O 重用 Q/K、V lookahead | 当前 `Process` 已按 GVA group 的 firstHvInGroup 加载一次 Q/K，Stage4 已有 V 提前一 head 搬运 | 不作为新增优化；MTE2大不等于重复读同一 Q/K。若继续定位，应查 H/V 的布局、cache、pipe等待 |
| 单独消除Cast/Transpose | 四项均值合计45.11712us；Prepare/H/O保持原耗时 | 理想上限2.798%，低于3%，直接放弃该独立方向 |
| Stage6提前K / H双head lookahead | 两项历史实验均未达投入门槛 | 不重启、不叠加进 K32 |

H 直接供数的毛预算：本 case H 的每个完整 head/chunk，H BF16 32KiB 回读，right BF16 16KiB workspace 写+读；tail/首尾状态需按真实有效长度计数。可以减少 GM 流量，但新增 UB→L1 格式搬运会占 AIV MTE3，FP32 state、GM H输出和最终状态均不能删除。尚无该候选的可兑现时长预算，当前排在已经完成代数筛查的分块求解之后。

## 5. 下一步实验和进入阶段 04 的条件

按用户3%理论筛选线，下一步推进分块求解详设：核对非对称LoadData、32×32 FP32 X ChannelSplit/回读、C10→UB、单VF A重组的实际API参数、地址、方向和stride，并按当前规则审查全部ready/free。优先用“预填已隐藏”的情景核算新增代价能否控制在30.18861us条件余量内；该数字仅用于敏感性筛选，不替代设备测量。

需要设备时，只补能区分“预填已隐藏/仍暴露”和“Y中转在关键依赖上/已被隐藏”的K32片段，不重采旧16窗，不再为5%旧条件补证据。N32不与主候选并行编码；仅在它的额外中转/依赖收益足以支撑净3%时重新提升优先级。以当前容量为硬限制，不新增一份完整A scratch，不压缩原workspace或改host；可兑现预算和详设均通过后才构建实验版。

开发按最早受影响的 Stage4 开始：X 对 CPU → C10 对 CPU → A（FP32/BF16、完整/有效区）→ W/U → 整算子；逐项通过后推进。device 调试记录要覆盖同核多 pack、db0/db1、slot 回绕与 tail，正常精度和性能关闭记录。CANN 构建/安装及产物一致性按阶段04执行；不得用本机 Python 结果代替。

精度通过后用 **K32 对新候选** 的相同普通采集条件比较增量，并保留相对原始 A 的累计收益。仍统计固定七项、每次求和、保留全部样本；复用原输入 hash/JSON，普通预热和采样设置相同。采集器与 wheel 身份分开记录；本分支继承 K32 入口，没有 `--reference-wheel` 等其它分支参数。相同 K32 源码/build 与其已验证 wheel 是 A；新候选源码/build 与新 wheel 是 B。

先用现有 K32 pipe 数据作筛选，无需再跑16 capture的旧细打点。主候选若编码，首轮精度后取得普通报告和一个能分辨 MMAD→X中转→C10→A拼接→Stage6/7 的硬件依赖片段；最小带出每轮 `run_metadata.json`、普通 `op_summary.csv`，以及关键 pipe/依赖的小摘要。内部命令按已安装 profiler 的 help 生成，不编造未核实参数，不添加测量用 PIPE_ALL 改变被测流水。

若实现成本吞掉收益，或 K32 的 Vector/Stage6 使缩短后的求解完全被隐藏，停止该候选，再独立验证 H 直接供数；不能凭MAC下降采用。采用与否需同条件重复普通数据超过波动，并保留正式回归：Prepare完整2416条/seed、前向500条双标杆、case1与六个Prepare perf、MSS/确定性、Example/ST。阶段05完成前保留 validation，不将本轮探索写为ATK验收通过。

## 6. 本轮复现与结果边界

NumPy2.3.5 / Python3.12.14；仓库根目录执行：

```bash
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/explore_prepare_solve.py --results-root D:/Code/results --output tests/atk/chunk_gated_delta_rule_fwd/atk_output/next_path_analysis/screen.json
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/test_prepare_k32.py
python -m unittest discover -s fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut -p test_timer_contract.py -v
```

筛查脚本只读用户报告；输出为仓库已忽略的 JSON，不提交原始 CSV/metadata。已有 K32 四项和 timer合同十四项共18项通过。生产 Prepare kernel 对 K32 无 diff；本机未执行 CANN 构建、NPU 精度、MSS、确定性或新性能实验。本轮完成状态为“路径筛选与离线可行性证据完成，主候选详设/设备验证待进行”。
