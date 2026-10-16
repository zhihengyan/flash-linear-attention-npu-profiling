# Prepare Stage5 A 直接供 L1 实验

方案设计规则版本：V2。分支 `codex/gdn-stage5-a-direct-l1`，直接基于K32 `d0cb628df7c8c23c0342392dfaf4ec910aba9947`。本轮唯一变量是完整chunk且outputA开启时，Stage5如何把A交给Stage7；GM公开A输出、K32求解及其它阶段保持。接口沿用 [api.md](api.md)、现有Prepare ATK README和CPU golden；不新增公开参数或实验开关。

## 1. 目标与实测边界

前轮Stage3结果与放弃决定已保存于 `codex/gdn-stage3-nz-upload` 提交 `21d9998a`，本分支不继承其kernel或资源改动。按用户授权直接接入普通/Profile整链，跳过逐Stage设备验证的编码门槛；正式设备精度、MSS、确定性和性能仍须验证，不把K32精度通过当作本轮通过。

模型case0仍为Ascend950PR_957c，B1/HK16/HV32/T11274/K=V128/BT64，G2、varlen，q/k/v与beta为BF16、g及初态为FP32。固定case1为B2/HK=HV32/T8192/K=V128，原ATK配置保持。原始输入、CPU标杆、值域和比较阈值保持。

当前K32普通case0七项均值1610.88620us、Prepare620.28335us；约2.7%参考对应净省43.4939274us（若全来自Prepare则7.011945%）。3%与2.7%不是强制指标。本轮代价包括额外Fixpipe和L0C占用，原回读可能命中L2且已被Stage6隐藏，没有已证明的收益。以同环境K32对本候选的普通msprof Task Duration判断，不能用逻辑字节或timer下发窗推算加速。

主指标为每次Cast、Transpose(g)、Transpose(beta)、Prepare、H、O、Transpose(gCumsum)七项Task Duration之和，最后转置是gCumsum `[B,H,T]→[B,T,H]`。普通原采样配置保持（历史20组/轮），warmup5只控制诊断重放；case0至少三轮独立进程并用新鲜K32交错对照，case1及全支持场景按原要求回归。

## 2. 数据、地址与生命周期

沿用CHUNK_INDEPENDENT V1参考（来源 `a78efa9512dfa64c20b2a58a75e84e88a054d694`）的独立chunk与原ready/free闭环；工程参考为K32自身的tail/outputA=false直接Fixpipe实现。不进行R01任务映射或R06数据流的全量迁移，也不增加Vector→Cube边界；本轮是已有Cube结果的L0C→L1供数。

| 区域 | 生产与最后消费 | 本轮变化 |
| --- | --- | --- |
| 四个L0C task槽，byte基址`t*64KiB`，每tile FP32 16KiB | Stage4/5计算；Stage5 Fixpipe读取；后续Cube复用 | 地址和份数保持；完整chunk的第二个Fixpipe必须读完后才能释放FIX_M credit |
| L1 A，`[256+8t,264+8t)KiB`，t=0..3 | Stage5写；Stage7 W/U的MTE1读；随后下一pack覆盖 | 同一8KiB BF16 NZ16槽，改由Fixpipe写入完整chunk |
| GM A，BHT64 ND，元素基址`OffsetBHTD(batch,hv,tokenStart,HV,T,64)` | Stage5写有效M行；公开输出和后续消费者读取 | 写出参数/范围保持，仅移除完整chunk的GM→L1读取 |
| UB、其它L1/L0、Y workspace、host workspace | K32原生命周期 | 全部保持；UB上限236/248KiB、L1最高496/512KiB |

输入A是Stage5已完成的同一份FP32 L0C结果，两次Fixpipe均只读它，不改变FP32求解、归约顺序或BF16转换目标。L0C为NZ16，GM输出为ND，L1输出为NZ16/C0=16，不能把GM ND物理字节直接当NZ。沿用原 `DumpAL0cToL1` 参数：M=N=64、srcStride=64、dstStride=64*16（元素）、F322BF16、isChannelSplit=false。沿用原GM写出：mSize=有效M、nSize=64、srcStride=64、dstStride=64、F322BF16。

完整chunk时目标L1恰覆盖4096个BF16值，8KiB；tail仍完整写L1补齐tile，而GM只写有效行。两个消费者的有效数值应与原同一L0C转换结果一致。离线模型验证有限FP32/RNE地址与位模式，不能替代硬件舍入、NaN/Inf行为或设备精度验证。

case0中176个完整chunk×32heads=5632tasks受影响，每个少一次8KiB GM ND回读/转NZ，共44MiB逻辑GM地址读取；新增5632次8KiB L0C→L1 Fixpipe。其余32个tail task、全部outputA=false路径不纳入该减量。GM输出写量保持，无新增GM缓冲。

## 3. Stage5、Stage7与同步详设

K32完整chunk路径为：Wait M_FIX(bank) → Fixpipe GM → FIX_MTE2(bank) Set/Wait → CopyGmNdToL1Nz → Set MTE2_MTE1(task) → Set FIX_M(bank)。

本轮完整chunk路径改为：

```text
Wait M_FIX(bank)
→ 原Fixpipe GM（若outputA开启）
→ 原DumpAL0cToL1(task,64)
→ Set FIX_MTE1(task)          # A已可被MTE1消费
→ Set FIX_M(bank)            # 两次Fixpipe都完成后才归还L0C的Cube credit
```

GM和L1两个Fixpipe在同一PIPE_FIX上按序下发，源L0C只读共享；不增加PIPE_ALL或跨核flag。保持原tail/outputA=true分支的FIX_MTE2 Set/Wait顺序，以限制实验只改变完整chunk路径；outputA=false沿用原单次L1输出。共享收尾为DumpAL0cToL1、Set FIX_MTE1(task)、Set FIX_M(bank)。

ProcessAic中Stage5前的 `Wait MTE2_MTE1(task)` 是Stage4 Y的ready，必须保留。Stage7前则统一 `Wait FIX_MTE1(task)`，匹配Stage5新ready；不能再按outputA/full条件等待不再发布的A回读MTE2事件。Stage6 ready等待及其独立推进、Stage7 W/U内部事件均保持。

FIX_MTE1的task id为0..3，与K32的tail/a0已有协议相同。ProcessAic初始化的FIX_MTE1(0) credit在本pack起点先消费；Stage5逐task发布的0..3分别在Stage7前消费；pack末原FIX_MTE1(0)只在Stage7全部下发后发布，供下一pack起点消费。id0不能在Stage7漏消费，否则会留下重复credit。G3 pack3以及末pack不足4只为实际task发布/等待，空核不产生task事件。

Stage7 W/U读取L1 A之前必须等待新的FIX_MTE1。Stage5后的FIX_M(bank)仍在原位置、但排在新增L1 Fixpipe之后，保护偶/奇bank复用；Stage7重用L0C0/1及下一packStage2之前还受原pack末FIX_MTE1(0)顺序保护。Stage7最后消费L1 A后才允许下一packStage5写入；AIC顺序和原pack drain不变。

## 4. 实现、验证与停止条件

生产kernel只改Prepare arch35主头中的Stage5_Dump与ProcessAic对应的A-ready消费，并更新注释；原DumpAL0cToL1、GM helper、K32 MMAD/VCS、内存表、入口、host/ABI及H/O全部保持。普通和Profile消费同一共享Prepare实现，不创建未接入helper。

离线检查从实际C++提取分支、Fixpipe参数、事件和消费者顺序：核对有限FP32→BF16的GM/L1地址、全部valid1..64、canary与错误stride/转置负例；检查full/a0/tail、G1/2/3/4、V128/256、partial pack和同核连续pack的事件计数、L0C和L1复用。历史K32四项明确冻结d0；当前新检查覆盖修改后的源码及其余区域与K32保持。原timer布局保持，源清单行号随代码更新。

验证入口沿用完整wheel→安装→产物核对→原精度→profile_performance.py。移植前轮已验证的可选`--reference-wheel`及其15项测试，默认入口仍对照build staging；不移植Stage3 kernel。设备命令与结果统一写入 [validation.md](validation.md)。

本机没有CANN/NPU，完成整链编码、离线检查与可复用设备步骤，不宣称编译/设备验证通过。若额外Fixpipe抵消回读收益，或A回读原已被Stage6完全隐藏，则停止该候选；采用需原精度、MSS、确定性和同条件重复普通性能证据。
