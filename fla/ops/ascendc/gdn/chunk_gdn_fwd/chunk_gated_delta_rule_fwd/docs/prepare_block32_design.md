# Prepare Block32 整链实验设计

方案规则版本：V2。基线为当前分支 `codex/gdn-prepare-block32` 的 `f27e968b`（K32）；开始前工作树干净。用户确认K32精度全过，并明确授权暂时跳过逐Stage设备验证门禁，要求在当前分支编码并接入普通调用链。本授权允许整链实现，不将本候选记作设备验收通过。

## 1. 目标

按[路径探索](optimization_exploration_k32_2026-10-15.md)主候选，复用Stage3两leaf，仅计算 `X=R11@(-L10)`、`C10=X@R00`（两次FP32 32³ MMAD），再拼 `A=[[R00,0],[C10,R11]]` 并转BF16。求解MAC从655360降至65536/task；该90%仅指求解计算量。

case0为B1/HK16/HV32/T11274/K128/V128/BT64，BF16输入，变长，G2，FP32初态/末态；Ascend950PR_957c，28 AIC/56 AIV，5664 tasks、1416 packs、50/51 packs/core。普通性能口径保持Cast→Transpose→Transpose→Prepare→H→O→Transpose七项每次Task Duration(us)之和；沿用普通ATK采样配置并核对每轮20组，重复三轮，采集器默认warmup5只控制诊断重放。K32均值1612.33015us，原A均值1704.80063us；理论增量3%投入线为48.36990us。扣除可能隐藏的预填后，旧边际proxy约78.55851us/4.872%，尚需覆盖拼接/同步成本，不能视为性能保证。

公开接口、host tiling、CPU golden、输入生成/容差、workspace分配量及Stage1/2/3/6/7数值算法不变。共享Prepare arch35普通/诊断入口直接执行Block32，无实验开关。公开中间输出仍取得完整GM A，WU仍消费完整padded L1 A，H消费W/U。case1、独立Prepare及其它受支持组合同样需回归。

## 2. Stage与资源

保留既有owner-AIV映射：task t的owner=t%2，db=(t/2)%2；G3 pack3，其余pack4，满G2的PackWorkId交换不变。这是既有实现的局部优化，未迁移V2 R01的双AIV切片映射；不声称全量规则迁移。

| 驻留对象 | 地址/容量 | 生命周期 |
| --- | --- | --- |
| L1 X（原Y） | 四个16KiB槽，[0,64)KiB；各仅前4KiB有效 | S4 GM回读→S5 MMAD |
| L1 NegL/khat | [64,128)KiB，四槽 | S2 khat读完后S3覆盖；S4读完后释放 |
| L1 leaves | Right[128,192)、Left[192,256)KiB，full64 NZ C0=8 | S3上传→S4/S5消费 |
| L1 A/Kbg/Vb | 原[256,480)KiB映射 | S6 ready后A GM→L1→S7；不扩容 |
| L1旧I | [480,496)KiB不再使用 | 删除初始化和I@I预填 |
| L0A/B | 原四个16KiB bank | S4用P bank，S5用普通bank；每次仅前4KiB |
| L0C | 原四个64KiB task槽 | S4 X/S5 C10各init=true，结果在各槽基址前4KiB |
| UB index | 原[0.5,1)KiB | 前32B给VCS；上256B保存0..63的scatter索引，S0生成一次 |
| UB C10 | 原LPacked前4KiB：db0[10,14)、db1[50,54) | VCS消费完LPacked后S5 Fixpipe写入→拼A |
| UB packed leaf | 原ResVcs有效8KiB：db0[18,26)、db1[58,66) | S3生成→拼A消费 |
| UB完整A | 原LFull16KiB：db0[34,50)、db1[74,90) | S3 NegL上传完→FP32拼A→原位压BF16前8KiB→GM |
| GM workspace | 每AIC四个16KiB槽，共64KiB | 槽[0,4)为X FP32 NZ，[8,16)为tail/a0 padded A BF16 ND |

所有区间半开，以上片上区间单位KiB。保留原UB 248KiB、L1 512KiB、L0A/B各64KiB、L0C256KiB容量边界。无分配增量。FP32 X的元素offset=t*4096；BF16 padded A的元素offset=t*8192+4096，两种view同GM基址但不重叠。

### Stage0：Vector，常量

保留VCS identity与leaf零填充。删Cube I初始化；在原512B index区域上半部以int32 Arange/Store生成0..63，避免调用不支持uint32的Arange。下半部VCS index保持。

### Stage4：Cube，X

A源为full64 BR的R11.T，Load mStart2/kStart4/mStep2/kStep4/srcStride4/dstStride2/transpose=true；B源为full64 BL的-L10，相同配置但kStart0。L0B Zn解释后为逻辑-L10。MMAD m=n=k=32、FP32 initC=true。

保留Acc(t)/Dump(t-1)节奏。X经Fixpipe ChannelSplit写GM tight32 NZ C0=8，srcStride32、dstStride256，4KiB；FIX_MTE2后以DataCopy(1,128,0,0)读入原L1 Y槽前4KiB。发布原S4 free8..11及MTE2_MTE1(t)；S5消费后者。GM写读各4KiB代替原各16KiB。

### Stage5：Cube，C10

A=X tight32，Load(0,0,2,4,2,2,false)；B=full64 TL R00.T，Load(0,0,2,4,4,2,false)，L0B Zn解释为R00。MMAD32³、FP32 initC=true。M_FIX后Fixpipe FP32 ND32×32到owner UB LPacked前4KiB，dualDstCtl0、subBlockId=t%2；src/dstStride32。

之后PIPE_FIX发布C10 ready，复用旧KKT方向的flag0..3；owner在PIPE_V等待。AIC已消费此task的S3 ready，故旧KKT token及VCS读取都已完成，不会与新通知混用。

### Stage5 A重组：Vector，A

一个VF读取packed leaf每行64个float，通过Scatter转置恢复两对角块；每次读取C10两个32元素行，Scatter到左下；同样的两行索引将零写入右上。每个FP32 A元素恰写一次。VEC_STORE→VEC_LOAD后逐行读64float，再用既有CAST_RINT cast/store64BF16，原位向前压缩。目的仅覆盖已经读取的float行，没有未初始化高半寄存器读取或C10越界。

V_MTE3后，outputA=true写公开A的M*64个元素；M<64或outputA=false再写完整padded A到私有槽后8KiB。完整块有公开A时直接供后续读取，避免重复GM输出。

同一AIV必须先完成**全部owner task的A拼接/GM写回下发**，再drain上一pack W/U，经MTE3_MTE2闭环后进入S6循环。不能逐task拼A后立即S6：db0 V[64,80)会覆盖db1仍活跃的leaf[58,66)及A[74,90)，V256冲突更大。S6数值算法保持。

### Stage6/7交接与同步

S6的PIPE_MTE3 ready4..7排在A GM写回之后；AIC改为PIPE_MTE2等待该通知，才可从GM公开A或padded A读完整64×64 BF16到L1 A。随后MTE2_MTE1(t) Set/Wait，S7读取A和既有Kbg/Vb。X与A的两轮MTE2_MTE1(t)各自闭合，不遗留旧token。

M_MTE1、FIX_M、M_FIX bank=t%2维持原credit初始/消费/释放；S4与S5每次MMAD都init，不依赖旧L0C内容。S4 free仍独立保护下一pack k'对NegL的覆盖。S3/S6复用4..7、S2/S5复用0..3均按已消费前一用途后再生产。空task不新增通知/等待。原尾pack Stage7完成通知和末pack W/U drain保持。

同核必须先完成本pack S7才会进入下一pack S4，故X/padded A槽跨pack不覆盖未读数据；旧W/U的Fixpipe在当前C10 ready之前，当前S6 ready又在旧W/U drain之后。S3 leaf上传完成后，旧LFull不再读取，允许拼A覆盖。

## 3. 数值与验证边界

保留FP32乘加及最终一次BF16 cast。若|L|≤l、|R|≤r，则|X|≤32rl、|C10|≤1024r²l；有效依赖不比原K32左下路径增加数值风险。tail的L无效行列为零，leaf补单位对角，公开输出仍仅有效行。删除结构零项不承诺NaN/Inf或硬件舍入位级等价；原ATK阈值不改。

离线核对实际描述符的非对称operand、tight32 NZ中转、所有有效长度1..64、两leaf反转置及A覆盖、BF16原位压缩、WU RHS、workspace/UB边界及同步接线；这些模型不能替代CANN/NPU验证。首次/回绕/尾pack的实际flag计数仍需设备记录。重点设备回归outputA=true/false、tail32附近、G1/2/3/4、V128/256、fixed/varlen及多pack。

沿用普通wheel构建安装及原性能脚本；详见[验证记录](validation.md)。本机无CANN/NPU，构建、设备精度/MSS/确定性、Example/ST和实际性能均待内部设备执行；私有Stage4 probe不是本轮接入或验证的前置条件。
