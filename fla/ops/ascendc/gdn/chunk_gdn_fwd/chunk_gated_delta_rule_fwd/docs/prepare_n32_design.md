# Prepare N32 实验设计

方案设计规则版本：V2。实现分支`codex/gdn-prepare-n32`，严格从`codex/gdn-prepare-k32` / `d0cb628df7c8c23c0342392dfaf4ec910aba9947`创建。公开接口见[api.md](api.md)；数学、任务和未受影响阶段沿用[既有K32设计](prepare_k32_design.md)。用户明确授权跳过逐Stage设备验收门禁，直接完成普通调用链编码；这不表示N32设备精度通过。

Block32首版及后续分析已保留在其独立分支，分析提交445422a4。N32不继承该分支的AIV拼A、C10 ready或Stage6等待修改；本分支的Stage0/1/2/3/6/7、AIV处理、host/ABI和workspace分配以K32为准。

## 1. 目标与预算

只裁剪Stage4/5无效的输出列：Stage4保存`Y_left=Y[:,0:32]`，Stage5仍用Cube生成完整A，直接更新左32列；不增加Vector计算或等待。四次FP32 MMAD：

| 操作 | K32 M,N,K | N32 M,N,K | initC |
| --- | --- | --- | --- |
| S4 I@I预填 | 64,64,64 | 64,32,64 | true |
| S4 Left@N | 64,64,32 | 64,32,32 | false |
| S5 I@Left | 64,64,32 | 保持64,64,32 | true |
| S5 Y_left@R00 | 64,64,32 | 64,32,32 | false |

求解MAC由655360降至393216/task（-40%），Y GM写/读各16→8KiB，case0减少88.5MiB逻辑搬运。MMAD、LoadData及Fixpipe调用次数不变。该比例不等于Prepare或七项净收益。

目标SoC Ascend950PR_957c。case0：B1/HK16/HV32/T11274/K=V128/BT64，q/k/v BF16、g FP32、beta BF16，BNSD、G2、varlen、FP32初态和末态输出；177 chunks、5664 tasks、1416 packs、28 AIC/56 AIV，每AIC50或51 packs。每核最多204 tasks，原grid-stride/PackWorkId不变。case1为B2/HK=HV32/T8192/K=V128/BT64定长，同样走Prepare，需回归。

K32普通case0七项逐调用Task Duration(us)之和均值1612.33015us、Prepare620.64430us；原A为1704.80063us。沿用原ATK普通采样配置，每轮20组、重复独立进程比较；采集脚本warmup5只控制诊断重放。约2.7%对应43.53291us/总均值1568.79724us，仅作投入参考，3%和2.7%均不是硬门槛。旧MAC边际proxy在预填完全隐藏/完全暴露的两情景为31.42341/62.84681us（1.949%/3.898%），不是上下界或收益预测；保守情景还需搬运等净省12.10951us才能达到2.7%参考。实际也可能被既有重叠隐藏。

## 2. Stage与数据依赖

设`Right=diag(R00,0)`、`Left=diag(0,R11)`，`N=-L`；两leaf来自原S3，物理L1是逻辑leaf的转置。令`E=I64[:,0:32]`，则：

```text
S4: Y_left = I64 @ I64[:,0:32] + Left[:,32:64] @ N[32:64,0:32]
             [64,64] @ [64,32]    [64,32] @ [32,32] -> FP32 [64,32]
S5: A_full = I64[:,32:64] @ Left[32:64,:]       -> FP32 [64,64]
    A_full[:,0:32] += Y_left @ R00              -> FP32 [64,32] update
    A_bf16 = original Fixpipe cast(A_full)      -> original GM/L1 outputs
```

`Y_left=[I32; R11@(-L10)]`，故最终A为`[[R00,0],[-R11@L10@R00,R11]]`。S5完整初始化必须保留；不能只初始化左半区或假定S4未写的L0C右半区为零。

### 共同资源与任务

task t=0..nThis-1；owner AIV=t%2，db=(t/2)%2；G3 pack3，其余pack4，原G2 work交换不变。支持范围/flag/tail保持K32。Prepare为独立chunk阶段，方法参考CHUNK_INDEPENDENT V1 / a78efa9512dfa64c20b2a58a75e84e88a054d694，工程参考仅当前K32源码；公开H链仍有状态依赖。沿用原owner映射和既有上传边界，不宣称本轮完成R01–R21全量迁移。

地址均半开，单位KiB：

| 区域 | 绝对地址/份数 | 生命周期与覆盖条件 |
| --- | --- | --- |
| L1 Y_left | 原[0,64)，4×16KiB，每槽只用前8KiB | S4 MTE2写→S5最后一次MTE1读，保留每task独立槽 |
| L1 NegL/khat | 原[64,128)，4×16KiB | S2读取khat后S3覆盖；S4读取后原S4 free放行下一pack k' |
| L1 Right/Left | [128,192)/[192,256)，各4×16KiB | S3上传→S4/5读取，full64 NZ不改 |
| L1 A/Kbg/Vb/I | 原[256,496)，A每task8KiB，I在[480,496) | 保持原S5输出、S6上传、S7消费及下pack保护 |
| L0A/B | 普通bank0/1 [0,16)/[16,32)，P bank0/1 [32,48)/[48,64) | 预填A写16KiB/B写8KiB；S4与S5更新A写8KiB/B写4KiB，均在原槽内 |
| L0C | task t为[64t,64(t+1))，共256KiB | S4仅前8KiB有效；S5第一次完整写前16KiB，再仅更新前8KiB |
| GM Y workspace | coreBase=coreIdx*65536B，每task16KiB，共64KiB/core | 只写/读槽前8KiB，其余不使用；原host分配和四槽复用不变 |
| UB | 原K32全部区域、峰值与别名保持 | 未增加AIV计算或临时tensor，S6仍可与S4/5并行 |

不改变L2策略。新路径中每task Y写一次8KiB、读一次8KiB；A仍按原full/outputA、tail、a0路径输出，读写字节数和消费者保持。L1/UB/L0/GM分配增量为0。

### Stage4：Cube，生成Y_left

LoadData2D V2参数以下顺序为`mStart,kStart,mStep,kStep,srcStride,dstStride,transpose`；FP32源L1 NZ C0=8，row/column源起始单位分别16元素/8元素，stride为512B：

| 操作数 | 参数 | 逻辑MMAD operand / L0 packed image |
| --- | --- | --- |
| 预填A：I64 | 0,0,4,8,4,4,false | A[64,64] / NZ[64,64] |
| 预填B：I左32列 | 0,0,4,4,4,2,true | B[64,32] / NZ[32,64]按Zn解释 |
| 累加A：Left | 2,0,2,8,4,4,true | A[64,32] / NZ[64,32] |
| 累加B：N左下 | 2,0,2,4,4,2,true | B[32,32] / NZ[32,32]按Zn解释 |

预填仍只Load一次I、循环每task MMAD并隐藏在S3 ready之前；P bank的Acc(t)/Dump(t-1)节奏不变。Y_left Fixpipe到GM：m64/n32/srcStride64/dstStride512、NoQuant、ChannelSplit=true。L0C源C0=16、GM目的C0=8。GM再用DataCopy(1,256,0,0)读8KiB到L1原Y槽前半。

L1/GM地址为`(col//8)*512+row*8+col%8`，row<64、col<32，覆盖float[0,2048)；高半槽禁止被后续Load读取。保留S4 free与MTE2_MTE1(t)的发布时机/数量，不提前释放NegL。

### Stage5：Cube，完整A初始化及左半区累加

第一次仍调用K32的`SolveHalfKMatmulToL0C<false,true>`，输出64×64，initC=true。第二次：

| 操作数 | 参数 | 逻辑operand / L0 packed image |
| --- | --- | --- |
| A：Y_left | 0,0,4,4,4,4,false | A[64,32] / NZ[64,32] |
| B：Right TL | 0,0,2,4,4,2,false | R00[32,32] / R00.T NZ[32,32]按Zn解释 |

MMAD m64/n32/k32、initC=false。L0C NZ C0=16偏移`(col//16)*64*16+row*16+col%16`：左32列恰为float[0,2048)，右32列为[2048,4096)。M始终64，故较窄更新与完整A使用相同列前缀布局，右半区保留第一次生成的R11及零块。不得将C10单独32×32的布局套用到这里。

原Stage5_Dump完全保留：full+outputA经GM A回读L1；tail/a0直接L0C→L1，公开A只写M有效行；最终仍由Fixpipe转换BF16，无新增VF。原S6、S7及ProcessAic/Aiv调度保持。

### 同步与跨pack复用

不新增、移除或重排event/flag调用；新Load helper每次仍在两Load后Set/Wait MTE1_M，MMAD前Wait FIX_M。各资源原闭环：

- ProcessAic初始化M_MTE1和FIX_M两个bank credit，以及原FIX_MTE1 pack gate。
- S4预填Wait M_MTE1(0)，加载一次；逐task Wait FIX_M(bank)→MMAD→M_FIX Set/Wait→FIX_M Set，最后释放M_MTE1(0)。
- Acc Wait M_MTE1(bank)→Load ready→Wait FIX_M(bank)→MMAD→Set M_FIX/M_MTE1；Dump Wait M_FIX→Fixpipe→Set FIX_M→FIX_MTE2 Set/Wait→MTE2回读→原S4 free→Set MTE2_MTE1(t)。
- S5 Wait MTE2_MTE1(t)→Wait M_MTE1(bank)→完整初始化MMAD→原Set FIX_M→N32累加MMAD→Set M_FIX/M_MTE1→原Dump。所有调用次数与K32相同。
- 原S3 ready4..7、S4 free8..11、S6 ready4..7及S7尾保护保持，含原+16 AIV1映射、初始free、下一pack回绕和最终drain。空/partial task仍由原cur.nThis界定，不额外通知或等待。

上述只证明源码设计的资源闭环；实际动态flag计数、MSS和确定性仍需设备证据。

### 数值与验证

有效运算顺序及K32归约长度保持，裁剪的是不再消费的输出列；没有HF32、更早BF16 cast或新近似。若|N|≤l、|R|≤r，则|Y_left|≤1+32rl，|A|≤r+32(1+32rl)r，边界不高于K32。tail继续用原完整64tile padding，不按M缩小MMAD维度。IEEE非有限值与硬件归约舍入不承诺位级等价；原ATK golden、生成器及容差不改。

离线验证直接读取本轮实际描述符，核对非对称operand、L0边界/poison高半槽、矩形NZ往返、L0C右半区保留，以及全部valid1..64的A/软件BF16 W/U对K32和独立FP64模型。另对比K32源码，验证四个受影响函数以外及全部原事件序列保持，并核对普通/Profile均进入相同实现。

本机无CANN/NPU，不能执行真实编译/设备精度/性能。按用户授权完成整链接入后，内部仍沿原wheel→安装→Prepare/公开前向精度→原profile_performance.py --device 0进行验证，默认含case0/1。细节与结果写入[validation.md](validation.md)，未取得设备结果前不宣称收益或精度通过。
