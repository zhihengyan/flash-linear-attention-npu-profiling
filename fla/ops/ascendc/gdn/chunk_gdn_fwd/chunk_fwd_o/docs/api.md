# ChunkFwdO 既有接口冻结记录

本次是 K32 基线上的内部实现优化，没有新增公开接口或支持组合。既有接口的签名、默认值、
参数表和调用示例归档在 [README](../README.md#2-接口定义)，本文件记录优化所依赖的边界。
接口实现依据为 `op_host/op_api/aclnn_chunk_fwd_o.h/.cpp` 及现有 Python 适配；不修改这些文件。

- 稳定入口为 `fla_npu.ops.ascendc.chunk_fwd_o`，ACLNN 两阶段入口为
  `aclnnChunkFwdOGetWorkspaceSize` / `aclnnChunkFwdO`，参数顺序和返回值保持。
- q/k 为 `[B,HK,T,K]`，v 为 `[B,HV,T,V]`，g 为 `[B,HV,T]`；`G=HV/HK` 为正整数，
  `hk=hv//G`。H 为 dense `[B,C,HV,K,V]` 或 packed `[1,total_chunks,HV,K,V]`；
  state_v_first=true 交换 H 的末两维物理语义。
- 公开支持 K=128、V=128/256、chunk_size=64/128、FP16/BF16；g 为 FP32 或相应输入 dtype。
  O 的 BNSD/BSND/TND/NTD 布局、连续化和异常行为保持，详见既有 README 与 host 校验。
- 本轮只影响 A5 优化路径：BF16、K=V=128、BT=64、G=1..4、BSND/TND 输出；
  gate 为 BF16/FP32，exp/exp2、state_v_first 两种物理状态顺序都保持。
  A2/A3 和 A5 兼容路径不变。
- varlen 的 cu_seqlens 和 chunk_indices/chunk_offsets 仍按原入口要求成对传入，B=1；
  不改变元数据的校验或 tail 的有效区规则。
- 输出数学仍为块内 gated causal QK@V 加块间 Q@H；详细运算、FP32 中间量、BF16
  A-prime 舍入和最终 scale 的实际顺序见 [设计](design.md)。

CPU 标杆和固定值域沿用 `tests/atk/chunk_fwd_o/executor_chunk_fwd_o.py`；GDN 共享调用沿用
`tests/atk/chunk_gated_delta_rule_fwd/gdn_reference.py` 与六 ACLNN 双标杆。不修改输入生成、
混合容差、冻结 JSON 或 Python/ACLNN ABI。
