# Prepare Stage3–5 诊断实现与验证（2026-10-14）

基点codex/gdn-profiling：809bce6887639e831caa856b8ac3a197a6a94294。新分支codex/gdn-prepare-stage3-5-timer，独立managed worktree；原Stage6和H双head分支不改动。本轮先定位，不引入性能优化、不预报收益。沿仓库阶段03→04→05路由；完整设计/原流程/配对清单见 [设计](timer_prepare_stage35_design.md)、[流程](timer_flow.md)，复现实验见 [运行说明](timer_run.md)。

## 实现

- Prepare原操作边界追加32类、37处scope，总6固定/60动态类、232处scope。Stage3分KKT等候、后处理、叶块布局/求解/上传及原依赖；Stage4分Y输出/回读/完成等待；Stage5分A输出/回读与full/tail/no-output；Stage7现有输入/A就绪等待用于下游观测。
- 选定每核最多8pack，窗口0/16/32/48；采样槽仍32/类，独立actual/sample计数和原work ID。新raw协议v3，15384640个int64（117.38MiB）；host沿用常量尺寸校验，ctypes/公开ABI不改。共享头影响Prepare/H/O/Phase6四算子，必须完整重建。
- 原普通ATK先msprof再保存实际输入，同一snapshot做粗/细成对、十输出精确对照、严格解码及自动摘要。ordinary-only支持旧A归档wheel；reference-wheel复用此前已验证的包/OPP Python文件和内核对象身份检查。
- Python解析不外推采样，不跨核减时间戳，不将core-cycle和或嵌套scope之和当launch时延；关联依据同核work ID而不是迭代槽。独立输入任务数、每核完整必经计数、全部采样pack/task、真实A输出/尾块分支和header/capture控制一致性均校验。

## 本地验证

当前源码检查通过：基点809bce68，全部60动态类有使用、232处scope有文档锚点；Prepare去计时宏/新增词法块后业务token完全一致，原业务同步调用序列完全一致。旧公开C声明与生产Python签名不变。canonical共享计时器本身属于新增诊断逻辑，由独立审核检查其布局/采样/发布，未把它纳入业务token相等比较。

离线测试：原timer契约14、原摘要11、归档wheel10、新Stage35诊断21，共56项；最终回归结果及审核结论见本节后续记录。合成数据覆盖v3控制/错版本/actual与sample/tag、晚窗空样本、work0/错tag、同核配对、full/tail/no-output、真实4种rank4布局别名、完整计数/事件同时漏记、部分阶段actual漏记、CSV→layout→capture链、控件不一致、输入哈希/粗细配对、普通模式不调用Profile和保留旧结果。合成测试不证明设备数值或性能。

独立fresh-context审核按op-timer-ascendc Step9执行，报告见 [本轮审核](timer_prepare_stage35_audit.md)。初审发现“全部新计数/事件遗漏可伪装空尾窗”，已加入输入独立计数、每核完整计数和实际分支/控件校验，并补相应回归。

既有aclnn_abi_contract.py失败：direct BSND output源码模式缺失。独立审核确认该脚本及失败断言所读host在809bce68与当前一致，基点同样失败：旧regex匹配executorPtr)，但已有timer接入后调用为executorPtr, timerOptional)。本轮未改该脚本或host，不能把此项写为通过，也不在本轮顺带修复历史检查。公开声明与签名的定向检查通过。

## 设备验证与解释边界

最终本地回归：56/56通过（20.307s），check_timer_source通过（6固定、60动态、232scope、业务token和原同步序列一致），git diff --check通过。独立审核修复复查后本轮源码通过；详见审核文档逐新增类型的位置、关闭作用域和源码证据。设备状态仍为下方待执行项。

本机无CANN/NPU，尚未编译A5或A2/A3、安装真实wheel、验证ENABLE_TIMER=0编译、GM发布、实际kernel模板/key、普通/诊断十输出、CPU标杆精度/确定性/MSS、Example/ST和普通性能。阶段05正式设备验收未完成；新分支编码与本地检查交付不等于算子正式验收。后续按timer_run和四个受影响算子ATK README执行。

计时器局部数组及分支可能影响编译资源/普通kernel；新旧wheel普通性能对照检查此项。细节/粗窗成对比较只检查固定结束戳前的观测扰动，count/tag与DCCI均在Finish结束戳后。H/O/Phase6共享行也扩大，该共同发布增量不由Prepare粗/细差反映。必要时ENABLE_TIMER=0编译和pipe指标辅助定位。

采样控制的GM可见性也待设备验证：元数据记录timer_data_ptr、timer_words与capture_process_id。交替0/16/32/48及coarse/detail并核对全部核work ID/数量，检查实际地址复用；同一数值地址跨不同进程不能直接视为同一缓存实例。主入口每capture为独立子进程，同进程复用需在重放调用中另测。源码未提供跨launch DCache失效保证；不为此加可能覆盖新控制的初始clean，严格校验能拒绝可见窗口错误，但不替代设备缓存契约。

PREPARE_PACK包含上一pack的AIV写回，最后当前pack写回在循环外；这是循环下发窗，不是当前pack输出完成时延。S7_A_READY_WAIT只观测原Stage6输入等待之后的剩余A依赖，不能代表完整A延迟。issue窗包含原内部等待，父子窗可嵌套；不能据窗口统计外推全工作量或85us收益。

验收仍为同条件重复普通七项时间和至少下降5%：原1708.3914us需节省85.41957us，若单靠Prepare约需下降11.95%。若诊断稳定期和pipe证据不足以支持该预算，停止该候选；本轮没有提前选择求解公式、dtype或Y/A布局优化。

## 参考wheel路径校验修复

最终本地验证：13项归档wheel合同测试、全部59项诊断离线测试通过（19.264s）；git diff --check通过。路径布局兼容不降低原产物完整性或逐文件SHA256校验。

用户报告c7d2f155平台wheel被误判为缺libcust_opapi.so/内核.o。定位到采集器只识别压缩包根目录fla_npu/，漏掉本仓setup.py标记Root-Is-Purelib:false时build_py产生的&lt;dist&gt;.data/purelib/fla_npu/；scripts/check_pypi_wheel.py::_install_path已经定义了同类安装路径映射。用该实际布局的合成wheel复现相同异常，再修复采集器按pip安装路径映射，保留原ZIP成员读取和完整哈希验证。

新增3项回归，覆盖purelib/platlib通过、重定位内核字节不一致拒绝、根目录与重定位重复映射/路径越界拒绝；原10项归档wheel测试保留。只修改外部采集入口及测试/说明，不改内核、host、timer布局或wheel打包。旧c7d2f155 wheel可继续安装/引用，仅需同步新的profile_performance.py。原独立审核中的采集器指纹属于c7d2f155原版本；内核审核证据不因本次入口路径修复而改变。真实用户wheel尚未在本机读取，本轮验证为可复现的包结构离线测试，NPU实测仍待内部运行。
