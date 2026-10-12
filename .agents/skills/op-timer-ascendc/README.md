# op-timer-ascendc

为一个已有 AscendC 算子接入 Tik/Tok，复用工程原始性能用例，导出逐核阶段 CSV。先确认芯片，推荐 A5 (`ascend950`)，实际参数按工程与设备核对。一个公开入口内部的多个 kernel 属于同一任务。

流程见 [SKILL.md](SKILL.md)，实现细节见 [迁移指南](references/migration-guide.md)。保留数值、ABI、原流水与同步，普通调用不自动计时。

交付集中在算子工程 docs：timer_flow.md 为原流程和选定方案，timer_run.md 为 Docker 到一条主要采集命令，timer_report.md 为当前结论、审核/实测状态与数据链接。
新入口与原测试语言一致，真实输入/参数复用，原性能循环与诊断分开。默认 cycles，确认频率和共同时钟才生成 Perfetto。A5 优先检查 GM 发布，所有相关平台检查 OPP 初始化与构建/加载一致性。

项目级 Codex 安装示例；其他客户端使用其约定的 skill 目录：

```bash
git clone git@gitcode.com:zhihengyan/op-timer-ascendc.git .agents/skills/op-timer-ascendc
```

assets 是需按目标工程适配的模板，references 按需阅读。模板回归：

```bash
python .agents/skills/op-timer-ascendc/scripts/validate_assets.py
```

离线检查不能代替 CANN 编译和设备采集。[MIT](LICENSE) © 2026 Send_Tanks。
