# <算子名> profiling 操作说明

生成实际文档前替换全部占位参数，核对目标工程命令。以下 wheel/Python 示例不是其他工程的固定要求。
环境：<产品型号、构建SoC、CANN版本、镜像Python环境>。
原用例：<公开入口、原测试语言/入口、用例文件>。
流程见 [timer_flow.md](timer_flow.md)，当前状态/结论见 [timer_report.md](timer_report.md)。

## 1. 宿主机进入已有容器

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
docker exec -it <容器名> /bin/bash
```

容器已启动、透传 NPU、含依赖及本轮源码。

## 2. 容器内激活、构建、安装

本仓 A5 示例；生成时按镜像真实激活方式、SoC 和构建脚本填写，已有依赖不重复安装。

```bash
cd <容器内仓库根目录>
# <按镜像约定激活 Python；不需要时删除该行>
source <实际CANN环境脚本>
npu-smi info
FLA_NPU_SOC=ascend950 FLA_NPU_OPS= python scripts/build_wheel.py
# 或直接复制构建末尾打印的准确安装命令。
python -m pip install --force-reinstall --no-cache-dir --no-deps 'dist/<本轮准确wheel文件名>.whl'
```

构建成功标志：<实际归档校验标志>。构建/安装/测试使用同一解释器，不按文件名排序选择 wheel。

## 3. 安装后只运行主要入口

```bash
python <原工程测试目录>/profile_performance.py --device 0
```

Python 工程示例；其他语言替换为同语言实际入口。自动检查产物/环境，执行原性能用例、同输入诊断和输出比较，导出 raw/CSV/metadata/汇总。
用例/设备选项：<实际支持选项>；成功标志：<成功消息与绝对结果目录>；失败日志：<实际文件>。

## 4. 阅读结果和可选离线操作

默认 <仓库>/profiling_results/<算子>/<run_id>/，或明确的工程既有输出路径。
先查看 <状态metadata>，再看 <逐核CSV、阶段汇总、普通/诊断比较与原性能报告>。
默认 cycles；核实频率与相关核型共同时钟后才生成 us/Perfetto。不推断跨 launch 间隔。
需要时提供至多第二条已有入口：<已保存输入重放或raw离线解析的实际命令>。
保留失败 raw，核对路由、加载版本、布局、GM 发布，不跳过 kernel/完整性检查。
验证状态：<源码/模拟/设备实测及证据>，结论集中在 timer_report。
