# 给另一台服务器 Codex 的任务

可将下面内容作为新会话第一条消息：

> 阅读本仓库 AGENTS.md、README.md、docs/STATUS.md、docs/ANNOTATION_LOGIC.md 和 docs/ROBOTWIN.md。任务是把已有 Astribot 的 GPT-assisted mask 标注流程迁移到本机 RoboTwin 数据，减少人工像素编辑。先检查本机路径、GPU、RoboTwin 版本与 task/camera 配置；再检查已有仿真 ID segmentation 是否可直接使用。原始数据只读，派生结果版本化。先做 CPU 交接检查和一个单相机短 clip，实际查看 RGB/叠加、给 seed、传播、检查非 seed 与 reentry。对 unknown 保持弃权，object/robot 分层，禁止 robot subtraction。汇报小批质量范围与问题，再按当前用户指定范围推进；不要启动策略训练或物理机器人，不要复用 Astribot 类别清单/历史确认授权。

## 接收机需要落实的信息

| 项目 | 当前状态 | 获取方法 |
| --- | --- | --- |
| RoboTwin 数据 root / repo ID / revision | 未提供 | 先检查用户指定目录与数据说明，再就缺项询问 |
| 数据配置 clean/randomized、任务和 episode 范围 | 未提供 | 从生成配置/manifest 读取；不能凭文件数量猜 |
| RGB / raw segmentation / timestamp / state-action 布局 | 未核验 | `inspect_robotwin.py` 与少量图像只读检查 |
| actor/link 到 semantic object/robot part 映射 | 未提供 | 从对应任务和机器人定义读取，保存 manifest |
| 可用 GPU、数据盘、模型安装 | 以新机实时状态为准 | GPU 归属、RAM、磁盘、环境；源机编号不适用 |
| 标注规模与人工确认要求 | 小批接线先行 | 使用当前用户授权；没有规定时先给 bounded canary 结果 |

## 前三项实际交付

1. `runs/robotwin_inventory_<date>/inventory.json` 与简短 data contract：真实格式、帧/时间、camera、semantic scope、segmentation 可用性；不读取/打印大 action 数组或凭据。
2. `runs/robotwin_canary_<date>/`：固定 clips、原图hash、seed与提示、SAM/仿真 mask、object/robot 独立 JSONL、未知队列、检查图和运行环境。禁止事后只留下最好样例。
3. 本机 `docs/STATUS.md` 更新：真实做了什么、看过哪些非 seed 帧、工程/语义/人审状态、unknown 与发现的失败、后续预算。

若需要全批调度，优先复用 `full_queue.py` 的独立工作单元和停止/恢复机制；`full_segmentation.py` 含 Astribot-specific seed/robot gallery 假设，不能只替换 dataset path 后直接跑满库。先替换 adapter 与配置并测一个任务，再决定正式覆盖。

当前 GitHub 仓库只含代码和示例。不要从这里寻找 64 全包、原始 5,803 episodes 或 SAM 权重；按本机数据来源与 ENVIRONMENT.md 获取。此次任务不要求把原服务器的训练运行或所有数据复制过去。
