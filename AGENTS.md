# 接手 Codex 的工作规则

这是独立标注交接仓库。用户要求在另一台服务器准备 RoboTwin 的 task-related masks，同时原服务器继续 Astribot 真机数据实验。本任务不包含启动策略训练或物理机器人。

先读 `README.md`、`docs/STATUS.md`、`docs/ANNOTATION_LOGIC.md`、`docs/ROBOTWIN.md`、`docs/NEXT_AGENT.md`。历史报告的“GPU1 禁用”、旧数据路径、旧待办和已被替代的流程属于历史机器；不能在新机器套用。实际代码、数据和当前用户要求优先。

- 不修改原始 HDF5/RGB、源 manifest、已确认标签或旧运行结果。新 adapter、运行、修复与释放均使用新路径和来源哈希。
- 不把 Astribot 的 task ID、物体 ID、相机名、20D action 或 robot anatomy 默认映射到 RoboTwin。先读真实数据/仿真配置。
- 先核查仿真原始 ID segmentation 和 actor/link 语义映射；彩色可视化图或 JPEG 不能直接作为无损 ID 标签。
- GPT/Codex 必须实际查看 RGB、上下文和结果叠加才记录 visual_review。只运行代码或读取模型分数不能写“已看图”。
- 只有用户明确确认范围内的标签可记 `human_confirmed=true`；传播输出不继承 seed 的人工确认。assistant 复核可以支持带来源的 pseudo-label 发布。
- 模型未输出 mask 不是确认不可见。`unknown` 不变成负标签。不能把一个空 seed 扩成整段物体不在场。
- object 与 robot 层独立。禁止从 object mask 扣除 robot mask。机器人左右按物理身份确定；未知边界保留 unknown。
- 新模型调用前检查 GPU 归属和可用显存。用 tmux 和新服务器的数据盘运行，不使用本仓库历史 GPU 编号。限制 CPU 线程和预处理并发。
- 先做单 clip 数据/坐标/RLE/seed/非 seed 检查，再做少量跨 task/camera canary。不因候选生成完成或高 coverage 自动释放全库。
- 疑难帧尽量由上下文复看、局部正负点或重播种解决。需要人工时提供已去重的原图/叠加、具体歧义与选项；不要先派大量手工逐像素任务。
- 给每次运行记录版本、hash、case/task/episode/camera、source/local frame、timestamp、seed 来源、unknown、修复窗口、GPU 时间和 QA 范围。seed 帧不用于独立质量评分。
- 先复用已提供代码；不新增 GPT API 服务、在线训练模型或新的检测 backbone，除非实际故障或当前用户明确要求。

完成实质工作后更新 `docs/STATUS.md` 和本机另建的 `memory/WORKING.md`；不要重写 `docs/history/` 和 `SOURCE_SNAPSHOT.json` 中的原始证据。
