# 交接状态：2026-09-19

本仓库为了在另一台机器开始 RoboTwin 标注而导出。原服务器继续 LingBot-VA 标准 Astribot 后训练和 ACEG 实验；这里不需要重现训练或等待其结束。

## 已经做到的部分

- **420 张图 / 1,860 个 object-frame 标签已由用户确认**：78 条初始人工记录和 1,782 条辅助补齐经复核。四项不清楚的不可见状态由用户判断后才确认；现有确认不授权自动确认新数据。
- 后续 **87 张补充复核图**用于覆盖更多任务/robot parts；已有像素与确认历史保留。人工绘制、浏览、assistant 复核和标签条数分开计量，不把 420+87 都称作从零手绘量。
- 稀疏正确 mask/点框 + SAM2 在有限困难片段显著优于原自动种子，但依赖已查看的任务、种子与上下文，不能外推全部语料精度。
- Seed Bank 的 265 个候选 exemplar 经 assistant 检查，排除 24 个不合适项，用户接受 curated 决策。**下游 transfer 没有证明整体传播改善**。cosine/margin 只作为排序和复核线索。
- 多关键帧局部修复已实现。长 episode 暴露香蕉/夹爪污染、空 seed 抑制新出现物体、出画后串到杯架等错误；保留 UNKNOWN 和修复拼接审计。
- 多任务 canary：task5/13/27、9 个窗口、7 个 episodes。固定参考上的 visible mean IoU 旧/单 seed/双 seed 为 0.4135/0.7288/0.8323；双 seed 仍有 2 个参考定义的假阳性。评分后修复与固定分数分开，不用修复过的检查帧伪装独立测试。
- **64-clip 校准包已交付**：64 个训练 episodes、29 tasks、3,456 个单相机 source frames；48 C_fit / 16 C_diag / H16。22 个 assistant object-frame seeds 修复 12 clips，未新增人工任务。777 帧仍有部分对象未决，4 帧无可靠 ROI，全部保留。这是已检查范围内的 assisted pseudo-labels，不是全像素 human GT。
- DiT4DiT 已用该 64 包完成真实 VJP/离线方向扰动；当前 Q 分半稳定性未过，未进入方法训练。这不改变标注的可见性和 provenance 规则。

## 还没有做到的部分

- 30h Astribot 全库没有被验收为可靠 pseudo-GT。历史 full queue 的自动结果是 draft，不是当前推荐的无监督生产方案。
- 本仓库未拿到实际 RoboTwin 数据；未核实其版本、相机对应、task/actor 映射、RGB/segmentation 压缩或 checkpoint 数据划分。
- 没有自动 GPT API 标注服务，没有把 VLM 判断作为人工授权的机制。
- 没有证明 identity gate 可单独保证正确率，没有独立 RoboTwin IoU/质量结果，没有机器人执行。

## 当前交付

当前服务器标注代码完整快照、最初 Mac smoke 的辅助代码、真实小样例、失败/修复例子与新机器接入入口。案例和报告可用于理解流程；RoboTwin 自己的对象字典、seed、参考集和验收要根据实际数据建立。

读取历史报告时保留其日期：例如旧文档写“全量尚未启动”对应当时阶段，不覆盖后来“有自动 drafts、未全量验收”的状态。`confirmed_audit/manifest.json` 的旧 `PENDING_HUMAN` 是队列快照；真实确认以其哈希匹配的 `human_labels/*.json` 为准。
