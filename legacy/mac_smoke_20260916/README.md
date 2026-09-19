# 早期 GPT-assisted smoke test 原始工具

这些来自 2026-09-16 用户回传的完整 Mac smoke test 工程，用于解释 GPT 看图产生提示后怎样驱动 SAM、修复边缘并交付审核。它们保留固定 420 图数据布局，**不是新 RoboTwin 的入口**；原 420 全包、SAM Small 权重和 Mac 环境未复制。

- `tools/assist_masks.py`：读取人工 seed 与 `work/segmentation/extra_seeds.json`，SAM2.1 Small 传播。包括 Mac MPS 的 FP32 attention 兼容处理。
- `tools/refine_frames.py`：按原有局部 crop 与正负点修复单帧，提示见 `refinement_prompts.json`。
- `tools/build_proposals.py`：合并候选和场景可见性，保留原人工优先权。`scene_decisions.json` 的具体区间只属于原样本。
- `tools/review_masks.py`、`validate_proposals.py`：生成查看产物、检查候选与源标签。
- `handoffs/*确认*.json`：原用户对 4 个疑难不可见状态及剩余输出的实际确认范围，说明“何时才可改 human_confirmed”。它们不授权确认新的样本。

历史 Small/MPS 程序使用 multi-object non-overlap；当前服务器 base+ 流程保留 object 重叠、不做 robot subtraction。不能混用两版配置后声称复现同一结果。首轮新服务器使用根 README 的便携入口。
