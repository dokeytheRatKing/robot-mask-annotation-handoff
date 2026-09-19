# Full-Episode Assistant Recovery Pilot

Date: 2026-09-18

## 结论

多关键帧辅助恢复有明确局部收益，但**尚未通过全量 pseudo-GT 质量门槛**。
现有 Seed Bank 保持冻结，不立即扩库或重跑 5,803 条 episode。
本轮新增人工任务为 0；assistant 决定不等同于用户确认的真值。

## 范围与对照

- `episode_001662`，task1，241 帧；`episode_002503`，task24，1,178 帧。
- 每条完整运行 head、left_wrist、right_wrist，总计 4,257 camera-frames。
- 两组各 18,393 条 object/frame 记录，总计 36,786 条。不是 36,786 张不同图。
- SAM2.1 Hiera base+；现有 frozen DINOv2 masked-crop Seed Bank。
- baseline 与 recovery 使用相同首帧候选、模型及历史 assistant seeds。
  首帧候选先满足 detector >= 0.30，再按 bank 排名；不是上一轮 strict gate。
- 两组都保留 task24 head/right 在 548/598 的历史种子。因此 baseline 不是
  纯 detector-only，也不是原来的全库 production；只能比较本轮增量辅助恢复。
- recovery 新增 16 个唯一图像帧上的 50 个决定：19 positive masks、
  30 not_visible、1 uncertain。后两者在传播中均清空为 UNKNOWN，不能生成负真值。
- 正向因果传播，不用未来种子反向修改前面的帧。每 30 帧块保留同一个 SAM2 state。
- 本轮没有在每个后续触发帧自动运行新的 DINO；先生成上下文待审队列，再由
  active-session assistant 看原图给点/框，用 SAM2 image predictor 生成新 seed。
  没有后台 GPT 服务，也不能把此结果归功于全自动 bank selection。
- 使用 GPU0/2/3/4/6/7，未使用 GPU1，未终止其他用户进程。

## 可量化结果

task1 第 149 帧已有两个 accepted labels，未用作本轮 seed：

| Camera | 真值 | Baseline | Recovery |
|---|---|---|---|
| left_wrist | 水壶可见 | IoU/Dice 0 / 0 | IoU 0.88739 / Dice 0.94034 |
| right_wrist | 水壶不可见 | 错标平底锅/夹爪 | 清除错误 mask，状态 UNKNOWN |

左腕评分帧距最近的新 seed100 为 49 帧，约 1.63 秒，而非上一轮仅 5 帧。
这里只有一个可见实例、一个不可见实例，且是已反复研究的 development 数据；
不能据此宣称整体 IoU、三相机准确率、ID-switch rate 或 re-entry success rate。
task24 没有本轮可用的 accepted pixel GT，下面只做定性检查。
精确值和逐项记录见 `scoring.json`。

## 实际视觉检查

直接查看了 14 张非 seed 对比图，另检查了初始化上下文、原图和新 seed masks。

| Camera / frame | 实际观察 |
|---|---|
| task24 left270/390 | 水果与篮子恢复；工具盒、夹爪不再被当作水果 |
| left630/870/990 | 手持西瓜、牛油果、香蕉与夹爪分离；870 牛油果包含果核 |
| left1158 | 结尾不再残留 baseline 的工具盒/机器人假阳性 |
| right638/748 | seed618 后恢复香蕉、牛油果；748 已距 seed130 帧 |
| right668 | 遮挡时香蕉仍在机器人上留下错误小残影 |
| right898 | 篮内牛油果与远处被夹持的香蕉仍能找回 |
| right1008 | **两组都把杯架错当西瓜**，高 SAM presence confidence 不能识别身份漂移 |
| head0 | **香蕉包含夹爪的问题仍在**；本轮未新增头部纠正 seed |

重新入画后有真实恢复，但不能称为全程稳定：638 正确、668 有残影、748 又恢复。
不能用挑出的正确截图替代连续片段审计。种子 r1 曾漏牛油果果核，r2 增加
单独 component mask；r1 留存，r2 原图与覆盖图实际复验。
全部定性记录：`visual_review.json`，不改 accepted labels。

## 覆盖与审阅负担

| 每组 18,393 条记录 | Baseline | Recovery |
|---|---:|---:|
| 有正 mask | 15,026 | 10,547 |
| 未播种/主动清空，UNKNOWN | 1,160 | 6,114 |
| 已播种但预测为空，仍 UNKNOWN | 2,207 | 1,732 |

正 mask 减少包含清除假阳性，但 UNKNOWN 增加也意味着可能漏检；不能只报告低 FP。
同 object/reason 90 帧 cooldown、30 帧区间合并、120 帧周期身份抽查：
baseline 424 个 reason-interval 请求合成 90 张上下文；recovery 337 个合成 93 张。
这些是诊断请求，不是错误数量，不是新增人工任务。队列保持 pending，未谎称已全部审核。
目前 queue 没有收敛，重复 unknown/周期请求需要进一步去重和状态管理。

## 工程与性能

- 独立 validator 已通过全部 36,786 行：完整 frame/object 键、时间戳、RLE、
  mask-derived bbox、因果 seed 来源、模型/代码/候选/seed hash、原视频与缓存 hash。
- query episode 不进入自己的 Seed Bank 检索参考。
- 原始数据、accepted420/87、冻结 bank、全库旧标注均未修改。
- robot/arm/EE 使用原独立层做 QA，文件 hash 未变，**没有 object-minus-robot**。
  本轮没有提高 robot masks 的精度，也没有给它们新的语义验收。
- 72 项单元测试通过。结构校验 PASS 不表示 semantic acceptance。
- 6 个四宫格 MP4 全部导出并解码通过，共 4,257 帧，固定 852×692 分辨率；
  另外直接检查了导出后的头部、左右腕截图，画面和标签布局正常。
- 两组 SAM worker 时间总和：baseline 1,051.01 秒，recovery 799.59 秒；
  各组最慢 stream 425.51 / 386.98 秒；Torch peak allocated 0.940 GiB/worker。
  CPU offload 和任务/可见轨迹数量影响时间。未计 initial detection、seed 生成、
  assistant 审阅和 QA 导出，不能由此给全库完成时间或声称算法加速。
- 终端启动的 CPU 验收曾收 SIGTERM；中断产物保留在 `qa_interrupted_2143`，
  改用 tmux 重跑。中断目录不进入交付包，不影响已完成 GPU 推理结果。

## 交付与复现

实验目录：`annotations/full_episode_reentry_20260918/`。

- `qa/<episode_camera>/comparison.mp4`：左上原 RGB，右上 baseline，
  左下 recovery，右下未修改 robot parts；`spotchecks/` 为重点对比截图。
- 每 stream 的 `baseline|recovery/objects.jsonl.gz`、`events.json`、`complete.json`。
- `seeds_r2/`、`assistant_prompts_seeds_r2.json`、待审上下文及 config/hash。
- `independent_validation.json`、`validation.json`、`scoring.json`、`visual_review.json`。
- 下载包：`deliverables/astribot_full_episode_reentry_20260918.zip`。
  可直接在本地播放 MP4，无需启动网页；包内含文件级 SHA256 manifest。
  不包含全量 RGB frame cache、模型或冻结 robot 源文件；不是独立训练/推理安装包。

主要入口：`tools/object_annotation/full_episode_reentry.py`。
环境：`<ANNOTATION_ENV>/bin/python`。
分阶段 prepare/initial/seeds/run/queue/finish，输出禁止覆盖；恢复实验用 `--revision seeds_r2`。
冻结实验不可原地重跑，下一轮须新 root/revision；相关依赖已在现有项目环境中安装。
独立验收：`tools/object_annotation/validate_full_reentry.py`。

## 下一步

先在这两个 episode 的失败区间加入**可追溯的语义复检和局部修复**：
头部初始香蕉/夹爪、右腕遮挡残影、西瓜/杯架漂移。确认 seed 两侧的非 seed 帧，
并减少已审核不确定区间的重复请求；不要把整段 UNKNOWN 偷换成 absent。
随后再做多任务 canary，最后才做版本化全库选择性修复。
无需现在增加大批人工绘制或扩充整个 bank，也不启动 reconstruction/训练。
