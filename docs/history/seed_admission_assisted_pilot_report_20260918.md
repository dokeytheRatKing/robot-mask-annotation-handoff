# Seed 准入与助手辅助 reseeding 实验

日期：2026-09-18。结果目录：`annotations/seed_admission_pilot_20260918`。

## 结论

工程验收 PASS；助手确认 seed 的短距离传播有明显提升，但全量 pseudo-GT 质量门槛仍未通过。
**目前不扩充大规模种子库、不重跑 5,803 episodes。** 下一步优先解决新入画/重新入画的 seed 覆盖、少量身份判断错误与更长传播窗口。

已有 accepted bank、accepted420/87、原始 RGB/HDF5 和全量生产输出均未修改。
没有训练新模型，没有 temporal grounding/reconstruction，没有新增人工精标请求。

## 实验设计

- 冻结已有 bank：241 个 active exemplar；query episode 全部排除。
- SAM2.1 checkpoint SHA256：`a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`。
- 同一上一轮候选池/检测时刻，不重新运行 GroundingDINO，不根据本轮分数调阈值。
- 12 个片段、10 条 episode，覆盖 task1/9/24，642 camera-frames：head335、left93、right214。
- 9 个31帧评分窗口；另3个121帧新 episode 观察窗口。均30fps，窗口约1秒/4秒，**不是长 episode 验证**。
- 评分只用已有 accepted 标注的9帧/24个 object-frame：16 visible、8 absent。
- 助手直接检查15张非评分 RGB，生成33个正向 SAM seed；其余13项为不可见/不确定，清除为 unknown，不推定整段 absence。
- 所有33个正向 mask 均目视复核；修正篮子粘夹爪、腕部桃子局部边界共3个 RLE。r1/r2 均保留。
- 评分帧不作为 seed；助手评分 seed 在评分前5帧，**仅约0.17秒**。GT 仅用于跑完后的评分，不用于提示点/box。

| 组别 | 行为 |
|---|---|
| single_bank | 上一轮单次 bank 排序初始化，对照 |
| recovery_bank | 上一轮异常后 bank 排序自动重新 seed，对照 |
| guarded_replace | 保持对照初始化，仅约束后续 replacement；拒绝新候选不清除旧 track |
| guarded_all | 初始化和 replacement 都须通过准入，长时间未确认则清除为 unknown |
| assisted | 仅用本轮助手 RGB 提示点/box 的 SAM mask，不允许自动候选覆盖 |

自动准入固定为：detector≥0.30 的候选先参与排名；bank cosine≥0.50、竞争 margin≥0.05、SAM predicted IoU≥0.80；拒绝跨身份候选严重重叠；相邻两次有效观测相隔≤15帧且 mask IoU≥0.30。guarded_all 在检查点遇到30帧未再次确认时清除。robot mask 不相减，SAM presence score 不当作身份正确率。

## 定量结果

IoU/Dice 对全部16个 GT-visible 目标宏平均，漏检以0计入，不剔除困难项。达标指 IoU≥0.5，不等于精确 GT。

| 方法 | Mean IoU | Dice | 达标 /16 | 漏检 /16 | absent FP /8 | unknown /24 |
|---|---:|---:|---:|---:|---:|---:|
| single_bank | 0.3076 | 0.3245 | 5 | 2 | 5 | 0 |
| recovery_bank | 0.2602 | 0.2859 | 4 | 0 | 8 | 0 |
| guarded_replace | 0.3076 | 0.3245 | 5 | 2 | 5 | 0 |
| guarded_all | 0.2406 | 0.2451 | 4 | 12 | 1 | 19 |
| assisted | **0.5987** | **0.6379** | **11** | 5 | 2 | 11 |

降低误检不等于 visibility accuracy 提升：guarded_all 的19/24 unknown 和12/16漏检不可忽略；unknown 不可转成训练负标签。
guarded_replace 在评分帧与 single_bank 完全一致，说明避免了部分有害 replacement，但不能修复错误初始化。在其他非评分时刻可出现有效修复。
assisted 是额外视觉判断参与的辅助组，**不是 Seed Bank 单独带来的增益，也不是自动化准确率**。

| Camera | visible/absent | 单 seed IoU | 自动 recovery IoU | assisted IoU / Dice | assisted 达标 | assisted 漏检 / FP |
|---|---:|---:|---:|---:|---:|---:|
| head | 7 / 1 | 0.5575 | 0.4515 | 0.6274 / 0.6645 | 5/7 | 2 / 0 |
| left_wrist | 7 / 1 | 0.1455 | 0.1163 | 0.6351 / 0.6718 | 5/7 | 2 / 1 |
| right_wrist | 2 / 6 | 0.0000 | 0.0944 | 0.3708 / 0.4258 | 1/2 | 1 / 1 |

右腕只有2个 visible 样本，不可外推整库质量。全部结果属于反复检查过的小型困难 development set，不能声称独立 test accuracy。

## 观察到的改进与失败

已直接检查15张非 seed 对比图，明细在 `visual_review.json`。

1. task1 左腕壶把：助手提示将水壶从夹爪中分离，评分 IoU0.891；头部水壶 IoU0.942。新 episode 中，距 seed50帧后仍看到正确壶身分离，原自动初始化曾包含整条手臂。
2. task24 左腕：basket/banana/peach 分别达到0.887/0.807/0.895；自动分支有 basket/fruit 串轨。助手 seed 修正后明显改善。
3. task24 新右腕片段：助手避免 avocado-on-gripper、banana-on-arm；但598帧因遮挡清除后，618帧香蕉/牛油果又可见时仍无新 seed，证明**必须主动处理重新入画**。
4. 5个评分漏检均没有可用正向 seed：头部香蕉/牛油果、左腕西瓜/牛油果、右腕少量篮子像素。不能用“没有明显错误 mask”掩盖覆盖缺口。
5. 两个 absent FP 都是 task9 腕部 cutting-board 预测，与已确认的 not_visible GT 冲突。仍计为 FP，未改 GT、未在看完分数后修 seed。后续先看上下文确认候选身份。
6. 运动中西瓜边界仍有溢出：head IoU0.639，precision0.642。选对对象不保证精确边界。
7. guarded_all 仍接受一个 head 非桃子的目标为 peach；高相似度及时间一致性不能替代身份确认。

自动侧本轮19次 guarded_all seeding、2次过期清除；guarded_replace 初始35次、后续仅6次替换。过期清除是未确认超时，不是经 GT 证明的真正 out-of-view termination。没有时序 GT，不报告 ID-switch rate 或 re-entry 成功率。

## 工程验收与耗时

- 68项 annotation 单元测试全部通过。
- 独立 validator：6,225条新模式记录全部通过；验证候选准入证据、密集帧完整性、RLE/geometry、原始 timestamp/尺寸、seed 因果顺序、bank query 排除和 provenance。
- 连同两个旧对照，共10,375条记录通过；12段 MP4、642帧全部解码成功。
- 模型/代码/配置/候选/seed/复核结果均有 hash；robot 来源 hash 未改变。
- 使用GPU2/4/5，GPU1未使用。全部实验 worker 已结束。
- 三组传播合计268.68 case-seconds，Torch peak allocated1.111GiB/worker。该数包含三组共享进程且候选已缓存，**不包含检测/embedding/助手检查/QA导出成本，不可外推全量 ETA**。
- 33个 seed 仍为 assistant-assisted、`human_confirmed=false`，没有自动升级为 accepted GT。

## 下一步

先保留现有 bank，不做广泛扩容。选择少量完整 episode，保持自动候选门控，同时把未初始化、过期、新入画和身份冲突目标送入去重的助手检查队列；用时间上下文确认身份，再生成正/负点 SAM mask 并复验。
优先补完整入画事件覆盖，而非只在评分前5帧补 seed。用远离所有 seed 的已有 accepted 帧评分，并对更长非 seed 区间检查边界、串轨、机器人污染。robot parts 保留独立层；本轮未验证其新分割质量。
仅当反复出现同类 wrist/held/truncated 检索失败时，从现有 accepted 池定向补少量 exemplar。助手也不能判断的少量帧再汇总给用户；本轮无需用户画新 mask。

## 文件与复现

- 程序：`tools/object_annotation/seed_admission_pilot.py`
- 独立验收：`tools/object_annotation/validate_seed_admission.py`
- 提示记录：`tools/object_annotation/config/admission_assistant_prompts_20260918_r2.json`
- Seed及复核：结果目录 `seeds_r2/seeds.json`、`manifest.json`、`review.json`
- 定量结果：`scoring.json`；逐case事件：`events.json`
- 验收：`validation.json`、`independent_validation.json`
- QA视频：`qa/<case_id>/comparison.mp4`，无需网页。上排RGB/单seed/旧自动恢复，下排仅约束replacement/严格准入/助手提示。
- 下载：`deliverables/astribot_seed_admission_assisted_qa_20260918.zip`，附文件 SHA256 清单。

使用环境 `<ANNOTATION_ENV>`。只读重复验收命令：

```bash
cd <SOURCE_WORKSPACE>
<ANNOTATION_ENV>/bin/python tools/object_annotation/validate_seed_admission.py
```

创建/推理/QA命令拒绝覆盖已有目录。要复现实验使用新的 `--output` 路径，并记录新 seed hash 和助手复核；不要修改冻结输出以获得更高分数。
