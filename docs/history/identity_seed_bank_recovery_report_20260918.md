# Seed Bank + causal SAM2 recovery experiment

日期：2026-09-18。根目录：`annotations/identity_seed_bank_recovery_20260918_r2`。

## 结论

本轮完成小规模四组对照，**工程校验 PASS，当前自动恢复方案的质量门槛 FAIL**，不扩大到 5,803 episodes。种子库保留，不立即批量增加 exemplar。
重新检测能修复部分错误 track，但错误候选也会重新引入身份混淆和机器人污染。
目前不能将这些自动输出作为已验收 reconstruction pseudo-GT。

## 对照与范围

| 模式 | 初始化选择 | 后续自动重新检测/播种 |
|---|---|---|
| single_detector | detector score | 无 |
| single_bank | masked DINOv2 bank 排序 | 无 |
| recovery_detector | detector score | 有 |
| recovery_bank | masked DINOv2 bank 排序 | 有 |

- 21 段、1,731 camera-frames、三路相机；task1/9/24。
- 4 个不在 bank 或 accepted420/87 来源中的 episode，各三路相机、每段 121 帧；另 9 段每段 31 帧用于评分。不是完整 episode 长时遮挡测试。
- 评分仅使用 9 个未作 seed 的已审核画面，共 24 个 object-frame 标签：16 visible、8 absent。新的 4 个 episode 只有定性 QA，没有新人工 GT。
- bank 查询排除整个 query episode。该困难开发集已用于先前实验，不能当作独立最终测试集或全库准确率。
- 每 5 帧检查；常规重检测间隔 15 帧；两个 recovery 分支异常触发取并集，共享检测时刻和完全相同的候选池。候选先通过 detector >=0.30，再排序；bank 没有附加严格拒绝门限。
- candidate 与 track mask IoU <0.25 或严重时序异常时，移除该 object 的旧状态，再以新 mask 初始化。其他 objects 的状态保留。
- 连续检测缺失和低存在/空 mask 等联合证据可终止 track；unknown 不被伪造为已确认出画。robot overlap 仅警告，绝不执行 object-mask 减 robot-mask。
- 禁止评分帧重新播种；推理不读取人工 mask。实际 anchor 在 `checks.json` / `events.json`；config 继承的 `seed_frames` 是父实验元数据，不是本轮恢复时刻。
- frozen robot layer 仅 QA，本轮没有重新评估左右臂/EE 的身份与法兰盘边界。

## 定量结果

结果以 `scoring.json` / `summary.csv` / `object_frame_scores.csv` 为准。

| 模式 | visible mean IoU | Dice | IoU>=0.5 | 空 mask 漏检 | absent 误报 |
|---|---:|---:|---:|---:|---:|
| single_detector | 0.30759 | 0.32456 | 5/16 | 2/16 | 5/8 |
| single_bank | 0.30756 | 0.32451 | 5/16 | 2/16 | 5/8 |
| recovery_detector | 0.21401 | 0.24613 | 3/16 | 0/16 | 8/8 |
| recovery_bank | 0.26022 | 0.28594 | 4/16 | 0/16 | 8/8 |

| camera | visible/absent 标签 | single_detector IoU | single_bank IoU | recovery_detector IoU | recovery_bank IoU |
|---|---:|---:|---:|---:|---:|
| head | 7/1 | 0.55754 | 0.55748 | 0.34588 | 0.45150 |
| left_wrist | 7/1 | 0.14552 | 0.14552 | 0.11631 | 0.11631 |
| right_wrist | 2/6 | 0.00000 | 0.00000 | 0.09445 | 0.09445 |

bank 对比 detector-only recovery 的平均 IoU 高 0.04621，但低于 single_bank 0.04734。
右腕出现非空 mask 不能算恢复成功：两条可见标签均未达到 IoU 0.5，6 个 absent 标签全部误报。
简单二值 visibility accuracy 为 single 17/24=70.8%，recovery 16/24=66.7%；错误身份仍可能碰巧判对“存在”，不能替代 mask/identity 指标。
这只说明当前短片段恢复策略发生回退，不否定经助手确认的 multi-keyframe seeds，也不代表旧全量结果的整体准确率。

IoU/Dice 是 GT-visible 标签的逐 object-frame 宏平均，不是像素池化；未找到目标按 0 分处理。
absent FP 指已审核不可见目标仍被预测为非空 mask。visible 仅是 mask 非空预测，不代表身份正确。

## 已实际检查的失败与改善

助手实际查看 12 张六宫格截图，清单及具体观察保存在 `assistant_visual_review.json`。没有声称人工逐帧审核全部输出。

- `new_T1_left_head/188`：两种 recovery 比 single-start 更贴近水壶本体；128 帧仍有水壶被跟成机械臂的错误。
- `new_T9_right_head/223`：从篮内水果纠回水壶，但 mask 仍包含夹爪/前臂。身份改善不等于边界合格。
- `new_T24_left_head/588`：bank 帮助区分部分西瓜/桃子候选；其他水果仍会混淆。
- `new_T24_left_left_wrist/588`：重新找到画面边缘篮子，但水果仍被跟到餐具/机器人上。
- `new_T24_left_right_wrist/588` 和 `new_T24_right_head/225`：恢复新增机械臂被当成水果的问题；不能无条件接受新候选。
- `score_T24_left_wrist/360`：近景、运动模糊、裁切下，多种水果身份覆盖篮子或夹爪。`score_T24_right_wrist/363` 同样出现恢复后假阳性增多。
- task9 wrist：水壶夹爪粘连、砧板误落到篮子/背景，没有被 bank 自动解决。

## 为什么暂时不扩大 bank

补充诊断复用旧 pilot 在这 9 张评分图片上的候选缓存，仅事后分析；**这些评分帧候选从未提供给本轮传播作为 seed**。

- 16 个可见目标中，13 个已有 IoU>=0.5 的候选；detector Top1 选中 10 个，bank Top1 选中 12 个。
- 可选候选的 oracle mean IoU 为 0.73836；这是事后使用 GT 选最好候选的上限，不是模型成绩。
- task1 左腕水壶、task24 左腕牛油果、task24 右腕篮子缺少合格候选。只增加 embedding exemplar 不会凭空产生正确轮廓。
- task24 左腕桃子存在好的候选但仍排错，值得定向检查身份及背景证据，而不是同类重复采样。

因此先改善 candidate admission：显式记录不确定、候选间身份竞争、机器人污染提示与短时确认；异常不自动等于新候选可靠。必要时由助手查看局部 RGB 上下文并提供 SAM2 点/框 seed，保留 assisted provenance。不要把 robot overlap 当成硬减法或单独拒绝抓取物体的理由。

## 恢复统计的限制

两组 recovery 各有 63 个初始 object seeds；detector 后续重新播种 584 次，bank 510 次。
本轮真实数据没有触发 termination 或 reacquire，所有初始目标都得到候选，其中包括错误候选。
这说明现有假阳性可能阻止“缺失检测”计数积累；不是证明所有物体始终可见。
两条分支的终止与重入逻辑目前通过单元测试，但尚未在本轮真实片段验证成功。
无稠密身份/出入画 GT，不能报告真实 ID-switch rate、false-positive persistence 时长或 re-entry success rate。

## 工程验证与资源

- GroundingDINO 官方 SwinT + SAM2.1 base+ + frozen DINOv2 ViT-S14；权重和代码 hash 在 `run_config.json`。
- 首次尝试的 DINO CUDA 算子不支持外层 BF16 autocast；失败记录保留。r2 将 DINO 的 FP32 与 SAM2 的 BF16 作用域分开。
- GPU2/3/4/5 四个独立 worker 已完成退出，GPU1 未使用。峰值 Torch allocated 2.812 GiB/worker，不等于驱动总占用。
- 所有 case 计时加总 2,163.12 秒，包含每 case 四个分支/共享检测；不含所有进程初始化和 QA 导出，也不是单分支性能或全库 ETA。
- 独立 validator 验证 21,852 行逐帧对象输出、候选选择复算、17,313 次引用的 query-episode 排除、mask/bbox/时间戳、robot 来源 hash 和 causal provenance。
- 63 个单元测试 PASS；两套结构/来源 validator PASS；21 段 QA 视频全帧解码通过，共 1,731 视频帧，全部有限任务已结束。
- 自动结构通过不等于语义验收；原始数据、accepted masks、bank 以及旧全量生产输出均未改写。

## 交付与复现

本地下载 `deliverables/astribot_seed_bank_recovery_qa_20260918.zip`。无需网页、GPU 或安装软件，解压后观看 `qa/*/comparison.mp4`。
六宫格：上排 RGB / frozen robot QA / single_detector；下排 single_bank / recovery_detector / recovery_bank。
视频中的 confidence 是 SAM 存在分数，不是身份正确概率。所有新 mask 是未验收自动草稿。

代码在 `tools/object_annotation/seed_bank_recovery.py`，独立审计入口 `validate_seed_bank_recovery.py`。
服务端环境：`<ANNOTATION_ENV>/bin/python`。
原始 RLE、候选检索记录、事件、checks、完整 provenance 留在本轮根目录；包内有配置、指标、事件、报告、截图、21 段 MP4 和主要实验脚本。脚本依赖现有服务器工程，不是独立安装包。

下一轮应先做候选接纳/拒绝和少量助手纠正 keyframe 的消融，再用较长遮挡/重入片段验证。暂不请求新增人工描边，不扩 bank，不做全库重跑。
