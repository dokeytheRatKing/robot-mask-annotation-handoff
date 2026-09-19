# Mask 标注回传验收与辅助标注试验

日期：2026-09-17。项目根目录：`<SOURCE_WORKSPACE>`。

## 结论

用户已复核的 **420 帧 / 1,860 个 object-frame 标签全部接收为参考真值**，不再要求重新标注。
本地项目的多 mask 审核界面和修复已并入服务器工具。采用
“GPT 看 RGB/上下文确认物体身份与位置 → 少量 mask/点/框引导 SAM2 → 检查叠加图并局部修正”
作为后续小批量标注主流程；GroundingDINO 保留为候选生成器，不把反复高分检测等同于身份正确。

已完成旧方法的完整评估及三路困难片段的新试验。新试验非种子帧 IoU 为
head **0.9563**、left_wrist **0.9791**、right_wrist **0.9413**。
尚未启动全量标注、temporal grounding、reconstruction 或训练。

## 回传完整性与落盘

- 原 ZIP：`deliverables/astribot_smoke_test_full_project_20260916.zip`，保持不变。
- SHA256：`84e726c699d374bac64e988d3b28908d4fa2aa337e84473da829f5144aa0ab3f`。
- 原样解包：`incoming/mask_audit_returns/local_complete_20260916/astribot_smoke_test_full_project_20260916/`。
- 包内验证器检查 7,577 个文件 SHA256、420 张图片、全部标签和修订历史：PASS。
- 78 个原始手工标签逐字节保留；1,782 个辅助标签经过用户复核；待复核为 0。
- 1,141 个非空 mask，719 个空 mask。visibility：visible 322、partial_occlusion 819、fully_occluded 44、out_of_view 675。
- 参考集：`annotations/mask_audit_confirmed_20260917/`；`import_report.json` 保存导入检查及每个标签的哈希。
- 统一导出：上述目录的 `objects/task_XX/episode_XXXXXX/{camera}.jsonl`，共 9 个流。
- mask 使用原分辨率 COCO RLE，bbox 从 mask 推导。人工参考没有模型置信度，`confidence=null`，不伪造 1.0。
- `prediction_prefill`、`proposal_provenance` 和 1,870 条修订记录保留。兼容字段 `manual_pixel_editor` 不意味着所有像素都由鼠标绘制。

范围：21 个抽样片段，三路相机各 140 帧；task24/episode002478 为 300 帧，
task1/episode001608 为 60 帧，task24 对称版本 episode002577 为 60 帧。

## 旧冻结方案：完整 420 帧评估

输出：`annotations/mask_quality_metrics_confirmed_20260917/`，含 JSON、CSV、Markdown 和逐事件详情。
baseline 和 recovery 都是已冻结的因果自动种子试验，不是用户之前认可的人工辅助双向传播结果。

| 方案 | 相机 | mask IoU | Dice | 二值 visibility 正确率 | ID switch | 重新入画成功/可评事件 |
|---|---|---:|---:|---:|---:|---:|
| baseline | head | 0.8761 | 0.9079 | 94.35% | 0 | 0/1 |
| baseline | left_wrist | 0.1710 | 0.1954 | 37.42% | 0 | 0/7 |
| baseline | right_wrist | 0.2087 | 0.2186 | 56.94% | 0 | 0/4 |
| recovery | head | 0.8761 | 0.9079 | 94.35% | 0 | 0/1 |
| recovery | left_wrist | 0.3293 | 0.3606 | 31.45% | 3 | 1/7 |
| recovery | right_wrist | 0.4924 | 0.5124 | 57.10% | 0 | 0/4 |

IoU/Dice 按真实可见/部分遮挡的 object-frame 平均，漏检记 0，不把出画空 mask 纳入 IoU。
visibility 将可见/部分遮挡合并为可见；两种旧方法各有 487 个未初始化的 unknown 预测，记为错误并单独统计。

旧 recovery 虽改善可见物体的平均 IoU，却恶化了不存在物体的误检：719 个不可见标签上的误检率
从 23.23% 升至 31.57%；675 个出画标签上的残留率从 22.81% 升至 30.96%。
左腕最长错误持续 20 个抽样帧，观测跨度约 3.168 秒。该跨度是抽样片段下界，不是全视频错误生命周期。
左腕 recovery 的 3 次 ID switch / 86 次身份比较，且匹配后的语义身份正确率仅 59.79%；
baseline 只有 38 次成功实例匹配，不能因为其 switch=0 就认为身份可靠。

典型问题仍为左腕西瓜与桃子混淆、右腕把对侧机械臂当成牛油果，以及抓取接触时的边界污染。
因此不把旧 recovery v1 直接推广为全量自动 pseudo-GT。

## 新试验：看图确认身份 + 稀疏 mask 引导

输出：`annotations/mask_assisted_sparse_seed_20260917/`。
配置：`tools/object_annotation/config/assisted_seed_pilot_20260917.json`。
同一官方 SAM2.1 base-plus checkpoint，logit > 0，BF16，保留物体间 mask 重叠，不做 robot subtraction。
本地回传项目使用 Small；本轮刻意保持服务器原来的 base-plus，避免混入模型大小变化。

| 相机 | 源帧区间（stride 5） | 参考 UI 种子序号 | 非种子帧数 |
|---|---|---|---:|
| head | 850–945 | 61、76 | 18 |
| left_wrist | 420–515 | 143、153 | 18 |
| right_wrist | 470–565 | 242、254 | 18 |

共 60 张图；只读取 6 张种子图对应的 30 个已确认标签用于推理。
左腕片段中香蕉、桃子、牛油果均经 RGB/上下文确认无可见像素，明确配置为空；不是凭空 mask 推断出画。
SAM2 独立正反向传播；其余参考标签只在推理完成后的独立评分器中读取。
所有方法统一排除 6 张种子图，评价 **54 张图 / 270 个 object-frame**。

| 相机 | 旧 baseline IoU | 旧 recovery IoU | 新辅助 IoU | 新辅助 Dice | 可见标签数 |
|---|---:|---:|---:|---:|---:|
| head | 0.9295 | 0.9295 | **0.9563** | 0.9762 | 90 |
| left_wrist | 0.2828 | 0.6664 | **0.9791** | 0.9891 | 28 |
| right_wrist | 0.3430 | 0.7204 | **0.9413** | 0.9664 | 82 |

新方案在该子集的二值 visibility 为 270/270；ID switch 为 0/169 次比较；
62 个出画标签残留为 0，70 个不可见标签误检为 0。抓取接触 IoU 0.9463、腕部近景 0.9677。
新子集没有满足完整观察窗口的重新入画事件，**恢复成功率仍为 N/A**，不能写成 100%。

这是有已复核稀疏种子、视觉上下文及未来帧的离线辅助标注试验，不是未见 episode 的无人干预识别 benchmark。
其意义是证明少量正确的语义/像素引导可以明显减少后续修正工作。
新生成的 270 个预测保存在 `audit/proposed_labels/`，不会覆盖参考真值或冒充新的用户确认。

性能：GPU 1 的三个片段合计 SAM2 推理 10.36 秒，约 5.79 图/秒，
PyTorch 峰值 allocated 0.958 GiB。该计时不含模型加载、视觉审查和 QA 导出，不能直接推算全 30h 工期。
GPU 作业已结束并释放。本轮三个视频只比较 object mask；已有 robot mask 保持独立，未作为经用户验收的 robot GT。

## 网页与可下载产物

服务器审核器 `tools/object_annotation/audit_server.py` / `audit_ui.html` 已同步本地改进：
默认显示全部彩色物体 mask、按物体编辑、待复核导航、保存保留来源、加载失败禁止保存，
并补上手机端场景标签换行。既有批量验收记录和一次性脚本完整保留；不自动替用户验收未来预测。

当前服务：tmux `mask-review-confirmed-20260917`，仅监听 `127.0.0.1:8766`。
Mac 本地执行后访问 `http://127.0.0.1:18766`：

```bash
ssh -N -L 18766:127.0.0.1:8766 <SOURCE_SSH_ALIAS>
```

不用网页也可以下载 `deliverables/astribot_mask_assisted_review_20260917.zip`，
解压后打开 `index.html`，三路 MP4、叠加图、JSONL、指标和本报告均在包内，不需要 Python/模型/公网服务。
这是查看包，不需要重做标注。视频 `p` 为 SAM2 presence score，不是标注准确率。

## 复现与验收

沿用已安装的 `astribot-annotation` 环境（Torch 2.5.1/CUDA 12.4）。完整安装说明见
`tools/object_annotation/README.md` 和 `MASKS.md`。新试验不需要下载模型或更改训练环境。
以下输出路径必须是新目录，先检查 GPU 占用后再设置设备号：

```bash
conda activate astribot-annotation
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python tools/object_annotation/assisted_seed_pilot.py \
  --audit annotations/mask_audit_confirmed_20260917 \
  --config tools/object_annotation/config/assisted_seed_pilot_20260917.json \
  --output annotations/NEW_ASSISTED_RUN
python tools/object_annotation/score_assisted_pilot.py \
  --reference annotations/mask_audit_confirmed_20260917 \
  --pilot annotations/NEW_ASSISTED_RUN \
  --frozen-run annotations/tracking_recovery_pilot_20260916 \
  --output annotations/NEW_ASSISTED_RUN/evaluation
```

`seed_manifest.json`、`run_config.json` 和 metrics provenance 保存种子、checkpoint、脚本和标签哈希。
24 个单元测试及真实 Chromium 合成标注测试通过；真实参考集桌面/手机共 8 页只读验收通过，
1,860 份标签和修订历史哈希均未改变。没有修改原始 RGB/HDF5、现有训练数据或模型 checkpoint。

## 后续边界

已验收的 420 帧 object masks 可以作为候选像素监督和固定参考集；不因此声称 robot mask 已通过验收。
后续先在少量新 episode 上复用视觉引导流程，保留疑难帧局部重 seed、身份和可见性复查，
减少人工参与，仅把无法可靠判定的个别情况交给用户。暂不扩展全库或开展 reconstruction。
