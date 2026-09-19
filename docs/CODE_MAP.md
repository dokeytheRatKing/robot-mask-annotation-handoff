# 代码导航与依赖边界

当前服务器 `tools/object_annotation/` 的 134 个源文件/配置/说明全部交付，逐文件见 [SOURCE_FILES.md](SOURCE_FILES.md)。原 Python/JSON/HTML 保持字节一致；旧文档只脱敏源服务器路径，变换记入 SOURCE_SNAPSHOT.json。不要逐个运行所有 dated script 来寻找入口。

## 接收机可直接使用的便携入口

| 入口 | 用途 | 依赖 / 输入 |
| --- | --- | --- |
| `tools/handoff/verify_bundle.py` | 哈希和源码语法检查 | Python 标准库 |
| `example_check.py` | 真实 RLE/图像/ROI 复核，生成 HTML | requirements-cpu；无需 GPU |
| `copy_audit_demo.py` | 拷贝三张真实审核例到可写目录 | 标准库；后续原 audit_server.py |
| `inspect_robotwin.py` | 少量 HDF5 结构只读 inventory | h5py；实际数据 |
| `extract_hdf5_clip.py` | 显式 RGB key 的短片段提取 | h5py/OpenCV/PIL；不会猜 action shift |
| `prepare_point_seeds.py` | 原图点框 → SAM image mask 候选 | SAM2/GPU；显式 prompts JSON，输出待复看 |
| `propagate_clip.py` | 已看过的正 RLE seed → bounded SAM2 draft | SAM2/GPU；clip.json、seeds.json；有 CPU dry-run |

便携 GPU wrapper 是此次移植薄层，核心用官方 SAM2 和已有 nearest-seed 片段逻辑。源机 GPU 满载，因此没有新增 GPU 运行；接收机必须先做一个真实 clip 验收。空 seed、unknown interval 和复杂多对象修复仍使用下面的已执行工程逻辑，不能把正 seed runner 当全库成品。

`prepare_point_seeds.py` 输入例：

```json
{
  "clip_id": "实际 clip_id",
  "coordinate_system": "source_image_pixels",
  "seeds": [{
    "local_frame_idx": 0,
    "object_id": 1,
    "source_image_sha256": "对应原图 SHA256",
    "points": [[100, 120], [140, 120]],
    "point_labels": [1, 0],
    "box_xyxy": [80, 90, 130, 160]
  }]
}
```

这里只展示结构，坐标不适用于任何交付图。候选输出 `seeds.proposed.json` 的 `visual_review=false`；实际查看生成的 seed PNG、必要修复后，另存 reviewed copy 并记录 reviewed images/notes，才交给 propagate。assistant 的 visual_review 不等于 human_confirmed。

## 现有工程按职责复用

| 代码 | 作用 | 移植注意 |
| --- | --- | --- |
| `data.py`、`annotate.py`、`detector.py`、`diagnostics.py` | manifest/RGB/时间；task-conditioned DINO；异常线索 | 旧 data.scan 要三路指定相机；用新 manifest adapter 接 RoboTwin |
| `masks.py`、`audit_rle.py` | COCO RLE、bbox、面积、overlay | 编码与几何函数通用；旧 overlay 可能假设非 null confidence |
| `sam2_refine.py`、`assisted_seed_pilot.py` | verified seed + SAM 传播 | 调用逻辑可复用，原配置/参考 audit 是固定 Astribot |
| `robot_parts.py`、`robot_parts_probe.py` | 解剖学层、robot 内部冲突、参考定位 | Astribot physical side 与 flange seeds 不能搬到 RoboTwin |
| `identity_guard.py`、`seed_bank.py`、`seed_bank_pilot.py` | frozen DINOv2/crop/ranking | bank.build 依赖原 accepted 420/87；SeedBank.query 可复用接口 |
| `curate_seed_bank.py`、`seed_bank_review.py`、`check_seed_bank_review.py` | hash-bound exemplar 审核与展示 | curate 配置只适用于指定原 bank hash |
| `seed_bank_transfer*.py`、`seed_bank_recovery.py`、`seed_admission_pilot.py` | 固定候选/seed/恢复对照、保留失败 | dated bounded tests，不是通用生产调度；报告并无总体增益 |
| `full_segmentation.py`、`full_queue.py` | worker/SQLite queue、attempt、恢复/停止 | 调度可复用，初始化 gallery/config 含原项目假设 |
| `review_full_segmentation.py`、`build_assisted_review_pack.py` | 触发队列、分组查看、疑难片段 | 触发数不是错误数，pending 不被重排而自动清零 |
| `reseed_window_objects.py`、`semantic_interval_repair.py` | object×时间窗口局部替换 | 固定样例 load_parent 需换；保留原其他对象、hash、边界 |
| `refine_semantic_boundaries.py`、`finalize_semantic_seams.py` | 清除旧残影、检查拼接 | 不把 unknown 改成 absent；保存 r1/r2/r3 |
| `full_episode_reentry.py`、`multitask_reseed_canary.py` | 长片段/多任务 canary | 原 episodes/frames 固定，复用设计不复制样本 ID |
| `calibration64.py`、`repair_calibration64.py` | 固定训练 clip、精确 seed、多关键帧修复 | 完整再生依赖 Astribot full data 与 accepted 标签；不要在新库运行 prepare |
| `calibration64_release.py`、`calibration64_mask_adapter.py` | reviewed mask→ROI/known、release、原生支持 pooling | release 假设逐 clip 显式审核；pooling 本身不猜 VAE stride |
| `audit_server.py`、`audit_ui.html`、`portable_start.py` | 本地多对象 mask 编辑/审核 | 标准库 HTTP；另存 editable copy，不修改随包参考 |
| `build_portable_audit.py`、`export_audit_labels.py`、`import_*` | ZIP、回传、hash-bound 导入、修订 | 导入必须绑定源 hash；原批次确认不能批准新结果 |
| `validate_*`、`test_*`、`check_*` | 工程、边界、UI、已执行实验检查 | 许多 validator 默认指向原 dated run；测试不代表语义质量 |

## 最初 Mac 工程

`legacy/mac_smoke_20260916/tools/` 含最初 `assist_masks.py`、`refine_frames.py`、`build_proposals.py` 与浏览器/交付验证；点框、crop 和场景决策在其 `work/segmentation/`。这是 GPT-in-the-loop 的实际提示与执行证据；保留旧 layout 与 Small/MPS 配置，不作为新的服务后台。

## 没有打包的依赖数据

完整原始 RGB/HDF5/LeRobot，420/87 全量审核包，265 项完整 bank/embedding/gallery，robot references，full queue draft，所有 dated pilot 的完整 raw outputs，64 全包与 5,803 episodes 均留在源服务器。随包 3+16 张 RGB 和失败例足以检查 schema、看图和接线，不能重算全部报告指标。

这些缺失不是隐式下载任务。接收方应以 RoboTwin 的真实数据建立自己的 adapter/registry/seed，逐步复用模块，不为复现旧进度先复制几十小时真机数据。
