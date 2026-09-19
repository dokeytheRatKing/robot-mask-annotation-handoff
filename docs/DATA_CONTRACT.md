# Source-pixel 标注契约

便携 `clip.json` 的 schema 为 `mask_handoff.clip.v1`。一个 clip 只属于一个 episode/camera；`frames` 按时间顺序给出 `local_frame_idx`、`source_frame_idx`、`timestamp`（允许 null）、相对 `image`、PNG SHA256。RoboTwin task/episode/camera 是显式字段，不从 Astribot 名称猜测。

RGB 文件在原图坐标。`image_shape=[height,width]`；COCO RLE 的 size 顺序相同、counts 为压缩字符串、Fortran column-major。非空 bbox 从解码 mask 得到，xyxy 的右/下界 exclusive。不使用 GPU 坐标图、UI 缩略图坐标直接写原图 mask。

| 情况 | mask / visible | 人工与训练状态 |
| --- | --- | --- |
| 精确已确认 seed | 原始 mask、明确 visibility | 只在原 seed/frame/hash 上继承 human_confirmed |
| 自动传播有像素 | RLE、predicted_visible | human_confirmed=false；默认 training_eligible=false |
| 模型输出空但语义未决 | null、visible=null、unknown | 不生成 confirmed absence |
| 看上下文确认不可见 | 空 RLE + 对应 visibility | 保存实际 review 来源和范围 |
| 身份/局部边界不可信 | 保留候选或 null；明确 unknown/excluded | 不进入可靠 ROI，不修改原 accepted mask |

便携 `seeds.json` 绑定 `clip_id` 与每个 seed 的 source image hash。支持原图 COCO RLE，或原图像素 `points` + `point_labels`（1 正 / 0 负）与可选 `box_xyxy`。坐标不能省略来源尺寸，也不能将旧案例提示直接应用到新 RoboTwin RGB。

推荐输出 `objects.jsonl`、`robot_parts.jsonl`、`frames.json`、`task_roi.npz`、review manifest、哈希、QA。历史 scripts 的单行字段略有差异，使用显式 adapter，不强制把 null cast 成 false。组合时保留 source layer 和原始 row provenance。

`known` 定义本批可靠监督支持；未知对象不产生“背景已知”的结论。经过审核的完整对象清单与不完整对象分别记录。最终 source mask 经模型自己的 crop/resize/VAE 支持映射后才可参与训练，不在标注端猜 latent 分辨率或帧组。

## 评估隔离

示例 selected seed、人审参考与独立评分是不同角色。seed 帧排除；同 episode 的临近帧也不能包装成跨 episode 泛化。质量以 visible IoU/Dice、不可见误检/残留、身份切换、重新入画、unknown/coverage 分别报告。整体标签量不是独立样本数，技术 smoke 不输出模拟的质量分数。
