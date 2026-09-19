# 第一阶段固定 64 clip 数据包

日期：2026-09-19。对应 `task_centric_world_modeling_idea_summary.md` 中 ACEG4WAM 的首批校准数据。当前工作仅交付 RGB、动作/状态、mask、固定划分和 QA；不执行模型校准、重建或方法训练。

## 固定样本

| 项目 | 实际范围 |
|---|---|
| episode / clip | 64 个不同的 episode，各取一个相机窗口 |
| task | 1–29，29 个全部覆盖 |
| 相机 | head 23 / left_wrist 20 / right_wrist 21 |
| 左右手任务版本 | left 35 / right 29 |
| C_fit / C_diag | 48 / 16，episode 不相交；C_fit 覆盖全部 29 task，C_diag 覆盖 16 task |
| H | C_fit 内预先固定的 16 clip，覆盖 16 task |
| context | 每 clip 17 行 |
| prediction horizon | 16 / 32 / 64 行，分别 22 / 21 / 21 clip |
| RGB / mask 时间覆盖 | 合计 3,456 camera-frames，保留原分辨率 |

源数据为 `datasets/lerobot/astribot_full_v21_rgb_h264`。所有 episode_index 均小于 5223，没有使用正式 held-out `[5223,5803)`。样本来自已有审核池：accepted87 中的 62 个训练 episode，加 accepted420 中两个不同的密集标注 episode。选择利用已有可靠身份种子、均衡相机和任务，并在查看本次预测前固定。**这是审核池中的有偏样本，不是全数据库的随机代表性抽样。** C_diag 用于后续校准诊断，不是模型最终测试集，也不是从未查看过 mask 的盲测集。

抽样 seed 为 20260919；manifest 同时固定每 clip 的 action/video noise seed 和四个 probe seed。原始 manifest SHA256：

```text
d355cf571469befb3120388daee648ae32d8394d992ff69c4ad0cd1b4aa509f8
```

时间以原始相机时间戳为准。全帧检查发现 21 个 clip 合计 128 处相邻相机时间戳重复，清单在独立验收报告中。例如 C051 的 right_wrist 有 16 处重复，16 行 prediction 实际跨度约 0.270282 秒，不能按 16/30 秒解释。其他常见窗口约 0.53 / 1.07 / 2.13 秒。没有删帧、重排或篡改时间戳；QA 视频统一 30 fps，仅用于审阅，不作为物理时间接口。

## Mask 制作与质量边界

复用已经确认的 object/robot seeds，使用 SAM2.1 Hiera Base+、`sam2.1_hiera_b+.yaml`、`models/sam2/sam2.1_hiera_base_plus.pt`。checkpoint hash 保存在 manifest。每个身份使用独立的近邻关键帧区间，双向离线传播；未来帧可用于制作标签，不能成为模型输入。

逐 clip 查看 64 张时间抽样 QA 联系表（364 个原始检查帧）；对 12 个 clip 进行了局部 reseeding，共新增 **22 个 object–frame seed**，由助手检查 RGB 并给 SAM2 点/框提示，再检查分割结果。没有新增人工标注请求。修复涵盖刷子、杯子、锅具、杯架、置物架、海绵、刀片，以及篮子/牛油果的身份混淆。C050 再补两个 re-entry seeds，恢复了放下牛油果后篮子与牛油果的独立身份。

仍有局部限制：腕部运动模糊、出画后的细小残留、杯架细枝、置物架细结构。不能确认的区间标为 unknown；杯架/置物架不完整时不把 ROI 外当成已知背景。C038 的原 anchor 因模糊被忽略，初始物体清单因此为空；release 显式补回 task18 的 cup / basket 清单，保留原 manifest 和审核来源。

机器人 arm / EE 层独立保留。只有本次检查认为可靠的 EE 进入 task ROI；机械臂不进入。C015、C026 的 EE 有背景/物体泄漏，已排除其监督资格。未从 object mask 中减去 robot mask。

已有 accepted mask 不被覆盖。有少量 accepted negative 与可见物体冲突时，原始标签保持原样，仅在派生 release 将该条标记为 unknown，并记录 `accepted_negative_discrepancy_unknown`。所有修改可回溯到 seed、图像 hash、原始 draft 和 review。

这些是**经助手抽样复核的 pseudo-labels**；除原有 accepted seed 外，不宣称逐像素人工 GT，也没有本批全帧人工 IoU / Dice。SAM presence confidence 不是像素准确率，不能把接近 1 的值解释为 100% 正确。

## 数据接口

服务器数据根目录：`annotations/action_metric_calibration64_20260919/`。

| 文件 | 用途 |
|---|---|
| `dataset.json` / `clips.csv` | release 有效物体清单、覆盖、输出 hash、每 clip 路径 |
| `manifest.json` / `splits.json` | 不变的选择、源路径/hash、时间、C_fit/C_diag/H、随机种子 |
| `review.json` | 可靠身份、排除区间、缺失边界、修复版本及审核证据 |
| `clips/<id>/rgb/*.png` | 解码后的原分辨率无损 RGB；每个 clip 是指定的一个相机，不是三相机同步三份 clip |
| `clips/<id>/state_action.npz` | 原始 20D absolute action、25D state、有效动作维度和原始时间 |
| `clips/<id>/state_action.parquet` | 原数据对应连续行；时间列为数据集内相对时间 |
| `clips/<id>/accepted_seeds.json` | 不变的已确认来源标签 |
| `clips/<id>/release/objects.jsonl.gz` | object COCO RLE、bbox、visibility、confidence、provenance、资格和异常标记 |
| `clips/<id>/release/robot_parts.jsonl.gz` | 独立的 arm/EE mask，身份沿用已有规则 |
| `clips/<id>/release/task_roi.npz` | bool `roi` / `known`，均为 N×H×W，另有 frame_idx / timestamp |
| `clips/<id>/release/frames.json` | 每帧 ROI 像素数、未知身份和边距 |
| `clips/<id>/release/qa.mp4` | RGB / object / robot / reliable ROI 四面板 |
| `independent_validation.json` | 独立工程检查结果；不等同语义准确率 |

有效动作顺序为 torso 4、left arm 7、left gripper 1、right arm 7、right gripper 1。未改成 delta，未归一化，未加 32D padding。25D raw state 保留 chassis、torso、双臂/夹爪、head；模型如何选择 state 由后续模型 adapter 明确。

`state_action.npz` 的 `timestamp.source/state/action/images.head/images.left/images.right` 保留相应原始时间；相对 parquet 时间加 `original_timestamp_start` 可核验。所有 bbox 为 `[x0,y0,x1,y1]`，右下界不包含。`mask=null, visible=null` 是 unknown，不是 absent。`training_eligible` 控制是否进入可靠 ROI；它是数据质量资格，当前没有启动训练。

ROI 为可靠 object 与可靠 EE 的并集，加固定局部边距：640 宽腕部 4 px、1280 宽头部 8 px。未知前景不贡献 ROI；若某身份不可靠/边界不完整，仅可靠 ROI 内标为 known，其余 unknown；完整清单得到确认的帧允许已知背景。不要忽略 known 把 unknown 当作负标签。

最小读取方式（NumPy；无 GPU）：

```python
from pathlib import Path
import numpy as np

clip = Path("clips/<clip_id>")
data = np.load(clip / "state_action.npz", allow_pickle=False)
masks = np.load(clip / "release/task_roi.npz", allow_pickle=False)
action = data["action_absolute_20d"]  # [N,20]
roi, known = masks["roi"], masks["known"]
```

`calibration64_mask_adapter.py` 只提供显式时间支持上的面积池化：先使用与 RGB 完全相同的空间变换，再传入已核验的 VAE 源帧组。未知 m=0，不按 known 像素重新归一化。**没有猜测 native action chunk、condition latent、causal token、VAE 时间组或 action τ**；这些字段仍为空。数据准备完成不等于模型绑定或 VJP 校准通过，接入实际 checkpoint 时必须再核对。

## 验收与交付

64 个 release 已生成，共 22,314 条 object/robot 记录，其中 11,507 条 mask 可进入 ROI。3,452 / 3,456 帧有可靠 ROI，4 帧零 ROI 保留，不按响应或是否有前景删除样本。777 帧存在至少一个未解决或边界不完整的 object；这些帧继续保留可靠前景，但不能作为完整实例分割 GT。

独立验收 **PASS**：全部 3,456 帧和 22,314 条记录通过源数据对应、原始 seed/hash 不变、20D action / 25D state 数值与原 parquet 相等、时间对应、RLE 解码、bbox/面积、ROI 精确重算、known 规则、划分隔离及 64 段视频解码检查。记录中有 13,720 条非空 mask、7,680 条 unknown；非空 mask 包含仅供 QA 的机器人/被排除几何，不全都可监督。PASS 指工程和数据契约检查，不代表未测的语义 IoU 达标。

最终包为 `deliverables/astribot_action_metric_calibration64_20260919.zip`。包含 RGB、数值数据、mask、64 个 QA 视频、离线 `index.html`、代码快照与 SHA256SUMS。下载并解压后打开 `index.html`，无需公网服务。代码快照用于追溯；完整再生仍需服务器原始数据和已安装的 SAM2 环境，ZIP 不包含模型权重。
