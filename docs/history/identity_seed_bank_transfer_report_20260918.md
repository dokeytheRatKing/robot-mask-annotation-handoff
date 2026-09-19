# Seed Bank 实际传播验证

日期：2026-09-18。结论：**有局部身份排错价值，但尚未证明整体传播质量提升；不推广全量，不普遍扩充种子库。**

## 审核与实验范围

用户已确认全部种子库审核通过。本次记录独立 acceptance sidecar，保持原 bank.json 不变。
265 个归档 exemplar 中 241 个 active、24 个 retrieval-excluded 的审核决定均保留。
没有新增人工标注、修改原视频、accepted masks 或 frozen production。

冻结模型：GroundingDINO Swin-T、SAM2.1 base+、DINOv2 ViT-S/14 masked crop embeddings。
使用 GPU2–5，避开 GPU1；实验 GPU 工作已结束。

- 新 episode：task1 左侧001643、task9 右侧004607、task24 左侧002503/右侧002511，三路相机，共12段121帧短片。这4条 episode 从未进入 bank 或 accepted420/87。
- 定量评分：accepted87 中 task1/9/24 × 三路相机的9段31帧短片。每段在评分帧之前15帧初始化；评分 mask 不作为 seed。
- 主实验总计21段、1,731个原始 camera-frame，每帧分别生成3种策略输出。新片每40帧独立重新检测/初始化，评分短片仅开始时初始化一次。
- 每次检索排除整个 query episode；每个策略使用完全相同的 DINO+SAM image 候选池。
- 这些是困难开发样本，不是随机总体测试；新 episode 只有图像检查，没有新增像素 GT。

## 定量结果

评分集只有24个 object-frame 标签：16个可见、8个缺席。未初始化/拒绝保持 UNKNOWN，评分时按空预测惩罚可见漏检，不将它当作确认缺席标签。

| 策略 | 可见物体平均 IoU | IoU≥0.5 | 可见但无 mask | 缺席却有 mask |
|---|---:|---:|---:|---:|
| Detector-only | 0.3076 | 5/16 | 2/16 | 5/8 |
| Bank 排序，原筛选顺序 | 0.2947 | 5/16 | 6/16 | 2/8 |
| Bank 排序，先过滤 detector<0.30 | 0.3076 | 5/16 | 2/16 | 5/8 |
| Bank 排序 + 严格身份门槛 | 0.2945 | 5/16 | 11/16 | 0/8 |
| 旧 frozen production，仅供参照 | 0.7182 | 12/16 | 2/16 | 2/8 |

严格门槛为 cosine≥0.50、identity margin≥0.05，未根据本次标签调阈值。
它保留的5个可见预测均 IoU≥0.5，但漏检代价很高，不能只报告这些预测的精度。
先筛除低 detector 分数再排序的补充实验重用原候选和 embedding，只重跑9段传播，未更换模型或增加种子。
补充实验表明原筛选顺序的一部分“降低误检”其实来自额外拒绝，不能误认为纯排序收益。

修正筛选顺序后，各相机可见平均 IoU（detector / bank / gate）：

| Camera | 可见/缺席标签 | Detector | Bank | Gate |
|---|---:|---:|---:|---:|
| head | 7/1 | 0.5575 | 0.5575 | 0.5575 |
| left_wrist | 7/1 | 0.1455 | 0.1455 | 0.1157 |
| right_wrist | 2/6 | 0.0000 | 0.0000 | 0.0000 |

旧 production 包含不同提示词、历史身份筛选及未来关键帧/多条件传播；accepted masks 也由已有输出审核/修正而来，因此不是同条件、独立 GT 的算法对照。其较高数值不证明它全量正确，只提示不能用本次简化的单起点传播替代已有恢复能力。

## 新 Episode 的实际画面

助手实际检查12个新相机片段的21张六宫格截图，以及4张评分窗口截图；记录在主结果目录 `assistant_visual_review.json`。不是逐帧人工验收。

- task1 head：detector 把机械臂/夹爪当水壶，bank 可拒绝，但也漏掉被夹爪挡住的真实水壶；后续清楚可见时可恢复。
- task24 head，002503 frame528：bank 排序将西瓜与圆形桃子区分开；严格 gate 排除 avocado-on-gripper，但在588帧也漏掉真实可见牛油果。
- task24 right_wrist，002511 frame225：gate 保留被抓住的香蕉，并消除多种果类重复覆盖；不过165帧篮子候选已混入夹爪，单靠身份相似度无法修复边界。
- task9 wrist：水壶与夹爪接触后 mask 粘连仍存在；严格 gate 有时干脆丢掉整个真实水壶。
- task24评分窗口：起点视角中目标大部分不在画面内，15帧后篮子/水果重新出现。仅靠起点 seed 排名不能恢复后出现的物体，需要中途重新检测与 reseeding。
- robot layer 完全不变，只用于 QA。机器人左右身份、法兰边界质量没有在本次得到改善或验收。

QA confidence 是 SAM2 object-presence score，不是身份正确概率或 IoU；错误轨迹也可能接近1.00。

## 是否扩充种子库

**暂不做普遍扩容。** 先保留已审核 bank，修复候选与恢复流程，再进行有针对性的增补。

1. 当前测试涉及的水壶、篮子、桃子、西瓜、牛油果，在每个 camera 已有约3–6个 active exemplar；不是简单“没有参考”。增加同样的清晰正面图价值有限。
2. 优先解决新目标入画时没有 seed、坏起点持续传播、mask 粘夹爪、多个身份选择同一区域的问题。加入银行检索不会自动解决这些问题。
3. 后续只补“失败条件”：腕部近距离部分水壶/把手、被抓持的果类、边缘/侧面篮子、强遮挡下可辨认的物体。优先从现有 accepted masks 中重新挑选尚未入库的样本；必要时助手指导 SAM2 生成版本化 assisted exemplar，不自动称为人工 GT。
4. 补足跨 episode 多样性比增加同 episode 邻帧更有用，例如桃子 right_wrist 当前5个参考仅来自1条 episode。
5. 本次只验证了3个 task，不足以决定其余26个 task 的扩容优先级。机器人左右区分需要结构/运动证据，不应单凭视觉 embedding 补几张后自动接受。

下一阶段建议：保留 bank 排序作为身份信号，在独立小范围实现可见性变化/异常触发 reseeding；对照相同关键帧、相同候选、相同传播配置的“有无 bank”。未知继续保持未知，不为了填满标签放松正确性约束。本次没有启动这一阶段或全量重跑。

## 验证与产物

主结果：`annotations/identity_seed_bank_transfer_20260918_r2`。
筛选顺序补充实验：`annotations/identity_seed_bank_transfer_20260918_eligible`。
包含 frozen config、acceptance、模型/代码 hash、候选 mask/检索结果、选择原因、三套逐帧 RLE、评分和验证报告。
主实验每 worker 峰值 Torch allocated 最大2.152 GiB；四卡并行，21段计算耗时相加约25.9分钟（不是墙钟 ETA，包含3套传播与旧预览）。不据此外推全量时长。

单元测试57项通过。两组 dense validator 覆盖 RLE/bbox/area、行唯一性、时间戳、query-episode排除、同候选决策、来源 hash、视频全帧解码。
语义指标如上，结构 PASS 不等于质量 PASS。

预览工程问题已记录：首轮因旧 robot string-ID 绘图兼容失败；r2 的旧 writer 固定分辨率导致旧 `comparison.mp4` 不可用。保留诊断文件；只使用 `qa_verified/` 下重建的原比例、带ID/名称/分数的30个视频，视频清单与 hash 见 `qa_manifest.json`。没有因预览问题重写任何 mask。

下载包：`deliverables/astribot_seed_bank_transfer_qa_20260918.zip`。包含两组已验证视频、截图、报告、CSV/JSON与复现脚本；不需要服务器网页、GPU或新人工标注。
