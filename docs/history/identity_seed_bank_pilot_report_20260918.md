# Identity-aware Seed Bank pilot

日期：2026-09-18。结论：**选种子辅助验证通过，带明确限制；不批准全量自动验收。**

## 已完成

- 从已确认的420帧和87帧中复用 mask；没有新增人工精确标注任务。
- 初选265个 exemplar，逐页检查全部265个 RGB/masked crop；24个严重模糊、碎片过小或混入其他物体的 exemplar 退出默认检索，241个 active。原始确认标签不变。
- 覆盖73个直接 identity/camera 组合；计入逐样本明确的肉块别名后，87个目标组合中46个达到至少3个 exemplar、30个仅1–2个、11个缺失。不复制相邻帧凑数。
- 支持24个物体ID及 left_arm/right_arm/left_gripper/right_gripper/robot_unknown。四个已命名机器人部件有40个 active exemplar；unknown 没有可靠确认样本，留空，不将总 robot union 冒充 unknown。
- 保存原始分辨率二值 PNG、COCO RLE、RGB crop、masked crop、bbox、时间戳、相机、episode/frame、来源/hash、状态与可选视角标签。
- frozen DINOv2 ViT-S/14，384维L2归一化。RGB、灰背景 masked crop、两种相似度均值三种消融；没有训练模型，也没有下载第二个 encoder。三种输入不是三种 encoder。
- 新候选由官方 GroundingDINO Swin-T + SAM2.1 base+ 产生；记录 detector score、bank cosine、竞争身份 margin、nearest exemplar IDs、最终排序。缺少参考会明确回退排序并拒绝自动接纳。

## 比较方法

固定99帧：87个已确认难例，加对称 task24 episode_002577 的12帧；三路相机各33帧。共625个身份查询、2,718个 SAM 候选。

查询时排除**整个查询 episode**的所有参考，包括其他相机；优先同相机参考，没有则允许显式 cross-camera fallback。候选生成看不到参考 mask；各排序方法使用完全相同的候选池。比较指标为可见目标的 Top-1 mask IoU >=0.5，这是“可用 seed”代理指标，同时受身份和分割质量影响，不是纯身份分类准确率。

这是 **episode-excluded development pilot**，不是独立无偏测试集：难例被刻意选择，bank 的样本选择/视觉筛选使用了整个已确认池。因此不能据此宣称全库准确率，也未验证长时传播稳定性。

固定排序分数：0.25×detector + 0.75×cosine + 0.25×(正类cosine−竞争类cosine)。系数未在这99帧上拟合。detector候选floor=0.10、每个目标最多5个、NMS=0.5。简单接纳门槛为detector>=0.30；bank方法另需cosine>=0.50、margin>=0.05。这些门槛未校准为概率。

## 物体结果

182个可见查询，95个不可见查询；其中130个可见查询的候选池内存在IoU>=0.5的候选。另52个无论怎样重排都无法成功，需要改善候选生成/补种子。

| 方法 | Top-1可用 /182 | 比例 | 可见目标平均IoU |
|---|---:|---:|---:|
| Detector only |85|46.7%|0.445|
| Bank RGB |108|59.3%|0.553|
| Bank masked |119|65.4%|0.603|
| Bank hybrid |118|64.8%|0.597|

masked相对detector救回36个、损失2个；hybrid救回34个、损失1个。有独立参考且候选池包含可用mask的128个查询中，masked选对117个(91.4%)、detector83个(64.8%)。不能只报告这个较容易的子集而省略总体65.4%。

| Camera | 可见数 | Detector Top-1 | Masked Top-1 | 平均IoU detector → masked |
|---|---:|---:|---:|---:|
| head |91|51|74|0.531 → 0.762|
| left_wrist |39|16|20|0.391 → 0.468|
| right_wrist |52|18|25|0.337 → 0.426|

重点身份 detector → masked 的Top-1计数：kettle 9→10/16；basket 12→13/16；peach 3→4/5；watermelon 6→8/8；avocado 2→4/6。小样本，不作显著性声明。head与完整果实外观受益最大；wrist部分把手、边缘碎片、严重模糊仍困难。

## 接纳与拒绝的代价

| 方法 | 接纳查询 | 接纳且IoU>=0.5 | 可见但mask不合格 | 不可见误接纳 |
|---|---:|---:|---:|---:|
| Detector |245|79|85|81|
| Masked |54|52|1|1|
| Hybrid |47|46|0|1|

误接纳降低伴随**大量拒绝**，不是免费提升召回。Hybrid接纳中46/47符合该IoU门槛，但仅接纳了182个可见目标中的46个正确seed(25.3%)；不能说“97.9%全自动准确”。Q033_10仍把用户已确认不是peach的米色圆形物体当作peach，margin也未能阻止。

## Robot parts

165个可见部件查询：detector35/165(21.2%)，masked81/165(49.1%)，hybrid83/165(50.3%)；候选池可用上限91/165。左右同类部件使用相同通用检测提示，appearance排序只是诊断，不证明真实左右身份或法兰盘边界。所有机器人部件自动接纳均关闭，不能用“零误接纳”伪装高质量。unknown保留为合法未决身份，但没有编造 exemplar；accepted87 的四部件union未变。

## 直接视觉检查

已检查10张原图/GT/detector/hybrid对照图，记录在pilot目录的assistant_visual_review.json。

- Q038_10：peach由错误的手持西瓜片纠正到篮中桃子。
- Q033_21：avocado由右侧米色干扰物纠正到机械臂旁的牛油果。
- Q006_1：kettle从连带gripper的大mask改成水壶本身。
- Q032_4：left-wrist近景basket原检测正确，hybrid反而选中底边碎片；margin门槛拒绝了它，但排序依然是失败。
- Q019_1：只露出把手的kettle仍与机器人外壳混淆；门槛拒绝。
- Q034_4：模糊/出画边缘basket，两种方法均失败。
- Q044_4：basket有所恢复但仍混入手指，不能作为精确GT。
- Q033_10：上述peach干扰物误接纳仍存在，必须保留为失败案例。

不将这些问题重新交给用户标注：已有确认参考足以判定。下一步优先增加助手确认的干扰物/局部视角参考，以及跨视角/解剖约束；与本次冻结实验分版本比较。

## 性能与验证

单张物理GPU2运行，未使用GPU1。99帧全部候选生成+embedding+排序+QA耗时345.8秒，约3.49秒/帧；不含模型冷启动。Torch峰值allocated显存1.695GiB，不代表进程全部reserved/驱动显存。初建bank58.2秒，筛选后同候选复算147.1秒。不能用这个稀疏难例pilot直接估算全库SAM传播耗时。

49个annotation单元测试通过。结构校验通过：265个资产/RLE/bbox/来源检查、384维unit embeddings、2,718个候选、一致候选池、30,048次nearest/competitor跨episode排除检查。结构PASS不是像素准确率PASS。

## 路径与下一步

- 库：annotations/identity_seed_bank_20260918_curated
- 对比：annotations/identity_seed_bank_pilot_20260918_curated；metrics.json、comparison_tables.csv、queries.jsonl、candidates.jsonl、qa/。
- 离线审核：库中的review.html；筛选身份/相机/状态，批量确认当前筛选中的未审核项，导出seed_bank_user_review.json。只影响审核文件，不直接改变源标签或运行bank。
- 使用说明：tools/object_annotation/SEED_BANK.md。

建议将masked crop作为下一版物体选种子的优先消融候选，hybrid作为保守交叉检查，而不是在本轮结果上继续调阈值并宣称独立验证通过。先审核bank、增加少量自动/助手确认的困难视角、做新episode的multi-keyframe recovery canary。此次没有全量重跑、传播、重建、训练或temporal grounding。
