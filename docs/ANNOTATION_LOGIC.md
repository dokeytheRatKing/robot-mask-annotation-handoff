# 标注逻辑与 GPT-in-the-loop

## 标什么

标注任务中需要被正确建模的对象可见表面。task-object 清单限定语义范围，不在每帧对所有类别无差别检测。物体 `object_id` 表达身份/类型，多个同类实例另外保留 instance ID；不能让 class 编号兼任跨时间稳定实例。

object、robot parts 和 task ROI 分层保存。Astribot 的臂为 1101/1102，末端为 1103/1104，1100 表达 robot_unknown；1000 只是历史 union。已确认的 union 不改写。法兰属于 arm，工具侧外壳与手指属于 EE；这是 Astribot 的约定，新机器人须重新绑定 link 语义。跨手臂或视角变化不能按图像左右重命名。

**绝不从 object mask 中减去 robot mask。** 两层可能都有误差，直接相减会把接触物体的真实可见像素删掉。重叠是复核线索；只在 robot 自身冲突处理里保留 uncertain 区域。

## 三种判断分开保存

| 判断 | 示例 | 保存方式 |
| --- | --- | --- |
| 语义与可见性 | 香蕉在画面底部还有一条可见边缘 | visible/partial_occlusion；mask 可有多个不连通区域 |
| 无可见表面且有上下文证据 | 完全遮挡 / 确实出画 | 对应 visibility，空 mask，记录依据和审核来源 |
| 不知道或模型未检出 | 只看到夹爪，目标可能被遮挡；SAM 输出空 | unknown；不可冒充背景、absent 或零监督误差 |

`predicted_visible` 只表示模型有正 mask。SAM presence 接近 1 仍可能追到杯架。面积小、空 mask、IoU 连续、相似度高都不能独立决定身份或确认状态。

## 每个小批的实际循环

1. **绑定数据**：固定 task/episode/camera、源 RGB/hash、帧顺序、真实时间戳与显示尺寸。区分 UI 序号、源 `frame_idx` 和 SAM 局部索引；没有时间戳时留 null。不同相机不拼成一个传播序列。
2. **看上下文**：先看无 overlay 的原图与接触图，再看跨时间 contact sheet。找目标出现/消失、遮挡、近景、快速移动和身份歧义；精细边界回到原分辨率或 crop。
3. **给 seed**：优先复用同 episode 的已确认 mask；否则由 Codex 看图给原图坐标的 box、内部正点和夹爪/相似物上的负点。记录坐标系和尺寸。种子与图片 hash 绑定，不能仅凭近似文件名挪用。
4. **验证 seed mask**：SAM image predictor 的输出也必须复看叠加。混入夹爪时收紧框、添加负点或局部 crop；不要盲目保留最高 predicted IoU。没有实际看图就不写 approved/visual_review。
5. **传播**：SAM2.1 独立按 camera/episode 推进。离线标注允许使用未来可见帧作为 seed、双向传播和相邻多关键帧片段，但未来 RGB/GT 不因此进入策略的当前模型输入。
6. **检查非 seed**：检查两端、中间、离 seed 较远处、接触边界和重新出现。初次入画、跨相机切换、遮挡后露出与真实 out-of-view reentry 分开描述。
7. **局部修复**：只替换确认有问题的 object×时间窗口；其他对象和 robot 记录原样透传。旧漂移状态要清空，重新可见处要重播种。无法判断的区间变 UNKNOWN，不伪造整段 negative。复查窗口前后拼接，防止下一帧接回旧残影。
8. **升级/发布**：assistant 复核可支持明确范围的 pseudo-label 使用；需要 human-confirmed 时等待真实人工确认。保存版本、审核范围、原话/确认 sidecar、hash 和保留失败列表。标签 seed 的人审状态不传播给邻帧。

## GPT 的提示模板

`templates/visual_review_request.md` 提供发给能看图的 Codex 的上下文；`templates/review_decision.example.json` 提供显式 decision 结构。模板不是模型调用，也不是默认同意。实际 Codex 可以用本地图像工具打开原图/叠加、写提示 JSON，执行 SAM 后继续查看；如果另一台 Codex 不能看图，应保留视觉复核待办并交给能看图的会话。

优先处理去重后的具体异常：某个实例从哪个 source frame 开始串轨、需要看哪些邻帧、拟用什么提示。只有在看过上下文仍有歧义时，把少数原图/overlay 与问题交给人工。手工预算以独立图像、object-frame 标签、浏览审核和像素编辑分别统计；不承诺靠减少标注状态来达成“零人工”。

## ROI 与 known

64-clip release 的具体实现见 `calibration64_release.py`。可靠 task objects 与经审核可靠 EE 的 union 加固定边缘余量；臂不进入 ROI。部分对象未决时只把可靠 ROI 当 known，其余背景保持未知；完整对象清单通过本批复核时才可以标记已知背景。该判断是审核假设，必须记录缺失/不完整对象，不能由模型的正负输出自动推断。

下游同时保存 `roi: bool[N,H,W]` 与 `known: bool[N,H,W]`。latent 映射必须复用实际 RGB crop/resize/layout 和 VAE 原生时间支持，unknown 对应的权重为零。不要复制 DiT4DiT 16-channel 或 LingBot 48-channel 的假定进 RoboTwin 标注器；标注交付留在 source pixel/time 坐标。

## 已踩过的坑

- 正确 seed 后可以长期稳定，也会在遮挡后串号；单 seed 一张好图不证明整段可用。
- 空 seed 会抑制稍后新出现的物体；不能把空提示当全窗口 negative。
- bank margin gate 能拒绝杯架假阳性，也拒绝所有受检正确腕部香蕉；不以一个全局阈值代替上下文复核。
- 改完窗口要查拼接：原修复在 790 停止时，791 又接回旧错误 track。
- 小 mask 可能是真实可见薄边，统一面积阈值会删掉有效目标。
- 已人工确认标签与后来看图矛盾时，保留原文件；新 release 可以标记 discrepancy/unknown，不能静默改参考答案。
- 同一小批反复看图、改提示后测到的好结果属于开发证据。seed 帧排除、按 episode 留出和失败计数是独立质量评估的前提。
