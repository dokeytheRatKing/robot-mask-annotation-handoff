# RoboTwin 接入：先核验实际数据

用户所说的 rebotwin 在本交接中按 **RoboTwin** 理解。没有收到目标服务器的数据样本，以下是已核验的官方代码与实际接入步骤，不是对某份下载数据内容的保证。

## 官方代码核验

2026-09-19 直接读取官方仓库 `RoboTwin-Platform/RoboTwin` 的 main，当时 commit 为 `6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755`。

- [camera.py](https://github.com/RoboTwin-Platform/RoboTwin/blob/6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755/envs/camera/camera.py#L376) 有 mesh/actor segmentation 接口，但该 helper 会把 raw renderer channel 转为 uint8 后映射颜色。**它返回的彩色图不能自动当作完整、无损、已语义绑定的 actor ID 图。** 应优先在渲染端保留原始整数 IDs 及 episode 对应的 actor/link/name 映射；重新运行仿真还要验证与既有 RGB 的场景、时间、相机完全匹配。
- [_base_task.py](https://github.com/RoboTwin-Platform/RoboTwin/blob/6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755/envs/_base_task.py#L438) 按 `data_type` 配置选择收集 RGB、mesh/actor segmentation 等，存在接口不等于某份已导出的数据启用了该选项。
- [pkl2hdf5.py](https://github.com/RoboTwin-Platform/RoboTwin/blob/6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755/envs/utils/pkl2hdf5.py#L79) 的这版 exporter 写 XPolicyLab 风格 `vision/.../colors`、`state`、`action`，state 使用序列前 N−1 项、action 使用后 N−1 项；这里检查到的 exporter 未写 mesh/actor segmentation。此前版本/其他 policy 转换数据可能仍用 `observation/.../rgb` 等结构。

因此：**先检查目标文件，已有可信仿真 mask 时优先利用；只有 RGB 或无法匹配 ID/帧时，再运行 GPT-assisted SAM2。** 本次不为拿 masks 强制重采整个 RoboTwin 数据集，也不假设当前 main 与用户数据版本相同。

## 只读 inventory

```bash
python tools/handoff/inspect_robotwin.py \
  --input /DATA/robotwin --max-files 3 \
  --output outputs/robotwin_inventory.json
```

工具列出 HDF5 key、shape、dtype、候选 RGB/segmentation/time 字段；不会执行 pickle、不下载整库、不改原数据。文件名和 schema hints 不是最终语义映射，须结合对应数据文档检查。默认只取遍历时首先发现的少量文件并排序，不把它当作随机统计样本；图像解码在下一步显式选择 RGB key 后检查。

## 提取一个显式选择的 RGB clip

先从 inventory 选择真实的 key。下列 key 只是示意，请用本机实际值：

```bash
python tools/handoff/extract_hdf5_clip.py \
  --input /DATA/robotwin/task/data/episode.hdf5 \
  --rgb-key observation/head_camera/rgb \
  --channel-order RGB --task-id TASK_NAME --episode-id EPISODE_NAME \
  --camera head_camera --start 0 --count 32 --stride 1 \
  --output outputs/robotwin_clip_001
```

支持 uint8 `[N,H,W,3]` 与逐帧 JPEG/PNG 字节。raw ndarray 通道次序由 `--channel-order` 明确给出；压缩图按 OpenCV 解码再转 RGB。若存在真实一维时间戳字段，用 `--timestamp-key` 显式指定；没有就保存 null，不用 FPS 凭空制造采集时间。只有真实 schema 明确表示 float RGB 范围时才另行实现转换，本工具不会自动乘 255 猜测。

提取保留 source row，不变更 action/state，不猜动作与图像 shift。不要直接照搬 Astribot 的 20D absolute action、25D state 或相机 source-time 规则。

## Task 与身份字典

按对应 RoboTwin task 的操作目标建立版本化 registry：

- semantic object / instance ID、角色（被操作对象、目标容器、约束物等）、允许的同类多实例；
- 对应 actor/link 名字与 IDs、episode provenance、是否可复用跨 episode；
- robot base/arm/EE 的物理 link 分组，左右不按画面 x 排序；
- distractor 如何处理、可见像素还是 amodal、已知/未知的定义。

`templates/robotwin_task_registry.example.json` 是空结构，不包含伪造的 task 列表。最终 object scope 由任务定义和用户要求共同确定，不能直接导入 `tools/object_annotation/config/task_objects.json`。

## 与训练端的交接

输出 source resolution 的 mask + known、camera/source frame/time、task/instance ID、seed/review/version/hash。训练端自己复用实际 RGB 几何变换与 VAE 时间支持。3 路相机中只标了一路就显式记录另外两路未知，不能复制 mask。不要让 mask、seed 或上下文中的真实 future 成为原动作时刻的额外模型输入。

任何模拟器原始 ID 导出路径都要单独做：RGB/ID 同帧、分辨率、遮挡、actor-ID aliasing、机器人 link 身份、丢失 ID/背景、source revision 的验证。满足这些条件时它才是可靠的 simulator-derived mask；无需为了复现 Astribot 工作流再人为加一层 SAM。
