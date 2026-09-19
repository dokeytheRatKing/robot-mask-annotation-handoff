# Astribot 本地人工 mask 标注包

这个包可以在你的 Mac、Windows 或 Linux 电脑上运行。只需 **Python 3.9+ 和浏览器**，无须 pip install、Conda 新环境、GPU、CUDA、PyTorch 或模型权重。下载完成后可以断网标注。

## 1. 下载和解压

在**你自己的电脑终端**执行，不能在 SSH 登录后的服务器终端执行：

```bash
scp yhwang@147.8.117.242:<SOURCE_WORKSPACE>/deliverables/astribot_mask_audit_420_20260916.zip .
unzip astribot_mask_audit_420_20260916.zip
cd astribot_mask_audit_420_20260916
```

如果你平时用 SSH 别名（例如 `<SOURCE_SSH_ALIAS>`）连接服务器，也可以将上述 `yhwang@147.8.117.242` 换成那个别名。使用你当前已确认可用的 SSH 路径；直连服务器需要的 VPN/网络条件与平时 SSH 相同。

Windows 可以在资源管理器中解压，然后在该文件夹打开终端。

## 2. 启动本地标注网页

Mac / Linux：

```bash
python3 start.py
```

Windows：

```powershell
py -3 start.py
```

新版程序默认打开本机浏览器 <http://127.0.0.1:8766>。若没有自动打开，使用终端中打印的网址；旧包可能使用 8765。保持这个终端运行；这里的 `127.0.0.1` 是你的电脑，而非 HKU 服务器。不要直接双击编辑器 HTML 文件。

若端口已经被占用，可让系统选择空闲端口：`python3 start.py --port 0`。若 Python 命令不存在，先安装 Python 3，或使用电脑已有 Anaconda 的 Python。包内程序只依赖 Python 标准库。

## 3. 如何标注

包内有 420 张图片（三路相机各 140 张），共 1,860 个 object/frame 标签。每帧的对象下拉框只列出本 task 涉及的物体。不是每帧都需要画满所有物体；实际看不到的物体应标明不可见状态。

1. 填写标注者名称，从下拉框选择一个物体。核对实例 ID；同一物体跨帧保持相同 ID，不要随模型预测改名字。
2. 若物体可见，选择“可见”或“部分遮挡”。沿真实可见轮廓点击多边形顶点，再点击“闭合多边形”；也可以用画笔添加、橡皮擦修正，支持多个不连通区域。只画物体可见像素，不补全遮挡部分，不画夹爪。
3. 若完全出画或完全被遮挡，选择对应状态，mask 必须为空。无法判断时选“无法确定”，也保持空 mask。
4. 勾选实际适用的场景标签（下面有英文对照），必要时填写备注。勾选人工确认，点击“保存当前物体”。保存后再切换物体或下一帧。
5. 可以随时退出。按 `Ctrl+C` 停止本地服务，已保存进度保留在 `audit/human_labels/`，每次修改记录保留在 `audit/human_label_revisions.jsonl`。重新运行启动命令会读回已有标签。

场景标签：`normal_unoccluded` 正常无遮挡；`grasp_contact` 抓取接触；`partial_occlusion` 部分遮挡；`out_of_view` 出画；`reentry` 重新入画；`similar_objects_near` 相似物体靠近；`wrist_closeup` 腕部近景。只勾选真实适用项。

新版页面默认显示全部物体的不同颜色 mask。若包中含辅助草稿，会明确显示待复核；已确认标签优先，保存时保留辅助来源。模型预测不会因为预填就自动变成人工确认。当前 pilot 每种类只有一个待标注实例；发现同类多个实例时请备注，不要把两个实例合并成一个 mask。

## 4. 导出并回传

可以先完成少量标签，导出一批验证流程；不必等 420 帧全部完成。

在本地另一个终端进入此文件夹，然后运行：

```bash
python3 export_labels.py
```

Windows 使用 `py -3 export_labels.py`。脚本会产生一个带时间戳的 `mask_audit_return_*.zip` 及其 SHA256 文件，显示已保存标签/完整帧数量。导出的包只包含人工标签、修订记录和校验信息，不含 420 张图片。导出不会清空本地进度。

回传命令（在你自己的电脑执行）：

```bash
scp mask_audit_return_*.zip yhwang@147.8.117.242:<SOURCE_WORKSPACE>/incoming/mask_audit_returns/
```

如果你使用别名，将服务器地址替换成同一个别名即可。上传后告知服务器端 Codex 压缩包文件名，由它校验、导入人工标签并计算真实 IoU/Dice、visibility、ID switch 和 re-entry 指标。部分标注也可评估，但会明确显示完成比例。

## 数据与可恢复性

这个包只包含独立派生 RGB 审核图片、manifest 和工具，不含模型、训练集全量数据或凭据。不要重命名 `audit/images/` 中的图片或修改 `audit/manifest.json`；标签回传通过图像/manifest 哈希匹配。保存前的未提交修改不会自动保存，离开页面前请点击保存。

可选完整性检查：Mac 执行 `shasum -a 256 -c SHA256SUMS`；Linux 执行 `sha256sum -c SHA256SUMS`。该清单覆盖打包时的文件，之后新增的人工标签不在清单内，导出时会另行校验。
