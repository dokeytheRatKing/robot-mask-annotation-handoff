# 交接验收：2026-09-19

本次只使用 CPU 与文件读取。没有启动新的 GPU 标注/训练，也没有修改原数据、原标注目录或正在运行的策略训练。

| 检查 | 实际结果 |
| --- | --- |
| 已有 `tools/object_annotation` 单元测试 | **74/74 PASS**，在已有 annotation 环境、禁用 CUDA 下运行 |
| 新便携入口测试 | **5/5 PASS**，新建隔离 CPU venv；测试 HDF5 inventory/提取、PNG/raw channel、重复/null 时间戳、source hash 不变、seed 图像绑定和分段覆盖 |
| 隔离 CPU 环境安装 | 按 requirements-cpu.txt 安装，pip check PASS；未修改已有训练环境 |
| 随包真实案例 | 16 个连续 source frames、80 条 released rows、40 条 null/unknown、15 个精确已确认标签检查 PASS |
| ROI/known 重建 | 每帧面积与源 release 的 frames.json 完全一致；object/robot 保持独立 |
| 便携传播 dry-run | 真正读取 kettle 的 8 帧与 local4/source150 seed，hash/尺寸/绑定 PASS，无 Torch/GPU 加载 |
| 源码与案例来源 | SOURCE_SNAPSHOT.json 逐文件源/导出 hash；旧路径脱敏显式记录 |
| 完整性 | SHA256SUMS 与 verify_bundle.py；交付前验证全部清单与 Python AST |
| 图像人工/assistant 检查范围 | 本轮实际查看 fruit315 的 RGB/overlay/ROI/known 以及 watermelon1008 的旧漂移/修复对比。其余图的语义依据为原报告，不声称本轮逐像素新验收 |

测试初次使用系统 Python 时缺少 pycocotools，已改在隔离环境安装完整 CPU requirements 后通过；不是绕过 seed/RLE 检查。source annotation 原环境没有 h5py，因此没有向该环境安装新依赖。CPU venv 为 Python 3.11；原已有 74 项测试在 Python 3.10 annotation 环境完成。

新 `prepare_point_seeds.py` / `propagate_clip.py` 的真实 GPU 推理**未在本轮运行**。它们使用已执行过的官方 SAM2 接口和原 nearest-seed 逻辑，但这是新包装，仍需要接收机一个真实短 clip 的输出/精度/显存验收。完整历史报告对应原 Astribot 运行，不用来替代该检查。

没有实际 RoboTwin 数据，因此 HDF5 测试是明确的合成工程 fixture，不是 RoboTwin mask 质量测量。未测 RoboTwin 语义映射、episode split、吞吐、覆盖、IoU/Dice 或全库质量；这些留给接手任务。

第三方源码/模型未打包，使用固定官方 revisions 与 hashes。根 NOTICE.md 说明原研究代码/图像的授权范围；GitHub 默认私有。
