# 环境、依赖和低负载接手

先运行标准库 `verify_bundle.py`。CPU 示例只依赖 `requirements-cpu.txt`；不会 import Torch 或占用 GPU。完整历史标注环境与下面的便携 smoke 分开。

## 已在原服务器使用的配置

Python 3.10；Torch 2.5.1+cu124、torchvision 0.20.1、NumPy 1.26.4、OpenCV 4.10.0.84、Pillow 10.4.0、pyarrow 17.0.0、pycocotools 2.0.8、scipy 1.15.3、Hydra 1.3.2、iopath 0.1.10。A100，CUDA toolkit 12.4；本次不修改任何已有训练环境。

```bash
conda create -n robot-mask python=3.10 pip
conda activate robot-mask
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-gpu.txt
mkdir -p third_party models/sam2
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
# 接收机 CUDA toolkit 路径和 GPU 架构不同则相应修改；下面是原 A100 配置。
CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 \
  python -m pip install --no-build-isolation --no-deps -e third_party/sam2
curl -fL --retry 3 -o models/sam2/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
python -m pip check
```

SAM2.1 base+ 权重 SHA256：`a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`。`dependencies.lock.json` 保存对应源 commit、URL 与哈希。权重只从官方来源下载，不包含在 Git 仓库里。

GroundingDINO 仅在需要自动候选时安装，按 `tools/object_annotation/README.md` 的官方源码/本地 BERT 安装步骤；其中 `<SOURCE_WORKSPACE>` 用本仓库 root 替代。已有 reviewed seed 的便携 SAM2 smoke 不需要 DINO 或 BERT。

DINOv2 仅在需要 bank 检索时安装官方 `facebookresearch/dinov2`，commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`，权重 `dinov2_vits14_pretrain.pth`。bank 不是首个 clip 的前置依赖。完整 historical unit suite 的 imports 可能需要 Torch、SAM2 或 scipy，但不应启动 GPU 推理。

## 一个便携短片段

```bash
python tools/handoff/propagate_clip.py \
  --clip examples/kettle_wrist/clip.json \
  --seeds examples/kettle_wrist/seeds.json \
  --output outputs/kettle_draft --dry-run
```

确认接收机 GPU 归属后，把 `DEVICE_ID` 换成可用的编号：

```bash
CUDA_VISIBLE_DEVICES=DEVICE_ID OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python tools/handoff/propagate_clip.py \
  --clip examples/kettle_wrist/clip.json \
  --seeds examples/kettle_wrist/seeds.json \
  --checkpoint models/sam2/sam2.1_hiera_base_plus.pt \
  --output outputs/kettle_draft
```

默认限制 128 帧，显式一个可见 GPU，参数冻结。新输出 root，PNG 只读，派生 JPEG cache 与候选 JSONL 在输出下。未来 seed 可用于离线标注；邻帧始终保持未人工确认，输出不自动变成 training_eligible。首次运行检查 GPU 日志和原图/叠加；此新 wrapper 本次仅做 CPU/契约验证，没有在满载源机额外启动 SAM2。

长期运行用 tmux，并把日志写入新服务器数据盘。先少量 worker，单 worker `OMP/MKL=2`、`OPENBLAS=1`；不要从历史 10 卡队列配置推断新机能承受相同并行度。

## 模型与第三方归属

源仓库链接与固定 commit 见 lock；各上游代码/模型按原许可使用。本交接没有 vendor 第三方源码，也没有为用户原始研究代码、真实图像或标注添加开源授权。可安装官方依赖后在私有协作仓库复现。历史 Mac smoke 用 SAM2.1 Small 和 MPS 兼容处理；当前服务器 base+ 的方法与非重叠策略不同，详见 legacy README，不混称相同配置。
