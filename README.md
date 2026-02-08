# DiT-Mindat

## 可用模型

| 模型名称 | 权重文件网址 |
| :--- | :--- |
| DiT-B | https://huggingface.co/Quanli1/DiT |
| DiT-B+RoPE | https://huggingface.co/Quanli1/DiT-B_RoPE |
| DiT-B+RoPE+Offset Noise | https://huggingface.co/Quanli1/DiT-B_RoPE_Offset |
| DiT-B+RoPE+SwiGLU | https://huggingface.co/Quanli1/DiT-B_RoPE_SwiGLU |
| DiT-B+RoPE+Offset Noise+SwiGLU | https://huggingface.co/Quanli1/DiT-B_RoPE_Offset_SwiGLU |

## 文件说明
以DiT-B为例
* **`models.py`**: 定义 DiT 的核心网络架构（Transformer Block 等）。
* **`train.py`**: 训练主脚本，支持多 GPU (DDP) 并行训练。
* **`sample.py`**: 单机推理脚本，用于快速生成少量演示图像。
* **`sample_ddp.py`**: 多卡并行采样脚本，用于高效生成大量图像以计算 FID、KID、IS。
* **`download.py`**: 辅助工具，用于加载本地权重或下载官方预训练模型。
  
### 训练DiT
```
torchrun --nnodes=1 --nproc_per_node=2 train.py --model DiT-B/2 --data-path /vdc1/ddpm_label/Minerals_type_images_6114_label
```
### 采样
```
python sample.py --model DiT-B/2 --image-size 256 --ckpt /vdc1/DiT/results/000-DiT-B-2/checkpoints/0066000.pt
```







