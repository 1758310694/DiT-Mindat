# sample_rope.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Sample new images from a pre-trained DiT (RoPE version).
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model

# -----------------------------------------------------------------------------
# 关键修改 1：引用 models_rope，以匹配 Exp 2 的架构 (RoPE + MLP)
from models_rope import DiT_models
# -----------------------------------------------------------------------------

import argparse

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 关键修改 2：RoPE 模型通常是自定义训练的，强制要求提供 ckpt路径
    if args.ckpt is None:
        raise ValueError("Must specify --ckpt for custom RoPE models.")

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    # 关键修改 3：使用 sample_ddp_rope.py 中的权重加载逻辑
    # 因为训练脚本保存的 pt 文件通常包含 "ema" 或 "model" 字典，而不是直接的 state_dict
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)

    # 自动处理 EMA 权重或普通权重
    if "ema" in state_dict:
        state_dict = state_dict["ema"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict)
    model.eval()  # important!

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    # 这里保留了你原始代码中的类别 ID
    class_labels = [918, 1793, 2240, 4653]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device) # 使用参数中的 num_classes
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    print(f"Sampling {n} images with {args.model} (RoPE)...")
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_path = "sample_rope.png"
    save_image(samples, save_path, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Saved sample grid to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=5218) # 这里的默认值对应 ImageNet-5K 或自定义数据集
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    # 强制要求输入 checkpoint
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to a custom RoPE DiT checkpoint.")
    args = parser.parse_args()
    main(args)