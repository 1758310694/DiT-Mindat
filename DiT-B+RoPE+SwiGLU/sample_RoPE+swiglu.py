# sample_swiglu.py
"""
Sample new images from an improved DiT (RoPE + SwiGLU version).
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model

# -----------------------------------------------------------------------------
# 关键修改 1：从 models1 导入，以匹配 RoPE + SwiGLU 架构
from models1 import DiT_models
# -----------------------------------------------------------------------------

import argparse
import os

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 关键修改 2：强制要求 checkpoint
    if args.ckpt is None:
        raise ValueError("Improved DiT (RoPE+SwiGLU) requires a custom --ckpt from train1.py.")

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    # 关键修改 3：同步 DDP 脚本中的权重加载逻辑
    ckpt_path = args.ckpt
    print(f"Loading checkpoint from {ckpt_path}...")
    state_dict = find_model(ckpt_path)

    if "ema" in state_dict:
        print("Using EMA weights.")
        state_dict = state_dict["ema"]
    elif "model" in state_dict:
        print("Using standard weights.")
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # 你可以自定义想要生成的类别 ID
    class_labels = [918, 1793, 2240, 4653]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    print(f"Sampling with {args.model} (RoPE + SwiGLU)...")
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_path = "sample_swiglu.png"
    save_image(samples, save_path, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Done. Sample saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema") # 建议默认 ema
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=5218)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to your models1.py checkpoint.")
    args = parser.parse_args()
    main(args)