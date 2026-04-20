from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


SCRIPT_DIR = Path(__file__).resolve().parent
BOOK_ROOT = SCRIPT_DIR.parent
IPP_ROOT = BOOK_ROOT / "IPPy"
if str(IPP_ROOT) not in sys.path:
    sys.path.append(str(IPP_ROOT))

import operators
from nn.diffusion import (
    DiffusionUNet,
    cosine_beta_schedule,
    denormalize_to_01,
    extract,
)


class MayoDataset(Dataset):
    def __init__(self, data_path: str | Path, data_shape: int = 64) -> None:
        super().__init__()
        self.fname_list = sorted(glob.glob(f"{data_path}/*/*.png"))
        self.transform = transforms.Compose(
            [
                transforms.Resize((data_shape, data_shape), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self) -> int:
        return len(self.fname_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = Image.open(self.fname_list[idx]).convert("L")
        return self.transform(x)


def get_device() -> str:
    try:
        if torch.mps.is_available():
            return "mps"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def gaussian_noise(y: torch.Tensor, noise_level: float) -> torch.Tensor:
    e = torch.randn_like(y, device=y.device)
    return e / torch.norm(e) * torch.norm(y) * noise_level


def make_beta_schedule(num_steps: int) -> torch.Tensor:
    return cosine_beta_schedule(num_steps)


def predict_x0_from_eps(
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    t: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    return (
        x_t - extract((1 - alpha_bars).sqrt(), t, x_t.shape) * eps_pred
    ) / extract(alpha_bars.sqrt(), t, x_t.shape)


def deterministic_ddim_update(
    x_t: torch.Tensor,
    x0_hat: torch.Tensor,
    eps_pred: torch.Tensor,
    t_next: int,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    if t_next < 0:
        return x0_hat
    alpha_bar_next = alpha_bars[t_next].to(x_t.device)
    return torch.sqrt(alpha_bar_next) * x0_hat + torch.sqrt(1 - alpha_bar_next) * eps_pred


def forward_noise_from_clean(
    x0: torch.Tensor,
    t_current: int,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    alpha_bar_t = alpha_bars[t_current].to(x0.device)
    noise = torch.randn_like(x0)
    return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise


def tensor_to_display_array(x: torch.Tensor, mode: str = "image") -> np.ndarray:
    x_cpu = x.detach().cpu().squeeze()
    x_min = float(x_cpu.min())
    x_max = float(x_cpu.max())
    if abs(x_max - x_min) < 1e-12:
        return torch.zeros_like(x_cpu).numpy()
    return ((x_cpu - x_min) / (x_max - x_min)).numpy()


def save_tensor_image(x: torch.Tensor, path: Path, mode: str = "image") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor_to_display_array(x, mode=mode)
    plt.imsave(path, array, cmap="gray", vmin=0.0, vmax=1.0)


def save_kernel_image(kernel: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kernel_np = kernel.detach().cpu().squeeze().numpy()
    kernel_np = kernel_np / max(kernel_np.max(), 1e-12)
    plt.imsave(path, kernel_np, cmap="gray", vmin=0.0, vmax=1.0)


def mse(x: torch.Tensor, y: torch.Tensor) -> float:
    x_01 = denormalize_to_01(x.detach())
    y_01 = denormalize_to_01(y.detach())
    return torch.mean((x_01 - y_01) ** 2).item()


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse_val = mse(x, y)
    return float("inf") if mse_val == 0 else -10.0 * math.log10(mse_val)


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
) -> float:
    x_01 = denormalize_to_01(x.detach())
    y_01 = denormalize_to_01(y.detach())
    coords = torch.arange(window_size, device=x_01.device, dtype=x_01.dtype) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    window_2d = torch.outer(gauss, gauss)
    window = window_2d.expand(x_01.shape[1], 1, window_size, window_size).contiguous()
    mu_x = torch.nn.functional.conv2d(x_01, window, padding=window_size // 2, groups=x_01.shape[1])
    mu_y = torch.nn.functional.conv2d(y_01, window, padding=window_size // 2, groups=y_01.shape[1])
    mu_x2 = mu_x**2
    mu_y2 = mu_y**2
    mu_xy = mu_x * mu_y
    sigma_x2 = torch.nn.functional.conv2d(x_01 * x_01, window, padding=window_size // 2, groups=x_01.shape[1]) - mu_x2
    sigma_y2 = torch.nn.functional.conv2d(y_01 * y_01, window, padding=window_size // 2, groups=y_01.shape[1]) - mu_y2
    sigma_xy = torch.nn.functional.conv2d(x_01 * y_01, window, padding=window_size // 2, groups=x_01.shape[1]) - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return ssim_map.mean().item()


def select_snapshot_positions(num_iterations: int, num_snapshots: int) -> list[int]:
    num_snapshots = max(1, min(num_snapshots, num_iterations))
    return sorted(set(np.linspace(0, num_iterations - 1, num_snapshots, dtype=int).tolist()))


def data_consistency_projection(
    x0_hat: torch.Tensor,
    y_delta: torch.Tensor,
    K: operators.Operator,
    scale: float,
) -> torch.Tensor:
    residual = K(x0_hat) - y_delta
    projected = x0_hat - scale * K.T(residual)
    return projected.detach()


def build_model(weights_path: Path, device: str) -> DiffusionUNet:
    model = DiffusionUNet(
        in_ch=1,
        base_ch=64,
        channel_mults=(1, 2, 4),
        time_dim=256,
        dropout=0.05,
        attn_levels=(1, 2),
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model = model.to(device)
    model.eval()
    return model


def trace_dps(
    model: DiffusionUNet,
    y_delta: torch.Tensor,
    K: operators.Operator,
    alpha_bars: torch.Tensor,
    num_diffusion_steps: int,
    sigma_y: float,
    guidance_scale: float,
    projection_scale: float,
    sample_steps: int,
    snapshot_positions: set[int],
    device: str,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    schedule = torch.linspace(num_diffusion_steps - 1, 0, sample_steps, dtype=torch.long, device=device)
    x = torch.randn_like(y_delta)
    snapshots: list[dict[str, Any]] = []

    for i in range(len(schedule) - 1):
        t_current = int(schedule[i].item())
        t_next = int(schedule[i + 1].item())
        t = torch.full((x.shape[0],), t_current, device=device, dtype=torch.long)
        x_t_vis = x.detach().clone()

        x = x.detach().requires_grad_(True)
        eps_pred = model(x, t)
        x0_hat_raw = predict_x0_from_eps(x, eps_pred, t, alpha_bars)
        d_t = data_consistency_projection(x0_hat_raw, y_delta, K, projection_scale)
        x0_hat = x0_hat_raw.clamp(-1.0, 1.0)
        data_loss = torch.mean((K(x0_hat) - y_delta) ** 2) / (2 * sigma_y**2)
        grad = torch.autograd.grad(data_loss, x)[0]

        if i in snapshot_positions:
            snapshots.append(
                {
                    "iteration": i,
                    "timestep": t_current,
                    "x_t": x_t_vis.detach().cpu(),
                    "x0_hat": x0_hat.detach().cpu(),
                    "d_t": d_t.detach().cpu(),
                }
            )

        with torch.no_grad():
            x_next = deterministic_ddim_update(x, x0_hat, eps_pred, t_next, alpha_bars)
            x = (x_next - guidance_scale * grad).clamp(-1.0, 1.0)

    with torch.no_grad():
        t0 = torch.zeros((x.shape[0],), device=device, dtype=torch.long)
        eps_pred = model(x, t0)
        x_final = predict_x0_from_eps(x, eps_pred, t0, alpha_bars).clamp(-1.0, 1.0).detach()

    return x_final, snapshots


def trace_diffpir(
    model: DiffusionUNet,
    y_delta: torch.Tensor,
    K: operators.Operator,
    alpha_bars: torch.Tensor,
    num_diffusion_steps: int,
    tau: float,
    sample_steps: int,
    snapshot_positions: set[int],
    device: str,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    schedule = torch.linspace(num_diffusion_steps - 1, 0, sample_steps, dtype=torch.long, device=device)
    x = y_delta.clone()
    snapshots: list[dict[str, Any]] = []

    for i in range(len(schedule) - 1):
        t_current = int(schedule[i].item())
        t_next = int(schedule[i + 1].item())
        t = torch.full((x.shape[0],), t_current, device=device, dtype=torch.long)

        with torch.no_grad():
            eps_pred = model(x, t)
            x0_hat_raw = predict_x0_from_eps(x, eps_pred, t, alpha_bars)
            d_t = data_consistency_projection(x0_hat_raw, y_delta, K, tau)
            x0_hat = x0_hat_raw.clamp(-1.0, 1.0)

            if i in snapshot_positions:
                x_t_vis = forward_noise_from_clean(x.detach(), t_current, alpha_bars)
                eps_pred_vis = model(x_t_vis, t)
                x0_hat_vis = predict_x0_from_eps(x_t_vis, eps_pred_vis, t, alpha_bars)
                d_t_vis = data_consistency_projection(x0_hat_vis, y_delta, K, tau)
                snapshots.append(
                    {
                        "iteration": i,
                        "timestep": t_current,
                        "x_t": x_t_vis.detach().cpu(),
                        "x0_hat": x0_hat_vis.detach().cpu(),
                        "d_t": d_t_vis.detach().cpu(),
                    }
                )

            x_prior = deterministic_ddim_update(x, x0_hat, eps_pred, t_next, alpha_bars)
            residual = K(x_prior) - y_delta
            x = (x_prior - tau * K.T(residual)).clamp(-1.0, 1.0)

    return x.clamp(-1.0, 1.0).detach(), snapshots


def save_snapshot_bundle(
    method_name: str,
    snapshots: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for snapshot in snapshots:
        prefix = f"iter_{snapshot['iteration']:02d}_t_{snapshot['timestep']:03d}"
        save_tensor_image(snapshot["x_t"], method_dir / f"{prefix}_x_t.png", mode="image")
        save_tensor_image(snapshot["x0_hat"], method_dir / f"{prefix}_x0_hat.png", mode="image")
        save_tensor_image(snapshot["d_t"], method_dir / f"{prefix}_d_t.png", mode="image")
        manifest.append(
            {
                "iteration": int(snapshot["iteration"]),
                "timestep": int(snapshot["timestep"]),
                "files": {
                    "x_t": f"{prefix}_x_t.png",
                    "x0_hat": f"{prefix}_x0_hat.png",
                    "d_t": f"{prefix}_d_t.png",
                },
            }
        )

    torch.save(snapshots, method_dir / "trace.pt")
    with open(method_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def save_method_grid(
    method_label: str,
    snapshots: list[dict[str, Any]],
    output_path: Path,
) -> None:
    cols = len(snapshots)
    fig, axes = plt.subplots(3, cols, figsize=(2.0 * cols, 6.0))
    if cols == 1:
        axes = np.array(axes).reshape(3, 1)

    row_titles = [r"$x_t$", r"$\hat{x}_0$", r"$D_t$"]
    row_modes = ["image", "image", "image"]
    for col, snapshot in enumerate(snapshots):
        images = [snapshot["x_t"], snapshot["x0_hat"], snapshot["d_t"]]
        for row, image in enumerate(images):
            axes[row, col].imshow(
                tensor_to_display_array(image, mode=row_modes[row]),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(row_titles[row], fontsize=12)
        axes[0, col].set_title(f"t = {snapshot['timestep']}", fontsize=11)

    fig.suptitle(method_label, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_combined_grid(
    dps_snapshots: list[dict[str, Any]],
    diffpir_snapshots: list[dict[str, Any]],
    output_path: Path,
) -> None:
    cols = max(len(dps_snapshots), len(diffpir_snapshots))
    fig, axes = plt.subplots(6, cols, figsize=(2.0 * cols, 11.0))
    if cols == 1:
        axes = np.array(axes).reshape(6, 1)

    for ax in axes.ravel():
        ax.axis("off")

    method_blocks = [
        ("DPS-style", dps_snapshots, 0),
        ("DiffPIR-style", diffpir_snapshots, 3),
    ]
    row_titles = [r"$x_t$", r"$\hat{x}_0$", r"$D_t$"]
    row_modes = ["image", "image", "image"]

    for method_label, snapshots, row_offset in method_blocks:
        for col, snapshot in enumerate(snapshots):
            images = [snapshot["x_t"], snapshot["x0_hat"], snapshot["d_t"]]
            for row, image in enumerate(images):
                ax = axes[row_offset + row, col]
                ax.imshow(
                    tensor_to_display_array(image, mode=row_modes[row]),
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(f"{method_label}\n{row_titles[row]}", fontsize=11)
            axes[row_offset, col].set_title(f"t = {snapshot['timestep']}", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate graphical-abstract assets for the diffusion inverse-problems notebook. "
            "The script reuses the course denoiser and records sparse DPS-style and "
            "DiffPIR-style trajectories."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "graphical_abstract_assets",
        help="Folder where the images, grids, and tensor traces will be saved.",
    )
    parser.add_argument("--image-index", type=int, default=0, help="Index of the Mayo test image to use.")
    parser.add_argument("--data-shape", type=int, default=64, help="Spatial resolution used by the course models.")
    parser.add_argument("--sample-steps", type=int, default=40, help="Number of reverse diffusion steps to run.")
    parser.add_argument("--num-snapshots", type=int, default=10, help="Number of representative timesteps to save.")
    parser.add_argument("--num-diffusion-steps", type=int, default=400, help="Total diffusion steps used by the trained model.")
    parser.add_argument("--noise-level", type=float, default=0.01, help="Relative Gaussian noise level for the corrupted datum.")
    parser.add_argument("--sigma-y", type=float, default=0.01, help="Observation noise scale used in the DPS-style update.")
    parser.add_argument("--guidance-scale", type=float, default=0.15, help="Guidance scale used in the DPS-style update.")
    parser.add_argument(
        "--dps-projection-scale",
        type=float,
        default=None,
        help=(
            "Image-space projection scale used only for visualizing D_t in the DPS row. "
            "If omitted, guidance_scale is reused."
        ),
    )
    parser.add_argument("--tau", type=float, default=0.6, help="Step size used in the DiffPIR-style data-consistency step.")
    parser.add_argument("--kernel-size", type=int, default=9, help="Motion-blur kernel size.")
    parser.add_argument("--motion-angle", type=float, default=20.0, help="Motion-blur angle in degrees.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = BOOK_ROOT / "weights" / "DDPMDenoiser.pth"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Diffusion weights not found at {weights_path}. Train the lecture diffusion notebook first."
        )

    test_dataset = MayoDataset(BOOK_ROOT / "Mayo" / "test", data_shape=args.data_shape)
    if len(test_dataset) == 0:
        raise FileNotFoundError("No Mayo test images were found. Check the course data directory.")
    if not (0 <= args.image_index < len(test_dataset)):
        raise IndexError(f"image-index must be in [0, {len(test_dataset) - 1}].")

    model = build_model(weights_path, device)

    betas = make_beta_schedule(args.num_diffusion_steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    K = operators.Blurring(
        img_shape=(args.data_shape, args.data_shape),
        kernel_type="motion",
        kernel_size=args.kernel_size,
        motion_angle=args.motion_angle,
    )

    with torch.no_grad():
        x_true = test_dataset[args.image_index].unsqueeze(0).to(device)
        y_delta = K(x_true)
        y_delta = y_delta + gaussian_noise(y_delta, noise_level=args.noise_level)

    save_tensor_image(x_true.cpu(), output_dir / "ground_truth.png")
    save_tensor_image(y_delta.cpu(), output_dir / "observation.png")
    save_kernel_image(K.kernel, output_dir / "motion_kernel.png")

    snapshot_positions = set(select_snapshot_positions(args.sample_steps - 1, args.num_snapshots))
    dps_projection_scale = args.guidance_scale if args.dps_projection_scale is None else args.dps_projection_scale

    x_dps, dps_snapshots = trace_dps(
        model=model,
        y_delta=y_delta,
        K=K,
        alpha_bars=alpha_bars,
        num_diffusion_steps=args.num_diffusion_steps,
        sigma_y=args.sigma_y,
        guidance_scale=args.guidance_scale,
        projection_scale=dps_projection_scale,
        sample_steps=args.sample_steps,
        snapshot_positions=snapshot_positions,
        device=device,
    )
    x_diffpir, diffpir_snapshots = trace_diffpir(
        model=model,
        y_delta=y_delta,
        K=K,
        alpha_bars=alpha_bars,
        num_diffusion_steps=args.num_diffusion_steps,
        tau=args.tau,
        sample_steps=args.sample_steps,
        snapshot_positions=snapshot_positions,
        device=device,
    )

    save_tensor_image(x_dps.cpu(), output_dir / "dps_final.png")
    save_tensor_image(x_diffpir.cpu(), output_dir / "diffpir_final.png")

    save_snapshot_bundle("dps", dps_snapshots, output_dir)
    save_snapshot_bundle("diffpir", diffpir_snapshots, output_dir)
    save_method_grid("DPS-style trajectory", dps_snapshots, output_dir / "dps_grid.png")
    save_method_grid("DiffPIR-style trajectory", diffpir_snapshots, output_dir / "diffpir_grid.png")
    save_combined_grid(dps_snapshots, diffpir_snapshots, output_dir / "graphical_abstract_overview.png")

    metrics = {
        "observation": {
            "mse": mse(y_delta, x_true),
            "psnr": psnr(y_delta, x_true),
            "ssim": ssim(y_delta, x_true),
        },
        "dps": {
            "mse": mse(x_dps, x_true),
            "psnr": psnr(x_dps, x_true),
            "ssim": ssim(x_dps, x_true),
        },
        "diffpir": {
            "mse": mse(x_diffpir, x_true),
            "psnr": psnr(x_diffpir, x_true),
            "ssim": ssim(x_diffpir, x_true),
        },
    }

    metadata = {
        "weights_path": str(weights_path),
        "image_index": args.image_index,
        "sample_steps": args.sample_steps,
        "num_snapshots": len(dps_snapshots),
        "num_diffusion_steps": args.num_diffusion_steps,
        "noise_level": args.noise_level,
        "sigma_y": args.sigma_y,
        "guidance_scale": args.guidance_scale,
        "dps_projection_scale": dps_projection_scale,
        "tau": args.tau,
        "kernel_size": args.kernel_size,
        "motion_angle": args.motion_angle,
        "device": device,
        "metrics": metrics,
        "notes": {
            "x_t": (
                "Current iterate shown at the selected reverse-diffusion timesteps. "
                "For visualization, every saved image in this script is mapped with its own "
                "per-image min-max normalization."
            ),
            "x0_hat": (
                "Clean-image estimate reconstructed from the denoiser output. "
                "It is also visualized with per-image min-max normalization."
            ),
            "d_t": (
                "Image-space data-consistency projection C(x0_hat, K, y). "
                "For DPS this is saved only as a visual proxy; the actual DPS iterate still follows "
                "the likelihood-gradient update used in the lecture notebook. "
                "It is visualized with per-image min-max normalization."
            ),
            "raw_values": (
                "The PNGs are display-normalized image exports. "
                "If you need the exact saved tensors rather than the visualized panels, use the saved trace.pt files."
            ),
            "diffpir_visualization": (
                "For DiffPIR, the saved abstract triplets use a forward-noised visualization state at the current timestep "
                "so that x_t looks like a genuine diffusion variable. The actual reconstruction path and reported metrics "
                "still follow the pedagogical lecture notebook update."
            ),
        },
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Observation | MSE: {metrics['observation']['mse']:.6f} | "
        f"PSNR: {metrics['observation']['psnr']:.3f} dB | "
        f"SSIM: {metrics['observation']['ssim']:.4f}"
    )
    print(
        f"DPS-style   | MSE: {metrics['dps']['mse']:.6f} | "
        f"PSNR: {metrics['dps']['psnr']:.3f} dB | "
        f"SSIM: {metrics['dps']['ssim']:.4f}"
    )
    print(
        f"DiffPIR     | MSE: {metrics['diffpir']['mse']:.6f} | "
        f"PSNR: {metrics['diffpir']['psnr']:.3f} dB | "
        f"SSIM: {metrics['diffpir']['ssim']:.4f}"
    )
    print(f"Saved graphical-abstract assets to: {output_dir}")


if __name__ == "__main__":
    main()
