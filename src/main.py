import torch as th
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

import os
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

from model import ToyModel
from diffusion import Diffusion
from utils import get_dataset

# animation
from celluloid import Camera
import imageio

# Tensorboard
from torch.utils.tensorboard import SummaryWriter


def train(args):
    SAVE_IMAGE = True
    device = "cuda" if th.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/{timestamp}")
    model_dir = f"ckpts/{timestamp}"

    dataset = get_dataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)

    model = ToyModel(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        time_emb_dim=args.time_emb_dim,
        twoD_data=args.dataset != "mnist",
    )
    model.to(device)
    diffusion = Diffusion(num_timesteps=args.num_timesteps, device=device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = LambdaLR(optimizer, lambda step: 1 - step / (args.num_epochs * len(dataloader)))

    best_loss = float("inf")
    print("Training model...")
    for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            x_0 = batch[0].to(device)
            noise = th.randn_like(x_0)
            t = th.randint(0, args.num_timesteps, (args.train_batch_size, 1), device=device)
            x_t = diffusion.diffusion(x_0, noise, t)

            pred_noise = model(x_t, t)                # 预测噪声
            loss = criterion(pred_noise, noise)       # 计算损失
            optimizer.zero_grad()                     # 清空梯度
            loss.backward()                           # 反向传播（计算梯度）
            clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()                          # 更新模型参数
            scheduler.step()                          # 更新学习率
            total_loss += loss.item()

        if SAVE_IMAGE and args.dataset == "heart":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.scatter(x_0[:, 0].cpu(), x_0[:, 1].cpu(), alpha=0.5)
            ax1.set_title("x_0")
            ax2.scatter(x_t[:, 0].cpu(), x_t[:, 1].cpu(), alpha=0.5)
            ax2.set_title("x_t")
            writer.add_figure("scatter_plot", fig, epoch)
            plt.close(fig)
            SAVE_IMAGE = False

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar("loss", avg_loss, epoch)
        writer.add_scalar("learning rate", scheduler.get_last_lr()[0], epoch)
        for name, param in model.named_parameters():
            writer.add_histogram(f"params/{name}", param, epoch)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, epoch)

        # 记录最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = model.state_dict()

    print(f"Saving model to {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    th.save(best_weights, f"{model_dir}/model_{args.dataset}.pth")
    writer.add_graph(model, [x_0, t])
    writer.add_hparams(vars(args), {"loss": best_loss})

    writer.close()
    return model_dir


def evaluate(model_dir, args):
    print("Evaluate & Save Animation")
    device = "cuda" if th.cuda.is_available() else "cpu"
    model_weights = th.load(f"{model_dir}/model_{args.dataset}.pth", map_location=device, weights_only=True)
    model = ToyModel(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        time_emb_dim=args.time_emb_dim,
        twoD_data=args.dataset != "mnist",
    )
    model.to(device)
    model.load_state_dict(model_weights)
    model.eval()

    x_t = th.randn(args.eval_batch_size, model.input_dim, device=device)
    timesteps = range(args.num_timesteps - 1, -1, -1)
    diffusion = Diffusion(num_timesteps=args.num_timesteps, device=device)
    denoise_process = []
    denoise_process.append(x_t.cpu())
    for step, t in tqdm(enumerate(timesteps), desc="Sampling"):
        t = th.full((args.eval_batch_size, 1), t, device=device)
        with th.no_grad():
            pred_noise = model(x_t, t)
            x_t = diffusion.denoise(pred_noise, t, x_t)

        if step % args.show_image_step == 0:
            denoise_process.append(x_t.cpu())

    if args.dataset != "mnist":
        # also show forward samples
        dataset = get_dataset(args.dataset, n=args.eval_batch_size)
        x_0 = dataset.tensors[0].to(device)
        diffusion_process = []
        diffusion_process.append(x_0.cpu())
        for t in range(args.num_timesteps):
            noise = th.randn_like(x_0, device=device)
            x_t = diffusion.diffusion(x_0, noise, t)
            if t % args.show_image_step == 0:
                diffusion_process.append(x_t.cpu())

        x_min, x_max = -6, 6
        y_min, y_max = -6, 6
        fig, ax = plt.subplots()
        camera = Camera(fig)

        for i, x in enumerate(diffusion_process + denoise_process):
            plt.scatter(x[:, 0], x[:, 1], alpha=0.5, s=15, color="red")
            t = i if i < args.num_timesteps else i - args.num_timesteps
            ax.text(
                0.0,
                0.95,
                f"timestep {t + 1: 4} / {args.num_timesteps}",
                transform=ax.transAxes,
            )
            ax.text(
                0.0,
                1.01,
                "Diffusion" if i < args.num_timesteps else "Denoise",
                transform=ax.transAxes,
                size=15,
            )
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.axis("scaled")
            plt.axis("off")
            camera.snap()

        animation = camera.animate(blit=True, interval=200)
        animation.save(f"{model_dir}/animation_heart.gif")

    else:
        # 让动画停留 30 帧
        n_hold_final = 30
        for _ in range(n_hold_final):
            denoise_process.append(x_t.cpu())

        denoise_process = th.stack(denoise_process, dim=0)
        denoise_process = (denoise_process.clamp(-1, 1) + 1) / 2
        denoise_process = (denoise_process * 255).type(th.uint8)
        denoise_process = denoise_process.reshape(-1, args.eval_batch_size, 28, 28)
        denoise_process = list(th.split(denoise_process, 1, dim=1))
        for i in range(len(denoise_process)):
            denoise_process[i] = denoise_process[i].squeeze(1)
        denoise_process = th.cat(denoise_process, dim=-1)
        imageio.mimsave(f"{model_dir}/animation_mnist.gif", list(denoise_process), fps=5)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="heart", choices=["heart", "mnist"])
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-timesteps", type=int, default=200)
    parser.add_argument("--time-emb-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--show-image-step", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_dir = train(args)
    evaluate(model_dir, args)
