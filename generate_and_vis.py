# 使用训练好的模型生成图片（sample）与可视化对比图
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from config import config
from model import get_model
from data_utils import get_real_samples

# 预处理
def preprocess_img_for_vis(img_tensor):
    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(img_tensor).__name__} instead")
    if img_tensor.numel() == 0:
        raise ValueError("Input tensor is empty")

    img = img_tensor.detach().cpu().numpy()
    while len(img.shape) > 3:
        img = img[0]

    # 反归一化：从[-1,1]→[0,1]
    img = (img + 1) / 2.0
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.shape[0] == 1:
        img = img.squeeze(0)

    img = np.clip(img, 0, 1)
    if len(img.shape) not in [2, 3]:
        raise TypeError(f"Invalid shape {img.shape} (expected 2D/3D)")
    return img

# 反向去噪
def sample(model, dataset_name, num_samples, T, device):
    img_channels = config["img_channels"][dataset_name]
    img_size = config["img_size"][dataset_name]
    # 初始化噪声xt（纯纯的高斯噪声）
    xt = torch.randn((num_samples, img_channels, img_size, img_size), device=device)
    # 噪声调度
    beta, alpha, alpha_bar = get_cosine_schedule(T, device)
    # 反向去噪循环
    for t in tqdm(range(T-1, -1, -1), desc=f"Sampling {dataset_name}"):
        t_tensor = torch.tensor([t] * num_samples, device=device)
        # 预测噪声
        eps_pred = model(xt, t_tensor)
        # 采样公式
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        beta_t = beta[t]
        # 计算均值
        if t > 0:
            z = torch.randn_like(xt, device=device)
        else:
            z = torch.zeros_like(xt, device=device)  # t=0时无噪声
        # 反向去噪更新
        xt = (1 / torch.sqrt(alpha_t)) * (xt - ( (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) ) * eps_pred) + torch.sqrt(beta_t) * z
    return xt

# 余弦噪声调度（与训练一致）
def get_cosine_schedule(T, device):
    beta_start = 0.0001
    beta_end = 0.02
    beta = torch.linspace(beta_start, beta_end, T, device=device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar

# 绘制样本网格
def plot_sample_grid(samples, dataset_name, grid_size):
    row, col = grid_size
    fig, axes = plt.subplots(row, col, figsize=(col*2, row*2))
    for i in range(row):
        for j in range(col):
            idx = i * col + j
            if idx >= len(samples):
                break
            img = preprocess_img_for_vis(samples[idx:idx+1])
            axes[i, j].imshow(img, cmap="gray" if dataset_name == "MNIST" else None)
            axes[i, j].axis("off")
    plt.suptitle(f"DDPM Generated Samples ({dataset_name})", fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(config["vis_save_path"], f"{dataset_name.lower()}_generated.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

# 绘制真实VS生成对比
def plot_real_vs_generated_comparison(real_imgs, generated_imgs, dataset_name):
    grid_row, grid_col = config["grid_size"][dataset_name]
    num_samples = grid_row * grid_col
    real_imgs = real_imgs[:num_samples]
    generated_imgs = generated_imgs[:num_samples]

    # 直接创建1行2列的画布，避免嵌套subplot
    fig, (ax_real, ax_gen) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f"Real vs Generated ({dataset_name})", fontsize=16)

    # ========== 绘制真实样本（左边区域） ==========
    ax_real.set_title("Real Samples", fontsize=14)
    ax_real.axis("off")
    # 在左边大轴内创建网格子图
    real_grid = ax_real.inset_axes([0, 0, 1, 1])  # 占满左边区域
    real_grid.axis("off")
    for i in range(grid_row):
        for j in range(grid_col):
            idx = i * grid_col + j
            if idx >= len(real_imgs):
                break
            # 在网格内创建子图
            sub_ax = real_grid.inset_axes([j/grid_col, 1 - (i+1)/grid_row, 1/grid_col, 1/grid_row])
            img = preprocess_img_for_vis(real_imgs[idx:idx+1])
            sub_ax.imshow(img, cmap="gray" if dataset_name == "MNIST" else None)
            sub_ax.axis("off")

    # ========== 绘制生成样本（右边区域） ==========
    ax_gen.set_title("Generated Samples", fontsize=14)
    ax_gen.axis("off")  # 隐藏大轴的坐标轴
    # 在右边大轴内创建网格子图
    gen_grid = ax_gen.inset_axes([0, 0, 1, 1])  # 占满右边区域
    gen_grid.axis("off")
    for i in range(grid_row):
        for j in range(grid_col):
            idx = i * grid_col + j
            if idx >= len(generated_imgs):
                break
            # 在网格内创建子图
            sub_ax = gen_grid.inset_axes([j/grid_col, 1 - (i+1)/grid_row, 1/grid_col, 1/grid_row])
            img = preprocess_img_for_vis(generated_imgs[idx:idx+1])
            sub_ax.imshow(img, cmap="gray" if dataset_name == "MNIST" else None)
            sub_ax.axis("off")

    # 保存图像
    save_path = os.path.join(config["vis_save_path"], f"{dataset_name.lower()}_real_vs_generated.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# 绘制去噪过程
def plot_denoising_process(denoising_steps, dataset_name):
    seq_num = config["denoising_seq_num"]
    step_num = config["denoising_step_num"]
    T = config["T"]
    # 计算要展示的t值（和采样时的record_steps对应）
    show_t_list = [T - 1 - i*(T//(step_num-1)) for i in range(step_num)]
    show_t_list = [max(0, s) for s in show_t_list]  # 避免负数
    
    fig, axes = plt.subplots(seq_num, step_num, figsize=(step_num*2, seq_num*2))
    for seq_idx in range(seq_num):
        steps_imgs = denoising_steps[seq_idx]
        for step_idx in range(step_num):
            img_tensor = steps_imgs[step_idx]
            img = preprocess_img_for_vis(img_tensor.unsqueeze(0))
            axes[seq_idx, step_idx].imshow(img, cmap="gray" if dataset_name == "MNIST" else None)
            axes[seq_idx, step_idx].axis("off")
            # 标题对应正确的t值
            axes[seq_idx, step_idx].set_title(f"t={show_t_list[step_idx]}", fontsize=8)
    plt.suptitle(f"DDPM Denoising Process ({dataset_name})", fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(config["vis_save_path"], f"{dataset_name.lower()}_denoising_process.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

# 记录去噪过程
def sample_with_denoising_process(model, dataset_name, num_seqs, T, device):
    img_channels = config["img_channels"][dataset_name]
    img_size = config["img_size"][dataset_name]
    denoising_steps = []
    beta, alpha, alpha_bar = get_cosine_schedule(T, device)
    step_num = config["denoising_step_num"]
    
    # 明确要记录的关键t值（从T-1到0，均匀取step_num个点）
    record_t = [int(T - 1 - i*(T/(step_num-1))) for i in range(step_num)]
    record_t = [max(0, t) for t in record_t]  # 确保t≥0

    for seq_idx in range(num_seqs):
        # 1. 强制初始化：t=T-1时的「纯高斯噪声」（DDPM生成的唯一正确起点）
        xt = torch.randn((1, img_channels, img_size, img_size), device=device)
        seq_steps = []

        # 2. 反向去噪循环：t从T-1 → 0（减噪声，xt从纯噪声→干净数字）
        for t in tqdm(range(T-1, -1, -1), desc=f"Sampling Seq {seq_idx+1}/{num_seqs}"):
            t_tensor = torch.tensor([t], device=device)
            eps_pred = model(xt, t_tensor)
            alpha_t = alpha[t]
            alpha_bar_t = alpha_bar[t]
            beta_t = beta[t]

            z = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
            # 反向去噪公式（严格执行）
            xt = (1 / torch.sqrt(alpha_t)) * (xt - ((1 - alpha_t)/torch.sqrt(1 - alpha_bar_t)) * eps_pred) + torch.sqrt(beta_t) * z

            # 3. 仅记录关键t值对应的图像
            if t in record_t:
                seq_steps.append(xt.clone())

        # 4. 关键：将记录的步骤按「t从大到小」排序（对应图中从左到右）
        # （因为循环是t从大到小，所以seq_steps已经是t从大到小的顺序，无需反转）
        denoising_steps.append(seq_steps)
    return denoising_steps

if __name__ == "__main__":
    import numpy as np
    # 遍历数据集生成
    for dataset in config["dataset_list"]:
        # 加载模型
        model = get_model(dataset)
        model_path = config["model_save_path"][dataset]
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=config["device"]))
            model.eval()
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model {model_path} not found, skip {dataset}")
            continue

        # 配置
        T = config["T"]
        device = config["device"]
        grid_row, grid_col = config["grid_size"][dataset]
        num_samples = grid_row * grid_col

        # 生成样本
        with torch.no_grad():
            generated_imgs = sample(model, dataset, num_samples, T, device)
            # 正确调用：删去多余的 T//(config["denoising_step_num"]-1) 参数
            denoising_steps = sample_with_denoising_process(model, dataset, config["denoising_seq_num"], T, device)
            # 获取真实样本
            real_imgs = get_real_samples(dataset, num_samples)

        # 可视化
        plot_sample_grid(generated_imgs, dataset, (grid_row, grid_col))
        plot_real_vs_generated_comparison(real_imgs, generated_imgs, dataset)
        plot_denoising_process(denoising_steps, dataset)

    print("All generation and visualization completed!")