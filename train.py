# 训练模块
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from config import config
from model import get_model

def get_cosine_schedule(T, device):
    beta_start = 0.0001
    beta_end = 0.02

    beta = torch.linspace(beta_start, beta_end, T, device=device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar

def forward_diffusion(x0, t, alpha_bar, device):
    batch_size = x0.shape[0]
    eps = torch.randn_like(x0, device=device)
    # xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
    alpha_bar_t = alpha_bar[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps
    return xt, eps

# 训练函数
def train_dataset(dataset_name):
    # 配置加载
    img_channels = config["img_channels"][dataset_name]
    img_size = config["img_size"][dataset_name]
    batch_size = config["batch_size"][dataset_name]
    epochs = config["epochs"][dataset_name]
    T = config["T"]
    device = config["device"]
    model_path = config["model_save_path"][dataset_name]
    loss_log_path = config["loss_log_path"][dataset_name]

    # 数据加载
    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 模型初始化
    model = get_model(dataset_name)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # 噪声调度
    beta, alpha, alpha_bar = get_cosine_schedule(T, device)

    # 损失
    loss_log = []

    # 训练
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Training {dataset_name} Epoch {epoch+1}/{epochs}")
        for x0, _ in pbar:
            x0 = x0.to(device)
            batch_size = x0.shape[0]

            # 随机采样时间步t
            t = torch.randint(0, T, (batch_size,), device=device)

            # 前向加噪
            xt, eps = forward_diffusion(x0, t, alpha_bar, device)

            # 模型预测
            eps_pred = model(xt, t)

            # 计算损失
            loss = criterion(eps_pred, eps)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 保存损失
            total_loss += loss.item()
            pbar.set_postfix({"batch_loss": loss.item()})

        # 平均损失
        avg_loss = total_loss / len(dataloader)
        loss_log.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

        # 保存loss的log
        with open(loss_log_path, "w") as f:
            for l in loss_log:
                f.write(f"{l}\n")

        # 每20轮保存一次模型
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # 开练
    for dataset in config["dataset_list"]:
        train_dataset(dataset)
    print("All datasets training completed!")