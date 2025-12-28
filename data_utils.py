# 数据工具（与原有逻辑一致，兼容复刻代码）
import torch
from torchvision import datasets, transforms
from config import config

def get_dataloader(dataset_name):
    img_size = config["img_size"][dataset_name]
    batch_size = config["batch_size"][dataset_name]

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
    return dataloader

def get_real_samples(dataset_name, num_samples):
    dataloader = get_dataloader(dataset_name)
    real_imgs_list = []
    for imgs, _ in dataloader:
        real_imgs_list.append(imgs)
        total_collected = sum([img_batch.shape[0] for img_batch in real_imgs_list])
        if total_collected >= num_samples:
            break
    all_real_imgs = torch.cat(real_imgs_list, dim=0)
    real_imgs = all_real_imgs[:num_samples]
    return real_imgs