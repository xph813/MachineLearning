# 配置文件
import torch

config = {

    "dataset_list": ["MNIST", ], # "CIFAR10"
    "img_channels": {"MNIST": 1, "CIFAR10": 3},
    "img_size": {"MNIST": 32, "CIFAR10": 32},


    "T": 1000,
    "batch_size": {"MNIST": 128, "CIFAR10": 128},
    "lr": 2e-4,
    "epochs": {"MNIST": 40, "CIFAR10": 40},

    "device": "xpu" if torch.xpu.is_available() else "cpu",

    "model_save_path": {"MNIST": "./ddpm_mnist_original.pth", "CIFAR10": "./ddpm_cifar10_original.pth"},
    "loss_log_path": {"MNIST": "./mnist_loss_original.txt", "CIFAR10": "./cifar10_loss_original.txt"},
    "vis_save_path": "./",

    "grid_size": {"MNIST": (10, 10), "CIFAR10": (5, 5)},
    "denoising_seq_num": 4,
    "denoising_step_num": 7
}