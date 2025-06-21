import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.dec1(F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True))
        return x3


if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == 'cuda':
        print("GPU name:", torch.cuda.get_device_name(0))

    # Instantiate model
    model = UNet().to(device)

    # Simulate large batch of high-res images
    batch_size = 16
    height = 512
    width = 512
    input_tensor = torch.randn(batch_size, 1, height, width).to(device)

    # Warm-up
    print("Warming up GPU...")
    for _ in range(3):
        _ = model(input_tensor)
    torch.cuda.synchronize()

    # Benchmark loop
    print("Benchmarking...")
    start_time = time.time()
    for _ in range(10):
        output = model(input_tensor)
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    print(f"Avg inference time per batch (10 runs): {elapsed / 10:.4f} seconds")

    # Confirm device placement
    print("Model on:", next(model.parameters()).device)
    print("Input on:", input_tensor.device)
    print("Output shape:", output.shape)

    # Visualize one sample input/output
    input_sample = input_tensor[0, 0].cpu().detach().numpy()
    output_sample = output[0, 0].cpu().detach().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(input_sample, cmap='gray')
    axs[0].set_title('Input')
    axs[1].imshow(output_sample, cmap='gray')
    axs[1].set_title('U-Net Output')
    for ax in axs:
        ax.axis('off')
    plt.suptitle("U-Net Inference Visualization")
    plt.tight_layout()
    plt.show()

