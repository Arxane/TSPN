import io
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_to_image(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to numpy array
    image = Image.open(buf).convert("RGBA")
    image_np = np.array(image)  # [H, W, 4]
    image_np = image_np.transpose((2, 0, 1))  # [4, H, W]
    image_tensor = torch.tensor(image_np, dtype=torch.uint8).unsqueeze(0)  # [1, 4, H, W]
    return image_tensor


def set_to_plot(raw, true, sampled, predicted):
    num_elements = true.shape[0]
    img_size = 2
    plots_per_sample = 4

    figure = plt.figure(figsize=(img_size * plots_per_sample, num_elements * img_size))
    plt.grid(False)
    plt.tight_layout()

    for i in range(num_elements):
        if raw is not None:
            plt.subplot(num_elements, plots_per_sample, i * plots_per_sample + 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(raw[i], cmap='gray')

        plt.subplot(num_elements, plots_per_sample, i * plots_per_sample + 2)
        x_true = true[i, :, 1].detach().cpu().numpy()
        y_true = true[i, :, 0].detach().cpu().numpy()
        plt.scatter(x_true, y_true, s=5)
        plt.axis((0, 1, 1, 0))

        plt.subplot(num_elements, plots_per_sample, i * plots_per_sample + 3)
        x_samp = sampled[i, :, 1].detach().cpu().numpy()
        y_samp = sampled[i, :, 0].detach().cpu().numpy()
        plt.scatter(x_samp, y_samp, s=5)
        plt.axis((0, 1, 1, 0))

        plt.subplot(num_elements, plots_per_sample, i * plots_per_sample + 4)
        x_pred = predicted[i, :, 1].detach().cpu().numpy()
        y_pred = predicted[i, :, 0].detach().cpu().numpy()
        plt.scatter(x_pred, y_pred, s=5)
        plt.axis((0.0, 1.0, 1.0, 0.0))

    return figure


if __name__ == "__main__":
    B, N = 2, 100
    true = torch.rand(B, N, 2)
    sampled = torch.rand(B, N, 2)
    predicted = torch.rand(B, N, 2)

    raw_imgs = [np.random.rand(28, 28) for _ in range(B)]

    fig = set_to_plot(raw_imgs, true, sampled, predicted)
    img_tensor = plot_to_image(fig)

    print(f"Image tensor shape: {img_tensor.shape}")  # (1, 4, H, W)
