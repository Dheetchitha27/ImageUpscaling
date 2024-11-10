# utils.py
import matplotlib.pyplot as plt
import numpy as np
import torch

def show_results(lr, sr, hr):
    """Display low-res, super-res, and high-res images side by side."""
    images = [lr.squeeze(0).permute(1, 2, 0), sr.squeeze(0).permute(1, 2, 0), hr.squeeze(0).permute(1, 2, 0)]
    titles = ['Low-Resolution', 'Super-Resolution', 'High-Resolution']

    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img.detach().numpy())
        plt.title(titles[i])
        plt.axis('off')
    plt.show()
