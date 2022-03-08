# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:38:04 2022

@author: Wasik
"""

from torchvision import transforms
import torch as ch
import torch.nn as nn
import numpy as np
import json
import time
from argparse import ArgumentParser
from tools.datasets import ImageNet, ImageNet9
from tools.model_utils import make_and_restore_model, eval_model
from pathlib import Path, PureWindowsPath
from PIL import Image
import matplotlib.pyplot as plt

n_model = "normal"

fp=Path(r'C:\Users\Wasik\MAP583\Project\backgrounds_challenge-data\Nos_classifieurs', n_model+'.pt')

model = ch.load(fp)
model.eval()

img = Image.open(Path(r'C:\Users\Wasik\MAP583\Project\backgrounds_challenge-data\Data\in9l\train\05_insect\n02165456_6664.JPEG'))

convert_tensor = transforms.Compose([
                transforms.CenterCrop(256),
                transforms.ToTensor()
            ])

img = convert_tensor(img)
img = ch.unsqueeze(img,0)
img.requires_grad_()


output = model(img)

output_idx = output.argmax()
output_max = output[0, output_idx]

# Do backpropagation to get the derivative of the output based on the image
output_max.backward()

print(img)
print(img.size())

print(output)
print(output_idx)
print(output_max)


# Retireve the saliency map and also pick the maximum value from channels on each pixel.
# In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)

saliency, _ = ch.max(img.grad.data.abs(), dim=1) 
saliency = transforms.ColorJitter(brightness=1, contrast=1, saturation=0, hue=0)(saliency)
saliency = saliency.reshape(256, 256)

# Reshape the image
image = img.reshape(-1, 256, 256)

# Visualize the image and the saliency map
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
ax[0].axis('off')
ax[1].imshow(saliency.cpu(), cmap='hot')
ax[1].axis('off')
plt.tight_layout()
fig.suptitle('The Image and Its Saliency Map')
plt.show()

