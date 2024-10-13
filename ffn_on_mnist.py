import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

mnist_train = MNIST(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets/", train=True, download=False,
                    transform=ToTensor())
mnist_test = MNIST(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets/", train=False, download=False,
                   transform=ToTensor())

batch_size = 64

train_loader = DataLoader(mnist_train, batch_size=batch_size)
test_loader = DataLoader(mnist_test, batch_size=batch_size)


# Visualize Data

def visualize_images(image, label, num_images):
    plt.figure(figsize=(20, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
# The variable i is the loop counter, which starts from 0. Since subplot indices in Matplotlib start from 1, you need
# to add 1 to i to get the correct subplot index. This ensures that for the first image (when i = 0), you access the
# first subplot (index = 1), for the second image (when i = 1), you access the second subplot (index = 2), and so on.
        plt.imshow(image)
        plt.title(f"Label: {label[i].item()}")
        plt.axis('off')
    plt.show()


img, label = next(iter(train_loader))

visualize_images(img, label, num_images=5)
