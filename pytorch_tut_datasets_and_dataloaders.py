import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Loading MNIST data

training_data = datasets.FashionMNIST(root="data", train=True, download=True,
                                      transform=ToTensor())

test_data = datasets.FashionMNIST(root="data", train=False, download=True,
                                  transform=ToTensor())

# Iterating and Visualizing

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() # this method uses torch.
    # sample_idx = random.randint(0, len(training_data) - 1) # this will give you the same outcome as above.
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()




