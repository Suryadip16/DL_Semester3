import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Flatten, Linear, Softmax
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchvision.datasets import EMNIST
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using Device {device}')

# Define Transform for images
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

emnist_train = EMNIST(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets/", split="balanced", train=True,
                      download=False, transform=transform)
emnist_test = EMNIST(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets/", split="balanced", train=False,
                     download=False, transform=transform)

print("Basic Dataset Information:")
print(emnist_train)
print(emnist_train.data.size()[0])
print(emnist_test)
print(emnist_test.data.size())
print("Number of Classes: ")
# print(len(torch.unique(emnist_train.targets)))
labels = [target for _, target in emnist_train]
print(len(torch.unique(torch.tensor(labels))))
# Visualize images

# figure = plt.figure(figsize=(10, 8))
# rows, cols = 5, 5
# for i in range(1, rows * cols + 1):
#     sample_idx = np.random.randint(0, emnist_train.data.size()[0], )
#     img, label = emnist_train[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(label)
#     plt.axis('off')
#     plt.imshow(img.squeeze(), cmap='gray')
# plt.show()

# Load dataset onto Dataloader
train_loader = DataLoader(emnist_train, batch_size=100, shuffle=True, num_workers=3)
test_loader = DataLoader(emnist_train, batch_size=100, shuffle=True, num_workers=3)
print(len(train_loader.dataset))


# Define CNN Architecture:

class my_CNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(my_CNN, self).__init__(*args, **kwargs)

        self.model = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            # padding = 2 usually corresponds to same padding (input size = outsize)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # a 2x2 maxpool like this usually halves the dimesnions. SO 28x28 becomes 14x14

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # So we had started with 28x28, then due to maxpool of 2x2 we end up with 14x14 and now again a maxpool
            # of 2x2 we end up with 7x7.
            nn.Flatten(),
            nn.Linear(in_features=32 * 7 * 7, out_features=47)

        )

        # self.model = nn.Sequential(
        #     Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding="same", stride=1),
        #     ReLU(),
        #     Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same", stride=1),
        #     ReLU(),
        #     MaxPool2d(kernel_size=2),
        #     # Now we will have a 14 x 14 img dimension
        #
        #     Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same", stride=1),
        #     ReLU(),
        #     Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same", stride=1),
        #     ReLU(),
        #     MaxPool2d(kernel_size=2),
        #     # Now we will have a 7 x 7 img dimension
        #     Flatten(),
        #     # Now we have a 7 x 7 x 128 vector. Therefore, input for next layer is 6272
        #     Linear(in_features=7 * 7 * 128, out_features=1024),
        #     ReLU(),
        #     Linear(in_features=1024, out_features=1024),
        #     ReLU(),
        #     Linear(in_features=1024, out_features=47),
        # )

    def forward(self, x):
        res = self.model(x)
        return res


# Instantiate NN class:
cnn = my_CNN().to(device)
cnn = nn.DataParallel(cnn)
print(cnn)

# Define Loss Function
loss_fn = nn.CrossEntropyLoss()

# Define Optimizer
optimizer = optim.Adam(params=cnn.parameters(), lr=0.01)


def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)  # Total Sample Size
    model.train()
    running_loss = 0.0

    # loop over data for training
    for batch, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # Make the param grads zero.
        fp_res = model(images)  # FP Result after data runs through model
        loss = loss_func(fp_res, labels)  # Calculate Loss
        loss.backward()  # Backprop
        optimizer.step()  # update Weights based on Backprop
        running_loss += loss.item()  # Accumulating the training loss

        if batch % 100 == 0:
            num_samples_processed = len(images) * (batch + 1)
            print(f"Loss: {loss} [{num_samples_processed}/{size}]")

    avg_loss_epoch = running_loss / len(dataloader)
    print(f"Average Loss Across all batches in Epoch: {avg_loss_epoch}")
    return avg_loss_epoch


def test(dataloader, model, loss_func):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    correct_preds, test_loss = 0.0, 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            pred = model(images)
            test_loss += loss_func(pred, labels).item()
            correct_preds += (pred.argmax(1) == labels).type(torch.float).sum().item()
        final_test_loss = test_loss / num_batches  # Average Test Loss over all batches
        model_acc = correct_preds / size

    print(f"Test Loss(Averaged Over All Batches in Epoch): {final_test_loss}")
    print(print(f"Model Accuracy For Epoch: {model_acc * 100} %"))
    return model_acc


loss_over_epochs = []
accuracy_over_epochs = []
epochs = []
num_epochs = 10
for i in range(num_epochs):
    print("***************************************************")
    print(f"Epoch {i + 1}")
    l = train(dataloader=train_loader, model=cnn, loss_func=loss_fn, optimizer=optimizer)
    a = test(dataloader=test_loader, model=cnn, loss_func=loss_fn)
    loss_over_epochs.append(l)
    accuracy_over_epochs.append(a)
    epochs.append(i)

print("Done!")

# PLot Training Loss and Model Accuracy wrt Epochs

# Create a figure with 2 subplots (1 row, 2 columns)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Adjust figsize as needed
#
# # Plot average training loss
# ax1.plot(epochs, loss_over_epochs, marker='o', color='b')  # Add marker for better visibility
# ax1.set_title("Average Training Loss vs Epochs")
# ax1.set_xlabel("Epochs")
# ax1.set_ylabel("Training Loss")
# ax1.grid(True)  # Optional: add grid for better readability
#
# # Plot model accuracy
# ax2.plot(epochs, accuracy_over_epochs, marker='o', color='g')  # Add marker for better visibility
# ax2.set_title("Model Accuracy vs Epochs")
# ax2.set_xlabel("Epochs")
# ax2.set_ylabel("Model Accuracy")
# ax2.grid(True)  # Optional: add grid for better readability
#
# # Adjust layout for better spacing
# plt.tight_layout()
#
# # Show the plots
# plt.show()


# Visualize Model Performance

def visualize_model_performance(dataloader, model, num_preds):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(dataloader))
        pred = model(images)
        pred_label = pred.argmax(1)
        sample_index = np.random.choice(len(images), num_preds, replace=False)

        figure = plt.figure(figsize=(10, 8))
        for i, sample_idx in enumerate(sample_index):
            image = images[sample_idx].squeeze()
            label = labels[sample_idx].item()
            predicted_label = pred_label[sample_idx].item()

            # Add a subplot for each image
            ax = figure.add_subplot(1, num_preds, i + 1)
            ax.imshow(image, cmap='gray')
            ax.set_title(f"GT: {label} | Pred: {predicted_label}")
            ax.axis('off')
        plt.show()


visualize_model_performance(dataloader=test_loader, model=cnn, num_preds=5)
