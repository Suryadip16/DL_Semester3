import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Check Device Configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device {device}")

# Load the data

mnist_train = MNIST(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets/", train=True, download=False,
                    transform=ToTensor())

mnist_test = MNIST(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets/", train=False, download=False,
                   transform=ToTensor())
print("Dataset Basic Information: ")
print(mnist_train)
print(mnist_test)
print("Train Data Size:")
print(mnist_train.data.size())
print("Test Data Size:")
print(mnist_test.data.size())

# Visualize the image
# image, lbl = mnist_train[5]
# plt.imshow(image.squeeze(), cmap='gray')
# plt.title(lbl)
# plt.axis('off')
# plt.show()

# figure = plt.figure(figsize=(10, 8))
# cols, rows = 5, 5
# for i in range(1, cols * rows + 1):
#     sample_idx = np.random.randint(low=0, high=len(mnist_train))
#     img, label = mnist_train[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(label)
#     plt.axis('off')
#     plt.imshow(img.squeeze(), cmap='gray')
# plt.show()

# Load data onto dataloader

train_loader = DataLoader(dataset=mnist_train, batch_size=100, shuffle=True, num_workers=3)
test_loader = DataLoader(dataset=mnist_test, batch_size=100, shuffle=True, num_workers=3)


# Define the CNN

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
            # So we had started with 298x28, then due to maxpool of 2x2 we end up with 14x14 and now again a maxpool
            # of 2x2 we end up with 7x7.
            nn.Flatten(),
        )
        # define an FC linear layer
        self.out = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x):
        x = self.model(x)  # This will give me the reduced feature map
        output = self.out(x)  # This passes the output of the ConvNet through an FC layer.
        return output


# Instantiating class Object

cnn = my_CNN()
print(cnn)

# Define loss function:
# Several Loss Functions and When to use what:
# Classification Tasks:
#
# Multi-Class Classification: CrossEntropyLoss
# Binary Classification (probabilities): BCELoss
# Binary Classification (logits): BCEWithLogitsLoss
# Regression Tasks:
#
# Continuous Value Predictions (sensitive to outliers): MSELoss
# Continuous Value Predictions (less sensitive to outliers): L1Loss or SmoothL1Loss
# Other Special Tasks:
#
# Divergence Between Distributions: KLDivLoss
# SVM-style Classification: HingeEmbeddingLoss

# define loss function
loss_func = nn.CrossEntropyLoss()

# define optimization function
optimizer = optim.Adam(cnn.parameters(), lr=0.01)

print(optimizer)


# define a train function

def train(model, dataloader, optimizer, loss_fn):
    size = len(dataloader.dataset)
    model.train()  # put model in training mode
    running_loss = 0.0
    # loop over data for training
    for batch, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # zero the parameter gradients
        pred = model(images)  # remember we are returning both the output and reduced feature map of the images. But
        # for training, we only want the output.
        loss = loss_fn(pred, labels)
        loss.backward()  # backprop which calculates gradients of the loss wrt each parameter in the network.
        optimizer.step()  # updates the model's parameters using the gradients computed during backpropagation.
        running_loss += loss.item()

        if (batch + 1) % 100 == 0:
            num_samples_processed = (batch + 1) * len(images)  # num of samples processed so far
            print(f"Loss: {loss.item()} [{num_samples_processed:>5d}/{size:>5d}]")
    # Average running loss for the entire epoch
    avg_loss = running_loss / len(dataloader)
    print(f"Training loss (average over epoch): {avg_loss:.4f}")
    return avg_loss


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct_preds = 0.0, 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            pred = model(images)
            test_loss += loss_fn(pred, labels).item()
            correct_preds += (pred.argmax(1) == labels).type(torch.float).sum().item()
        test_loss /= num_batches  # Average Test Loss over all batches
        correct_preds /= size  # Model Accuracy
        print(f"Average Test Loss: {test_loss:.4f}")
        print(f"Model Accuracy: {correct_preds * 100:.2f}%")


num_epochs = 10
epochs = []
epoch_losses = []
for i in range(num_epochs):
    print("***************************************************************")
    print(f"Epoch {i + 1}")
    avg_loss_epoch = train(model=cnn, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_func)
    epoch_losses.append(avg_loss_epoch)
    epochs.append(i + 1)
    test(dataloader=test_loader, model=cnn, loss_fn=loss_func)

print("Done!")

plt.plot(epochs, epoch_losses)
plt.xlabel("Epoch")
plt.ylabel("Average Loss (Across All Batches)")
plt.title("Training Loss vs Epochs")
plt.show()


def pred_and_visualise(model, dataloader, num_preds):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(dataloader))
        pred = model(images)
        pred_labels = pred.argmax(1)
        sample_idx = np.random.choice(len(images), num_preds, replace=False)  # Choose random samples
        # Plot the selected images with Ground Truth and Predicted labels
        figure = plt.figure(figsize=(10, 5))
        for i, sample_index in enumerate(sample_idx):
            image = images[sample_index].squeeze()
            ground_truth = labels[sample_index].item()
            predicted = pred_labels[sample_index].item()

            # Add a subplot for each image
            ax = figure.add_subplot(1, num_preds, i + 1)
            ax.imshow(image, cmap='gray')
            ax.set_title(f"GT: {ground_truth} | Pred: {predicted}")
            ax.axis('off')
        plt.show()


pred_and_visualise(model=cnn, dataloader=test_loader, num_preds=4)
