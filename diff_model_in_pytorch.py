# Credits: https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# we define our network architecture, in this case a U-Net. The dim parameter specifies the number of feature maps
# before the first down-sampling, and the dim_mults parameter provides multiplicands for this value and successive down-samplings:

# model = Unet(dim=64, dim_mults=(1, 2, 4, 8))
#
# # our network architecture is defined, we need to define the Diffusion Model itself. We pass in the U-Net model that
# # we just defined along with several parameters - the size of images to generate, the number of timesteps in the
# # diffusion process, and a choice between the L1 and L2 norms.
# from inspect import signature
#
# print(signature(GaussianDiffusion.__init__))
#
# diffusion = GaussianDiffusion(
#     model,
#     image_size=128,
#     timesteps=100,   # number of steps
#     )
#
#
# # the Diffusion Model is defined, it's time to train. We generate random data to train on, and then train the
# # Diffusion Model in the usual fashion
#
# training_images = torch.randn(8, 3, 128, 128)  # 8 images, 3 channels, 128x128 dimension
# loss = diffusion(training_images)
# loss.backward()
#
# # Once the model is trained, we can finally generate images by using the sample() method of the diffusion object.
# # Here we generate 4 images, which are only noise given that our training data was random
#
# sampled_images = diffusion.sample(batch_size=4)
#
# print(sampled_images)
#
# # Assuming sampled_images is of shape (batch_size, channels, height, width)
# print(sampled_images.shape)  # This should output: torch.Size([4, 3, 128, 128])
#
# # Visualize Samples
#
# # Step 1: Normalize the image values to [0, 1] if needed (optional) (MinMax Scaling)
# # If your model outputs images with values outside the [0,1] range (e.g., between [-1, 1]), apply normalization:
# sampled_images = (sampled_images - sampled_images.min()) / (sampled_images.max() - sampled_images.min())
#
# # Step 2: Create a grid of images for better visualization (optional)
# grid = vutils.make_grid(sampled_images, nrow=2, padding=2)
#
# # Step 3: Convert the grid to a NumPy array and display it with matplotlib
# plt.figure(figsize=(8, 8))
# plt.axis("off")
# plt.imshow(grid.permute(1, 2, 0).numpy())  # PyTorch stores image data in the (C, H, W) format.
# # grid.permute(1[for H], 2[W], 0[for C] converts it to format (H, W, C) which is required for visualizing with Matplotlib)
# plt.show()

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,
).to(device)

trainer = Trainer(
    diffusion,
    folder="/home/suryadip/sem3_dl/vision_datasets/MNIST_jpg/trainingSample",  # for GPU
    train_batch_size=32,
    train_lr=2e-5,
    train_num_steps=700000,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    amp=True

)

trainer.train()

sampled_images = diffusion.sample(batch_size=4).to(device)
# Visualize Samples

# Step 1: Normalize the image values to [0, 1] if needed (optional) (MinMax Scaling)
# If your model outputs images with values outside the [0,1] range (e.g., between [-1, 1]), apply normalization:
sampled_images = (sampled_images - sampled_images.min()) / (sampled_images.max() - sampled_images.min())

# Step 2: Create a grid of images for better visualization (optional)
grid = vutils.make_grid(sampled_images, nrow=2, padding=2)

# Step 3: Convert the grid to a NumPy array and display it with matplotlib
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.imshow(grid.cpu().permute(1, 2, 0).numpy())  # PyTorch stores image data in the (C, H, W) format.
# grid.permute(1[for H], 2[W], 0[for C] converts it to format (H, W, C) which is required for visualizing with Matplotlib)
plt.show()







