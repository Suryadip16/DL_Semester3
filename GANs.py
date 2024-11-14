import torch
from torch import nn

import math
import matplotlib.pyplot as plt

torch.manual_seed(111)

# The training data is composed of pairs (x₁, x₂) so that x₂ consists of the value of the sine of
# x₁ for x₁ in the interval from 0 to 2π

train_data_length = 1024
train_data = torch.zeros((train_data_length, 2))
train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)  # you use the first column of train_data
# to store random values in the interval from 0 to 2π. torch.rand returns floats between 0 and 1.
# Scaling the returned values by 2*pi gives you random values within range (0, 2pi).
train_data[:, 1] = torch.sin(train_data[:, 0])
train_labels = torch.zeros(train_data_length)  # need a tensor of labels, which are required by
# PyTorch’s data loader. Since GANs make use of unsupervised learning techniques, the labels can be
# anything. They won’t be used, after all.
train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]  # create train_set as a
# list of tuples, with each row of train_data and train_labels represented in each tuple as expected by
# PyTorch’s data loader.
plt.plot(train_data[:, 0], train_data[:, 1], ".")
plt.show()

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)


# we have a 2 dimensional input of x1 and its corresponding sine x2. the discriminator's job is to
# classify whether the pair (x1, x2) is a "real" data point (belonging to the sine fn) or "fake" data point
# (not belonging to the sine fn).


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

# The generator will receive random points (z₁, z₂), and output a two-dimensional output that must provide
# (x̃₁, x̃₂) points resembling those from the training data.


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        output = self.model(x)
        return output


generator = Generator()
discriminator = Discriminator()

lr = 0.001
num_epochs = 300
loss_function = nn.BCELoss()

# The binary cross-entropy function is a suitable loss function for training the discriminator because it
# considers a binary classification task. It’s also suitable for training the generator since it feeds its
# output to the discriminator, which provides a binary observable output.

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        # Data for training the discriminator:

        real_samples_labels = torch.ones((batch_size, 1))
        # because i am trying to fool the generator into optimizing towards p(real) = 1

        latent_space_samples = torch.randn((batch_size, 2))  # generate random samples (z1, z2).
        # Each batch will contain batch size no of data points. Hence, batch_size num of rows and 2 cols.

        generated_samples = generator(latent_space_samples)
        # Forward pass of generator to generate samples from random data. The first time when this happens
        # it serves the purpose of initialization.

        generated_samples_labels = torch.zeros((batch_size, 1))
        # label = zero indicates fake data. So there are real samples from our train loader and fake
        # samples that our generator generated (generated_samples).

        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))
        # We put the real and fake samples (generated_samples) together along with their labels as 1 and 0
        # respectively. Now the data is ready to train the discriminator.

        # Train the Discriminator:

        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training generator:
        latent_space_samples = torch.randn((batch_size, 2))
        # When training the generator, we want it to learn to create samples that can consistently fool
        # the discriminator. By generating a new batch of latent_space_samples for the generator, we
        # ensure that it doesn’t simply “overfit” or learn to improve on the same set of generated data
        # from the discriminator’s training step.

        # training the generator:
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        # Generate samples from random data

        output_discriminator_generated = discriminator(generated_samples)
        # generated samples go to discriminator for it to classify them as fake o real.

        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        # We want the generator to learn to generate data that can consistently fool the discriminator.
        # This in turn would mean that it is generating data close to the training data's distribution.
        # Therefore, here we use the real sample labels in the loss, because we want the generator to
        # think that all the training data is real.

        loss_generator.backward()
        optimizer_generator.step()

        # At this point the parameters of the generator are updated. therefore during the next iteration
        # we create another set of random data (latent_space_samples), pass it to the generator to create
        # another set of generated samples using the weights learned here.

        # Show Loss:
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

# here training phase ends.

# now we try and generate samples using our GAN
latent_space_samples = torch.randn(500, 2)
generated_samples = generator(latent_space_samples)

generated_samples = generated_samples.detach()
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
plt.show()




        

