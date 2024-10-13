import os
import random
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode

# Read a single image and its annotation from  the csv file

face_landmarks = pd.read_csv('faces/face_landmarks.csv')
data_dim = face_landmarks.shape
n = random.randint(0, data_dim[0])
img_name = face_landmarks.iloc[n, 0]
img_landmarks = face_landmarks.iloc[n, 1:]
# Since the data contains x and y coordinates we are converting the 1D array to 2D array. See the data for clarity.
img_landmarks = np.asarray(img_landmarks, dtype=float).reshape(-1,
                                                               2)  # 2 signifies the no of cols my data requires. -1 tells numpy to figure out the number of elements and internally calculate how many rows would be required to represent the 1D array in 2D form.

print(f"Image Name: {img_name}")
print(f"Landmarks Shape: {img_landmarks.shape}")
print(f"First Few Landmarks: {img_landmarks[:4]}")


# a simple helper function to show an image and its landmarks and use it to
# show a sample

def show_sample_image(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(5)  # pause a bit so that plots are updated


plt.figure()
show_sample_image(io.imread(os.path.join("faces/", img_name)), img_landmarks)
plt.show()


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset. This is my custom dataset class"""

    def __init__(self, csv_file, img_root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.img_root_dir = img_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            # When using PyTorch, the idx argument might be a tensor (especially when using
            # DataLoader). If idx is a PyTorch tensor, it is converted to a Python list using tolist() for easier
            # manipulation.

            idx = idx.tolist()

        # Same processing as above. img_name gets the path of the image. io.imread is used to read the image.
        # Then we define the landmarks based on landmarks csv file. Convert it to 2d array with x and y
        # coords. Each sample is a dict with image and its landmarks.

        img_name = os.path.join(self.img_root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Let's instantiate this class and iterate through the data samples:

face_dataset = FaceLandmarksDataset(csv_file="/home/ibab/PycharmProjects/Sem3_DL/faces/face_landmarks.csv",
                                    img_root_dir="/home/ibab/PycharmProjects/Sem3_DL/faces")

# self.landmarks_frame = pd.read_csv(csv_file) reads the CSV file and loads it into a pandas DataFrame (self.landmarks_frame).
# self.img_root_dir = img_root_dir stores the root directory of images.
# self.transform = transform initializes the transform attribute (set to None by default).

fig = plt.figure()

# The for loop calls the __getitem__ method of the class repeatedly to fetch individual samples.

for i, sample in enumerate(face_dataset):
    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title(f"Sample {i}")
    ax.axis('off')
    show_sample_image(**sample)

    if i == 3:
        plt.show()
        break


# One issue we can see from the above is that the samples are not of the same size. Most neural networks expect the
# images of a fixed size. Therefore, we will need to write some preprocessing code. Let’s create three transforms:
#
# Rescale: to scale the image
#
# RandomCrop: to crop from image randomly. This is data augmentation.
#
# ToTensor: to convert the numpy images to torch images (we need to swap axes).
#
# We will write them as callable classes instead of simple functions so that parameters of the transform need not be
# passed every time it’s called. For this, we just need to implement __call__ method and if required, __init__ method.

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h = new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_dim = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_dim = output_size

    # We first check whether output_size is a tuple or int. we use assert for that. If it is neither,it'll throw an
    # assertion error. If output_size is an int, we have to do square crop. So output_dim of the final image will have
    # same h and w. If output_size is a tuple, make sure it is of len 2 corresponding to h and w. Then assign output_dim
    # to output_size.

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_dim

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top:top + new_h, left:left + new_w]
        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}
    # The call method allows us to call the instance of this object as a function. Here it takes the argument 'sample'.
    # 'sample' is a dictionary of image and corresponding landmarks. We first extract the h and w from the image.
    # the new h and w are defined while instantiating object. No we define the crop window. Cropped image should have
    # size new_h, new_w. therefore the top most portion of the crop window cannot be such that top + new_h exceeds
    # image h. therefore we find a random int between 0 and h - new_h. Same logic for width. finally it should return
    # the cropped image and adjusted landmarks.


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'lamdmarks': torch.from_numpy(landmarks)}


scaler = Rescale(256)
cropper = RandomCrop(128)
processed = transforms.Compose([scaler, cropper])

# Apply each of the above transforms on sample.

fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scaler, cropper, processed]):
    processed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_sample_image(**processed_sample)
plt.show()

