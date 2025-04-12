import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable

# Constants for preprocessing the image data
INPUT_DIM = 224         # Target spatial dimension (224x224 after cropping)
MAX_PIXEL_VAL = 255     # Maximum pixel value after scaling (0-255 range)
MEAN = 58.09            # Mean value for normalization
STDDEV = 49.73          # Standard deviation for normalization

# Custom Dataset class to load and preprocess medical imaging volumes
class Dataset(data.Dataset):
    def __init__(self, datadirs, diagnosis, use_gpu):
        super().__init__()  # Initialize the parent Dataset class
        self.use_gpu = use_gpu  # Flag to determine if GPU is used

        label_dict = {}  # Dictionary to map file paths to binary labels
        self.paths = []  # List to store paths to volume files

        # Read metadata.csv to create a mapping of paths to labels
        for i, line in enumerate(open('metadata.csv').readlines()):
            if i == 0:  # Skip header row
                continue
            line = line.strip().split(',')  # Split CSV line into columns
            path = line[10]  # Column 10 contains the file path
            label = line[2]  # Column 2 contains the diagnosis value
            # Convert diagnosis to binary label (1 if > threshold, 0 otherwise)
            label_dict[path] = int(int(label) > diagnosis)

        # Collect all file paths from the provided directories
        for dir in datadirs:
            for file in os.listdir(dir):
                self.paths.append(dir + '/' + file)  # Full path to each volume file

        # Create a list of labels corresponding to each file path
        self.labels = [label_dict[path[6:]] for path in self.paths]  # Strip prefix (e.g., "volXX/")

        # Calculate class weights for imbalanced data
        neg_weight = np.mean(self.labels)  # Proportion of positive labels (mean since labels are 0/1)
        self.weights = [neg_weight, 1 - neg_weight]  # Weights for negative and positive classes

    # Compute weighted binary cross-entropy loss to handle class imbalance
    def weighted_loss(self, prediction, target):
        # Assign weights to each sample based on its target label (0 or 1)
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)  # Convert to PyTorch tensor
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()  # Move weights to GPU if enabled
        # Compute weighted binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
        return loss

    # Retrieve and preprocess a single volume by index
    def __getitem__(self, index):
        path = self.paths[index]  # Get the file path for this index
        # Load the volume from a binary file using pickle
        with open(path, 'rb') as file_handler:
            vol = pickle.load(file_handler).astype(np.int32)  # Convert to 32-bit integer array

        # Crop the volume to INPUT_DIM x INPUT_DIM (224x224) in the spatial dimensions
        pad = int((vol.shape[2] - INPUT_DIM) / 2)  # Calculate padding to center-crop
        vol = vol[:, pad:-pad, pad:-pad]  # Crop height and width, keep all slices (s)

        # Standardize pixel values to range [0, 255]
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL

        # Normalize using mean and standard deviation
        vol = (vol - MEAN) / STDDEV

        # Convert grayscale volume to RGB by stacking 3 identical channels
        vol = np.stack((vol,) * 3, axis=1)  # Shape becomes [s, 3, 224, 224]

        # Convert to PyTorch tensors
        vol_tensor = torch.FloatTensor(vol)  # Volume tensor: [s, 3, 224, 224]
        label_tensor = torch.FloatTensor([self.labels[index]])  # Label tensor: [1]

        return vol_tensor, label_tensor  # Return preprocessed volume and its label

    # Return the total number of volumes in the dataset
    def __len__(self):
        return len(self.paths)

# Function to create data loaders for training, validation, and testing
def load_data(diagnosis, use_gpu=False):
    # Define directories for train, validation, and test splits
    train_dirs = ['vol08', 'vol04', 'vol03', 'vol09', 'vol06', 'vol07']
    valid_dirs = ['vol10', 'vol05']
    test_dirs = ['vol01', 'vol02']

    # Create Dataset instances for each split
    train_dataset = Dataset(train_dirs, diagnosis, use_gpu)
    valid_dataset = Dataset(valid_dirs, diagnosis, use_gpu)
    test_dataset = Dataset(test_dirs, diagnosis, use_gpu)

    # Create DataLoaders to batch and load data efficiently
    train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=8, shuffle=True)
    # batch_size=1: One volume per batch; num_workers=8: 8 threads for loading; shuffle=True: Randomize training order
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=8, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False)

    return train_loader, valid_loader, test_loader  # Return the three loaders