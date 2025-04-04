import argparse
import matplotlib.pyplot as plt  # Imported but not used in this script
import os
import numpy as np
import torch
from sklearn import metrics
from torch.autograd import Variable

# Import custom modules
from loader import load_data  # Function to load data (defined above)
from model import MRNet       # MRNet model class (defined above)

# Define command-line argument parser
def get_parser():
    parser = argparse.ArgumentParser()  # Create a parser object
    parser.add_argument('--model_path', type=str, required=True)  # Path to trained model file
    parser.add_argument('--split', type=str, required=True)       # Data split: 'train', 'valid', or 'test'
    parser.add_argument('--diagnosis', type=int, required=True)   # Diagnosis threshold for labeling
    parser.add_argument('--gpu', action='store_true')             # Flag to enable GPU usage
    return parser

# Run the model on a data loader, optionally training it
def run_model(model, loader, train=False, optimizer=None):
    preds = []    # List to store predictions
    labels = []   # List to store true labels

    # Set model to training or evaluation mode
    if train:
        model.train()  # Enable dropout, batch norm updates, etc.
    else:
        model.eval()   # Disable training-specific operations

    total_loss = 0.  # Accumulate loss over all batches
    num_batches = 0  # Count number of batches processed

    # Iterate over batches in the loader
    for batch in loader:
        if train:
            optimizer.zero_grad()  # Clear gradients before forward pass

        vol, label = batch  # Unpack batch: vol=[1, s, 3, 224, 224], label=[1, 1]
        if loader.dataset.use_gpu:
            vol = vol.cuda()    # Move volume to GPU if enabled
            label = label.cuda()  # Move label to GPU if enabled
        # vol = Variable(vol)     # Wrap in Variable for autograd (legacy, optional in modern PyTorch) --> check
        # label = Variable(label) # Wrap label in Variable --> check

        # Forward pass: Get model prediction
        logit = model.forward(vol)  # [1, 1] logit for the volume

        # Compute weighted loss using the dataset's method
        loss = loader.dataset.weighted_loss(logit, label)  # Scalar loss value
        total_loss += loss.item()  # Add to running total (item() extracts scalar)

        # Convert logit to probability using sigmoid
        pred = torch.sigmoid(logit)  # [1, 1] probability
        pred_npy = pred.data.cpu().numpy()[0][0]  # Extract scalar value to numpy
        label_npy = label.data.cpu().numpy()[0][0]  # Extract scalar label to numpy

        preds.append(pred_npy)   # Store prediction
        labels.append(label_npy) # Store true label

        # Backward pass and optimization (if training)
        if train:
            loss.backward()      # Compute gradients
            optimizer.step()     # Update model weights
        num_batches += 1         # Increment batch counter

    avg_loss = total_loss / num_batches  # Compute average loss over all batches

    # Compute ROC curve and AUC score
    fpr, tpr, threshold = metrics.roc_curve(labels, preds)  # False pos rate, true pos rate
    auc = metrics.auc(fpr, tpr)  # Area under the ROC curve

    return avg_loss, auc, preds, labels  # Return metrics and predictions

# Evaluate the model on a specified data split
def evaluate(split, model_path, diagnosis, use_gpu):
    # Load data for all splits
    train_loader, valid_loader, test_loader = load_data(diagnosis, use_gpu)

    # Initialize the MRNet model
    model = MRNet()  # Create a new instance of MRNet
    
    # Load the trained model weights
    state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
    model.load_state_dict(state_dict)  # Apply weights to the model

    # Move model to GPU if enabled
    if use_gpu:
        model = model.cuda()

    # Select the appropriate data loader based on split
    if split == 'train':
        loader = train_loader
    elif split == 'valid':
        loader = valid_loader
    elif split == 'test':
        loader = test_loader
    else:
        raise ValueError("split must be 'train', 'valid', or 'test'")

    # Run the model on the selected loader (evaluation mode)
    loss, auc, preds, labels = run_model(model, loader)

    # Print evaluation results
    print(f'{split} loss: {loss:0.4f}')  # Average loss for the split
    print(f'{split} AUC: {auc:0.4f}')    # AUC score for the split

    return preds, labels  # Return predictions and labels for further analysis

# Main execution block
if __name__ == '__main__':
    args = get_parser().parse_args()  # Parse command-line arguments
    # Run evaluation with provided arguments
    evaluate(args.split, args.model_path, args.diagnosis, args.gpu)