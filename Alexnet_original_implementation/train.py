import argparse  # For parsing command-line arguments
import json      # For saving arguments to a JSON file
import numpy as np  # For numerical operations (e.g., random seed)
import os        # For file system operations (e.g., creating directories)
import torch     # PyTorch library for deep learning
from datetime import datetime  # For tracking training time
from pathlib import Path  # For handling file paths in a platform-independent way
from sklearn import metrics  # For computing metrics like AUC (not directly used here but imported)

# Import custom modules
from evaluate import run_model  # Function to run the model (training or evaluation)
from loader import load_data    # Function to load the dataset
from model import MRNet         # The MRNet model class

# Define the training function
def train(rundir, diagnosis, epochs, learning_rate, use_gpu):

    # Load the train, validation, and test datasets into DataLoaders
    train_loader, valid_loader, test_loader = load_data(diagnosis, use_gpu)
    # Each loader provides batches of [1, s, 3, 224, 224] volumes and [1, 1] labels
    
    # Initialize the MRNet model
    model = MRNet()  # Creates an instance with pretrained AlexNet and custom classifier
    
    # Move the model to GPU if use_gpu is enabled
    if use_gpu:
        model = model.cuda()  # Transfers model parameters to GPU memory

    # Set up the Adam optimizer for training
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=.01)
    # model.parameters(): All trainable parameters (AlexNet convolutions + classifier)
    # learning_rate: Step size for weight updates
    # weight_decay=.01: L2 regularization to prevent overfittng
    
    # Define a learning rate scheduler to adjust LR based on validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.3, threshold=1e-4)
    # patience=5: Wait 5 epochs without improvement before reducing LR
    # factor=.3: Reduce LR by 30% when triggered
    # threshold=1e-4: Minimum improvement required to reset patience
    
    best_val_loss = float('inf')  # Track the lowest validation loss to save the best model

    start_time = datetime.now()  # Record the start time to track training duration

    # Training loop over the specified number of epochs
    for epoch in range(epochs):
        # Calculate and display the time elapsed since training started
        change = datetime.now() - start_time
        print('starting epoch {}. time passed: {}'.format(epoch + 1, str(change)))
        
        # Train the model for one epoch
        train_loss, train_auc, _, _ = run_model(model, train_loader, train=True, optimizer=optimizer)
        # run_model: Processes all batches in train_loader, updates weights, and returns metrics
        # train=True: Enables training mode (e.g., dropout active)
        # optimizer: Used to update model parameters during backpropagation
        print(f'train loss: {train_loss:0.4f}')  # Print average training loss
        print(f'train AUC: {train_auc:0.4f}')    # Print training AUC (area under ROC curve)

        # Evaluate the model on the validation set (no training)
        val_loss, val_auc, _, _ = run_model(model, valid_loader)
        # train=False (default): Evaluation mode (e.g., dropout disabled)
        print(f'valid loss: {val_loss:0.4f}')  # Print average validation loss
        print(f'valid AUC: {val_auc:0.4f}')    # Print validation AUC

        # Update the learning rate based on validation loss
        scheduler.step(val_loss)
        # If val_loss doesn’t improve for 5 epochs, LR is reduced by 30%

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Update the best validation loss
            # Create a filename with performance metrics and epoch number
            file_name = f'val{val_loss:0.4f}_train{train_loss:0.4f}_epoch{epoch + 1}'
            save_path = Path(rundir) / file_name  # Define the save path in the run directory
            torch.save(model.state_dict(), save_path)  # Save the model’s weights

# Define a function to parse command-line arguments
def get_parser():
    parser = argparse.ArgumentParser()  # Create an argument parser object
    parser.add_argument('--rundir', type=str, required=True)  # Directory to save run outputs
    parser.add_argument('--diagnosis', type=int, required=True)  # Threshold for binary labels
    parser.add_argument('--seed', default=42, type=int)  # Random seed for reproducibility
    parser.add_argument('--gpu', action='store_true')  # Flag to enable GPU usage
    parser.add_argument('--learning_rate', default=1e-05, type=float)  # Initial learning rate
    parser.add_argument('--weight_decay', default=0.01, type=float)  # L2 regularization strength
    parser.add_argument('--epochs', default=50, type=int)  # Number of training epochs
    parser.add_argument('--max_patience', default=5, type=int)  # Patience for LR scheduler
    parser.add_argument('--factor', default=0.3, type=float)  # Factor for LR reduction
    return parser

# Main execution block
if __name__ == '__main__':
    args = get_parser().parse_args()  # Parse command-line arguments into args object
    
    # Set random seeds for reproducibility across runs
    np.random.seed(args.seed)  # Seed for NumPy operations
    torch.manual_seed(args.seed)  # Seed for PyTorch CPU operations
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)  # Seed for PyTorch GPU operations

    # Create the run directory if it doesn’t exist
    os.makedirs(args.rundir, exist_ok=True)
    # exist_ok=True: Avoids error if directory already exists
    
    # Save the command-line arguments to a JSON file for record-keeping
    with open(Path(args.rundir) / 'args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)  # Convert args to dict and write with formatting

    # Start the training process with the provided arguments
    train(args.rundir, args.diagnosis, args.epochs, args.learning_rate, args.gpu)