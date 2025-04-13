import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            # data is a list of 3 tensors [axial, coronal, sagittal]
            data = [d.to(device) for d in data]  # Move each view to device
            target = target.to(device)
            
            # Forward pass
            output = model(data)  # Single logit
            pred_prob = torch.sigmoid(output).cpu().numpy()  # Probability for AUC
            pred_label = (pred_prob >= 0.5).astype(int)  # Binary prediction
            
            # Compute loss using dataset's weighted loss
            loss = data_loader.dataset.weighted_loss(output, target)
            total_loss += loss.item()
            
            # Track predictions and labels
            all_preds.append(pred_prob.item())
            all_labels.append(target.cpu().numpy().item())
            
            # Accuracy
            correct += (pred_label == target.cpu().numpy()).sum()
            total += 1
    
    # Compute metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else float('nan')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc': auc
    }

# Example usage (assuming you have the dataset and model set up)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data (using the adapted load_data3 from previous response)
    data_dir = "path/to/data"
    labels_csv = "path/to/labels.csv"
    train_loader, valid_loader = load_data3(device, data_dir, labels_csv)
    
    # Initialize model
    model = MRNet3().to(device)
    # Load pretrained weights if available (e.g., model.load_state_dict(torch.load("model.pth")))
    
    # Evaluate on validation set
    metrics = evaluate(model, valid_loader, device)
    print(f"Validation Loss: {metrics['loss']:.4f}")
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation AUC: {metrics['auc']:.4f}")

if __name__ == "__main__":
    main()