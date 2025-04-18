# MRNet-Inspired CNNs for ACL and Meniscus Tear Detection
(This repository is still a work in progress)

## Overview

This repository contains implementations of convolutional neural networks (CNNs) for detecting anterior cruciate ligament (ACL) and meniscus tears from MRI scans, inspired by the MRNet model introduced by Bien et al. (2018). Our project re-implements the original MRNet architecture with modifications and extends it by developing two additional models based on ResNet and EfficientNet. We aim to compare the performance of these architectures (AlexNet, ResNet, EfficientNet) and explore ensemble methods to enhance diagnostic accuracy.

The dataset used is sourced from the original MRNet study, which includes MRI scans with axial, coronal, and sagittal views for each sample. Our goal is to evaluate whether modern CNN architectures can improve upon the baseline MRNet model’s accuracy and robustness in detecting knee abnormalities.

> **Citation**: This work is based on the MRNet model described in:
> Bien, N., Rajpurkar, P., Ball, R. L., Irvin, J., Park, A., Jones, M., ... & Langlotz, C. P. (2018). Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet. *PLOS Medicine*, 15(11), e1002699. [https://doi.org/10.1371/journal.pmed.1002699](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699)

## Project Abstract

The advancement of deep learning has significantly impacted numerous fields, with particularly notable progress in medical imaging. Improved neural network architectures have enabled the development of deeper models capable of capturing increasingly complex relationships. These advancements have facilitated the automation of both routine and challenging diagnostic tasks, sometimes surpassing the capabilities of even experienced medical professionals.

This study aims to investigate the effectiveness of convolutional neural networks (CNNs) in detecting anterior cruciate ligament (ACL) and meniscus tears from MRI data. The dataset we are using stems from a Stanford research paper that proposed an AlexNet-based architecture for this task (MRNet) and includes multiple slices from three different points of view (axial, sagittal, and coronal) for each unique sample.

Our approach involves re-implementing this model — with minor modifications and hyperparameter tuning — as a performance baseline. We then evaluate whether modern CNN architectures can offer improved accuracy and robustness. Finally we combine our three CNN architectures (AlexNet, ResNet, EfficientNet) with an ensemble model to produce a final output.

## Dataset

The dataset is sourced from the MRNet study by Bien et al. (2018) and consists of MRI scans for knee injury detection. Each sample includes:
- **Multiple slices** from three views: axial, coronal, and sagittal.
- **Labels** indicating the presence of ACL or meniscus tears (binary classification).

The dataset is stored as `.npy` files, with accompanying CSV files for labels (e.g., `train-abnormal.csv`). The training set contains 904 samples, and the validation set contains 226 samples, split with stratification to maintain class balance.

## Model Architectures

We have implemented three CNN models, all inspired by the MRNet architecture but with distinct backbones:

1. **MRNet-AlexNet**:
   - **Backbone**: AlexNet, pretrained on ImageNet.
   - **Description**: This is a re-implementation of the original MRNet model with modifications (see below).
   - **Purpose**: Serves as the baseline for performance comparison.

2. **MRNet-ResNet**:
   - **Backbone**: ResNet (e.g., ResNet-18 or ResNet-50, pretrained on ImageNet).
   - **Description**: Replaces AlexNet with a deeper ResNet architecture to leverage residual connections for improved feature learning.
   - **Purpose**: Tests whether a more modern, deeper architecture enhances accuracy and robustness.

3. **MRNet-EfficientNet**:
   - **Backbone**: EfficientNet (e.g., EfficientNet-B0, pretrained on ImageNet).
   - **Description**: Uses EfficientNet’s compound scaling to balance depth, width, and resolution.
   - **Purpose**: Evaluates a highly efficient and scalable architecture for medical imaging tasks.

### Modifications to the Original MRNet Model

Our implementations retain the core idea of MRNet—processing multiple MRI views (axial, coronal, sagittal) and aggregating features—but include the following key changes:
- **Simultaneous View Processing**: Unlike the original MRNet, which processes each view independently and combines predictions later, our models process all three views (axial, coronal, sagittal) in a single forward pass within a unified architecture. This allows the model to learn cross-view relationships earlier, potentially improving feature integration.
- **Unified Model Architecture**: Instead of training separate models for each view, we use a single model with three parallel backbones (one per view), concatenating features before classification. This reduces training complexity and enables joint optimization.
- **Hyperparameter Tuning**: We adjusted learning rates (default: `1e-05`), weight decay (`0.01`), and dropout rates (0.3 per view, 0.5 in classifier) to optimize performance on our dataset.
- **Weighted Loss**: Implemented a weighted binary cross-entropy loss to handle class imbalance, with weights computed dynamically based on label prevalence (e.g., `[0.8075, 0.1925]` for abnormal labels).
- **Batch Size and Data Loading**: Used a batch size of 4 with a custom `collate_fn` to handle variable slice counts across views, ensuring compatibility with differing numbers of slices per sample.
- **Pretrained Weights**: All models use pretrained ImageNet weights, but we fine-tune the classifier layers (`768→256→1`) for the MRI task.

These modifications aim to enhance the model’s ability to generalize and handle the specific challenges of MRI data, such as variable slice counts and class imbalance.

## Planned Experiments

We are actively working on the following:
- **Performance Comparison**: Evaluate the accuracy, AUC, and robustness of MRNet-AlexNet, MRNet-ResNet, and MRNet-EfficientNet on the validation and test sets. Metrics include:
  - Area Under the ROC Curve (AUC)
  - Binary cross-entropy loss
  - Sensitivity and specificity for ACL and meniscus tear detection
- **Ensembling**: Combine predictions from the three models (e.g., via majority voting or weighted averaging) to explore whether an ensemble improves diagnostic performance over individual models.
- **Hyperparameter Optimization**: Further tune learning rates, dropout rates, and batch sizes to maximize performance for each architecture.
- **Analysis of View Contributions**: Investigate the importance of axial, coronal, and sagittal views by ablating individual views or weighting their features.

Results will be logged in the repository, including detailed metrics and visualizations (e.g., ROC curves, loss plots).

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
5. Download the MRNet dataset (refer to the original paper or Stanford's dataset page for access instructions) and place it in the data/ directory.

## Training a Model

Example command to train the Alexnet based model:
```bash
cd Alexnet
python train.py --rundir /path/to/runs/experiment-name \
    --diagnosis 0 \
    --data_dir /path/to/train \
    --labels_csv /path/to/train/train-abnormal.csv \
    --mps

```
### Command Details

- **`--rundir`**: Specifies the directory where checkpoints and logs are saved (e.g., `/path/to/runs/experiment-name`).
- **`--diagnosis`**: Select the classification task:
  - `0`: Abnormal detection
  - `1`: ACL tear detection
  - `2`: Meniscus tear detection
- **`--mps`**: Use Metal Performance Shaders (MPS) for macOS GPUs. Replace with `--gpu` for CUDA or omit for CPU.
- **Outputs**:
  - Training arguments are saved in `[rundir]/args.json`.
  - Metrics (loss and AUC) are printed per epoch and logged in `[rundir]/metrics.txt`.
  - Model checkpoints are saved as `[rundir]/valX.XXXX_trainY.YYYY_epochZ.pth`.




















