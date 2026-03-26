#without resume and low augmentation, with logging and plotting

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import util

import sys
import re
from pathlib import Path

# Add project root so we can import the existing student model code
project_root = Path.cwd().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.models.student_model import create_student_model
from scripts.dense121.balancing_augmentation import AugmentedXRayDataset, prepare_balanced_training_dataframe


train_df = pd.read_csv(r"C:\Users\User\Sadeepa\X-Lite\scripts\dense121\split_csv\train_df.csv")
valid_df = pd.read_csv(r"C:\Users\User\Sadeepa\X-Lite\scripts\dense121\split_csv\val_df.csv")

test_df = pd.read_csv(r"C:\Users\User\Sadeepa\X-Lite\scripts\dense121\split_csv\test_df.csv")


labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation',
          'No_Finding']

def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=128, seed=1, target_w = 224, target_h = 224):
     
    print("getting train generator...") 
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator



def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=128, seed=1, target_w = 224, target_h = 224):
    
    print("getting train and valid generators...")
    
    # Change to match the training generator EXACTLY (samplewise, not featurewise)
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator


IMAGE_DIR = "C:\\Users\\User\\Sadeepa\\X-Lite\\data\\clahe_cache"
train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", labels)
valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels)


def compute_class_freqs(labels):
    
    # total number of patients (rows)
    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies

    return positive_frequencies, negative_frequencies

freq_pos, freq_neg = compute_class_freqs(train_generator.labels)


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
neg_data = pd.DataFrame([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)])
data = pd.concat([data, neg_data], ignore_index=True)

pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights

data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
neg_data = pd.DataFrame([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(neg_contribution)])
data = pd.concat([data, neg_data], ignore_index=True)

# PyTorch equivalent of the weighted Keras loss defined in Cell 16
class PyTorchWeightedLoss(nn.Module):
    def __init__(self, pos_weights, neg_weights, epsilon=1e-7):
        super().__init__()
        self.pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
        self.neg_weights = torch.tensor(neg_weights, dtype=torch.float32)
        self.epsilon = epsilon
        
    def forward(self, logits, labels):
        pos_w = self.pos_weights.to(logits.device)
        neg_w = self.neg_weights.to(logits.device)
        
        # Model returns unnormalized logits, so we apply sigmoid first to get probabilities
        probs = torch.sigmoid(logits)
        
        loss_pos = -1 * pos_w * labels * torch.log(probs + self.epsilon)
        loss_neg = -1 * neg_w * (1 - labels) * torch.log(1 - probs + self.epsilon)
        
        # Mean across the batch for each class, then sum all class losses together
        return torch.mean(loss_pos + loss_neg, dim=0).sum()


# Build EfficientNet-B0 + Performer student model (PyTorch, CUDA only)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU is required for this experiment, but no GPU was detected.')

torch.backends.cudnn.benchmark = True
device = torch.device('cuda')
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Keep class count aligned with the notebook's labels list (14 classes)
model = create_student_model(
    architecture='efficientnet_b0_performer',
    num_classes=len(labels),
    pretrained=True
).to(device)

# Resume from interrupted run checkpoint (epoch 14) and continue full-model training.
resume_checkpoint = Path("checkpoints") / "model_epoch_14_0.7462.pth"
resume_from_epoch = 0
resume_best_auc = 0.0
if resume_checkpoint.exists():
    checkpoint_state = torch.load(resume_checkpoint, map_location=device)
    model.load_state_dict(checkpoint_state, strict=True)

    match = re.search(r"model_epoch_(\d+)_([0-9.]+)\.pth", resume_checkpoint.name)
    if match:
        resume_from_epoch = int(match.group(1))
        try:
            resume_best_auc = float(match.group(2))
        except ValueError:
            resume_best_auc = 0.0

    print(f"Resumed model weights from: {resume_checkpoint}")
    print(f"Continuing from epoch {resume_from_epoch + 1}, current best val AUC: {resume_best_auc:.4f}")
else:
    print(f"Resume checkpoint not found, training from initialization: {resume_checkpoint}")

# Freeze ImageNet-pretrained backbone and train only attention + head
# for param in model.backbone.parameters():
#     param.requires_grad = False
# for param in model.attention.parameters():
#     param.requires_grad = True
# for param in model.head.parameters():
#     param.requires_grad = True

criterion = PyTorchWeightedLoss(pos_weights, neg_weights)
#crate a plot of how data looks after applying weightd loss to the /plots directory
plt.figure(figsize=(12, 6))
sns.barplot(x="Class", y="Value", hue="Label", data=data)
plt.title("Class Contributions to Weighted Loss")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"plots/class_contributions_weighted_loss.png")

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
# trainable_parameters = list(model.attention.parameters()) + list(model.head.parameters())
# optimizer = optim.AdamW(trainable_parameters, lr=1e-4, weight_decay=1e-5)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")


# OPTIONAL: lightweight PyTorch training loop
import os
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 1. Create a native PyTorch Dataset
# class XRayDataset(Dataset):
#     def __init__(self, df, image_dir, image_col, label_cols, transform=None):
#         self.df = df
#         self.image_dir = image_dir
#         self.image_col = image_col
#         self.label_cols = label_cols
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         img_path = os.path.join(self.image_dir, row[self.image_col])
#         image = Image.open(img_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         labels = torch.tensor(row[self.label_cols].values.astype(np.float32), dtype=torch.float32)
#         return image, labels


class BaseXRayDataset(Dataset):
    def __init__(self, df, image_dir, image_col, label_cols, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.image_col = image_col
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row[self.image_col])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[self.label_cols].values.astype(np.float32), dtype=torch.float32)
        return image, labels

def normalize_tensor(x):
    return (x - x.mean([1, 2], keepdim=True)) / (x.std([1, 2], keepdim=True) + 1e-7)

# 2. Replicate the Keras samplewise normalization in PyTorch
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(normalize_tensor)
])

if __name__ == '__main__':
    import logging
    from datetime import datetime

    # Setup directories for saving
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # Setup logging
    log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting training process...")

    # 3. Apply class balancing and augmentation policy
    undersample_targets = {
        'Infiltration': 10000,
        'No Finding': 10000,
    }
    
    oversample_targets = {
        'Emphysema': 10000,
        'Mass': 10000,
        'Pleural Thickening': 10000,
        'Pneumonia': 10000,
        'Pneumothorax': 10000,
        'Atelectasis': 10000,
        'Edema': 10000,
        'Effusion': 10000,
        'Hernia': 10000,
        'Cardiomegaly': 10000,
        'Fibrosis': 10000,
        'Nodule': 10000,
        'Consolidation': 10000,
    }

    balanced_train_df, balance_stats = prepare_balanced_training_dataframe(
        train_df=train_df,
        image_col='Image',
        undersample_targets=undersample_targets,
        oversample_targets=oversample_targets,
        random_state=42,
        oversample_improvement_ratio=0.01,
        min_improvement_samples=100,
    )
    logging.info(f"Balancing stats: {balance_stats}")

    # 4. Initialize PyTorch Datasets and DataLoaders
    train_dataset = AugmentedXRayDataset(
        balanced_train_df,
        IMAGE_DIR,
        'Image',
        labels,
        transform=transform,
        apply_gaussian_blur=True,
        gaussian_radius=1.0,
    )
    valid_dataset = BaseXRayDataset(valid_df, IMAGE_DIR, 'Image', labels, transform=transform)
    test_dataset = BaseXRayDataset(test_df, IMAGE_DIR, 'Image', labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    num_epochs = 50
    early_stop_patience = 5
    start_epoch = 0
    best_auc = 0.0
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        train_preds = []
        train_labels = []
        model.train()
        
        # Training Loop natively using the PyTorch dataloader
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False, dynamic_ncols=True):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            logits = model(x)
            loss = criterion(logits, y)
                
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            train_labels.append(y.detach().cpu().numpy())

        avg_train_loss = epoch_loss / len(train_loader)

        train_preds = np.concatenate(train_preds, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        try:
            train_auc = roc_auc_score(train_labels, train_preds, average='macro')
        except ValueError:
            train_auc = 0.0

        history['train_loss'].append(avg_train_loss)
        history['train_auc'].append(train_auc)
        
        # Validation Evaluation Loop
        model.eval()
        val_preds = []
        val_labels = []
        val_loss_total = 0.0
        
        with torch.no_grad():
            for x, y in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False, dynamic_ncols=True):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                
                logits = model(x)
                val_loss = criterion(logits, y)
                val_loss_total += val_loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                
                val_preds.append(probs)
                val_labels.append(y.detach().cpu().numpy())
                
        val_preds = np.concatenate(val_preds, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        avg_val_loss = val_loss_total / len(valid_loader)
        
        # Calculate Macro AUROC
        try:
            val_auc = roc_auc_score(val_labels, val_preds, average='macro')
        except ValueError:
            val_auc = 0.0
            
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)
        
        log_msg = (
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"train_loss: {avg_train_loss:.4f} - val_loss: {avg_val_loss:.4f} - "
            f"train_auc: {train_auc:.4f} - val_auc: {val_auc:.4f}"
        )
        logging.info(log_msg)

        # Save the model checking for each epoch
        ckpt_path = f"checkpoints/model_epoch_{epoch+1}_{val_auc:.4f}.pth"
        torch.save(model.state_dict(), ckpt_path)
        logging.info(f"Model saved to {ckpt_path}")
        
        # Early Stopping check on validation AUC
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            best_ckpt_path = f"checkpoints/best_model.pth"
            torch.save(model.state_dict(), best_ckpt_path)
            logging.info(f"New best model: {best_auc:.4f} (Saved to best_model.pth)")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in Val AUC for {epochs_no_improve} epochs.")
            if epochs_no_improve >= early_stop_patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Plotting Loss and AUC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    ax2.plot(history['train_auc'], label='Train AUC', color='green')
    ax2.plot(history['val_auc'], label='Val AUC', color='orange')
    ax2.set_ylabel('Macro AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Train and Validation AUC')
    ax2.legend()
    
    plot_path = f"plots/training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path)
    logging.info(f"Plots saved to {plot_path}")
    # plt.show() # Disabled to prevent the script from halting to wait for user interaction