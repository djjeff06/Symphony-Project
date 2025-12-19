import numpy as np
import sys
import os
import random
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from model import SymphonyClassifier
import wandb
from model_configs import CONFIG

def set_random_seeds(random_seed = 15):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler=None, device="cpu", n_epochs=20, model_name="best_model", task="composer_era"):
    best_val_acc = 0
    alpha = 1.5
    beta  = 1.0
    current_accuracy = 0
    
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, composer_labels, era_labels) in enumerate(train_loader):
            data, composer_labels, era_labels = data.to(device), composer_labels.to(device), era_labels.to(device)

            if task == "composer_era":
                composer_output, era_output = model.forward_composer_era(data, device=device)
            elif task == "composer":
                composer_output = model.forward_composer(data, device=device)
                era_output = None
            elif task == "era":
                composer_output = None
                era_output = model.forward_era(data, device=device)

            composer_loss = criterion(composer_output, composer_labels) if composer_output is not None else torch.tensor(0.0, device=device)
            era_loss = criterion(era_output, era_labels) if era_output is not None else torch.tensor(0.0, device=device)

            if task == "composer_era":
                train_loss = alpha* composer_loss + beta* era_loss
            else:
                train_loss = composer_loss + era_loss

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()

        val_loss, val_acc_composer, val_acc_era = val(model, val_loader, criterion, device, task)
        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {running_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Composer Acc: {val_acc_composer if val_acc_composer is not None else "N/A"}, '
              f'Val Era Acc: {val_acc_era if val_acc_era is not None else "N/A"}')

        log_dict = {
            "epoch": epoch + 1,
            "train_loss": running_loss/len(train_loader),
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        if val_acc_era is not None:
            log_dict["val_era_acc"] = val_acc_era
            current_accuracy = val_acc_era
        if val_acc_composer is not None:
            log_dict["val_composer_acc"] = val_acc_composer
            current_accuracy = val_acc_composer

        wandb.log(log_dict)

        # Save best model
        if current_accuracy > best_val_acc:
            best_val_acc = current_accuracy
            torch.save(model.state_dict(), model_name+".pth")
            if task == "composer_era":
                print(f"Saved best model at epoch {epoch+1} with val_composer_acc: {val_acc_composer:.4f} and val_era_acc: {val_acc_era:.4f}")
            elif task == "composer":
                print(f"Saved best model at epoch {epoch+1} with val_composer_acc: {val_acc_composer:.4f}")
            else:
                print(f"Saved best model at epoch {epoch+1} with val_era_acc: {val_acc_era:.4f}")

        if scheduler is not None:
            scheduler.step(val_loss)

    wandb.finish()

def val(model, val_loader, criterion, device="cpu", task="composer_era"):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct_composer = 0
        correct_era = 0
        total_composer = 0
        total_era = 0

        for data, composer_labels, era_labels in val_loader:
            data, composer_labels, era_labels = data.to(device), composer_labels.to(device), era_labels.to(device)

            if task == "composer_era":
                composer_output, era_output = model.forward_composer_era(data, device=device)
            elif task == "composer":
                composer_output = model.forward_composer(data, device=device)
                era_output = None
            elif task == "era":
                composer_output = None
                era_output = model.forward_era(data, device=device)

            composer_loss = criterion(composer_output, composer_labels) if composer_output is not None else torch.tensor(0.0, device=device)
            era_loss = criterion(era_output, era_labels) if era_output is not None else torch.tensor(0.0, device=device)
            total_loss = composer_loss + era_loss

            val_loss += total_loss.item()

            # Compute accuracy
            if composer_output is not None:
                _, predicted_composer = torch.max(composer_output, 1)
                correct_composer += (predicted_composer == composer_labels).sum().item()
                total_composer += composer_labels.size(0)

            if era_output is not None:
                _, predicted_era = torch.max(era_output, 1)
                correct_era += (predicted_era == era_labels).sum().item()
                total_era += era_labels.size(0)

        # Compute percentages
        val_acc_composer = 100 * correct_composer / total_composer if total_composer > 0 else None
        val_acc_era = 100 * correct_era / total_era if total_era > 0 else None

    return val_loss / len(val_loader), val_acc_composer, val_acc_era

'''
class AugmentedDataset(Dataset):
    """
    X: A feature tensor of shape (N, T, D)
    y_composer, y_era: Label tensors of shape (N,)
    When augment=True, data augmentation is applied only to the training samples.
    """
    def __init__(self, X, y_composer, y_era, augment=False,
                 noise_std=0.02, time_mask_max_ratio=0.1):
        self.X = X
        self.y_composer = y_composer
        self.y_era = y_era
        self.augment = augment
        self.noise_std = noise_std
        self.time_mask_max_ratio = time_mask_max_ratio

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].clone()              # (T, D)
        y_comp = self.y_composer[idx]
        y_era  = self.y_era[idx]

        if self.augment:
            x = self._augment(x)

        return x, y_comp, y_era

    def _augment(self, x):
        """
        x: torch.Tensor, shape (T, D)
        two simple feature-space augmentations are applied:
          1. Gaussian noise injection
          2. Time-masking (SpecAugment-style)
        """

        # ① Gaussian Noise
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # ② Time Masking
        if torch.rand(1) < 0.5:
            T = x.size(0)
            max_len = max(1, int(T * self.time_mask_max_ratio))  # Ex) 10% of the total sequence length
            mask_len = torch.randint(1, max_len + 1, (1,)).item()
            start = torch.randint(0, T - mask_len + 1, (1,)).item()
            x[start:start+mask_len, :] = 0.0

        return x
    '''

def main(folder_path, model_name, mode):
    wandb.init(
        project="Symphony Project",
        config=CONFIG
    )

    config = wandb.config

    set_random_seeds()

    train_data = np.load(os.path.join(folder_path, "train.npz"))
    X_train_full = train_data["X"]
    y_composer_train_full = train_data["y_composer"]
    y_era_train_full = train_data["y_era"]

    # split training to train and val sets
    X_train, X_val, y_composer_train, y_composer_val, y_era_train, y_era_val = train_test_split(
        X_train_full, 
        y_composer_train_full, 
        y_era_train_full, 
        test_size=0.20,
        random_state=15, 
        stratify=y_composer_train_full)

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_composer_train_tensor = torch.from_numpy(y_composer_train).long()
    y_era_train_tensor = torch.from_numpy(y_era_train).long()

    X_val_tensor = torch.from_numpy(X_val).float()
    y_composer_val_tensor = torch.from_numpy(y_composer_val).long()
    y_era_val_tensor = torch.from_numpy(y_era_val).long()

    # Create Datasets
    # Only the training dataset uses augment=True
    '''
    train_dataset = AugmentedDataset(
        X_train_tensor, y_composer_train_tensor, y_era_train_tensor,
        augment=True
    )
    val_dataset = AugmentedDataset(
        X_val_tensor, y_composer_val_tensor, y_era_val_tensor,
        augment=False
    )
    '''

    train_dataset = TensorDataset(X_train_tensor, y_composer_train_tensor, y_era_train_tensor)
    val_dataset   = TensorDataset(X_val_tensor, y_composer_val_tensor, y_era_val_tensor)

    # Create DataLoaders
    batch_size = config.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SymphonyClassifier(
            input_size=X_train.shape[2],
            n_embedding=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout
        ).to(device)
    
    wandb.watch(model, log="all")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    '''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
    '''
    n_epochs = config.epochs

    train(model, train_loader, val_loader, criterion, optimizer, None, device, n_epochs, model_name, task=mode)

if __name__ == "__main__":
    if not (len(sys.argv) == 4 or len(sys.argv) == 5):
        raise Exception('Include the data path, model_name, and mode as argument, e.g., python training.py ML/dataset best_model composer_era.')
    data_folder = sys.argv[1]
    model_name = sys.argv[2]
    mode = sys.argv[3]
    main(data_folder, model_name, mode)
