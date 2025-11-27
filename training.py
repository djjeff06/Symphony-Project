import numpy as np
import random
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from model import SymphonyClassifier

def set_random_seeds():
    random_seed = 15
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler=None, device="cpu", n_epochs=20, task="composer_era"):
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, composer_labels, era_labels) in enumerate(train_loader):
            data, composer_labels, era_labels = data.to(device), composer_labels.to(device), era_labels.to(device)

            if task == "composer_era":
                composer_output, era_output = model.forward_composer_era(data)
            elif task == "composer":
                composer_output = model.forward_composer(data)
                era_output = None
            elif task == "era":
                composer_output = None
                era_output = model.forward_era(data)

            composer_loss = criterion(composer_output, composer_labels) if composer_output is not None else 0
            era_loss = criterion(era_output, era_labels) if era_output is not None else 0
            total_loss = composer_loss + era_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        val_loss = val(model, val_loader, criterion, device, task)
        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss}')

        if scheduler is not None:
            scheduler.step(val_loss)

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
                composer_output, era_output = model.forward_composer_era(data)
            elif task == "composer":
                composer_output = model.forward_composer(data)
                era_output = None
            elif task == "era":
                composer_output = None
                era_output = model.forward_era(data)

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

        if composer_output is not None:
            print(f'Validation Composer Accuracy: {100 * correct_composer / total_composer}%')
        if era_output is not None:
            print(f'Validation Era Accuracy: {100 * correct_era / total_era}%')

    return val_loss / len(val_loader)

def training():
    set_random_seeds()

    train_data = np.load(folder_path+"train.npz")
    X_train_full = train_data["X"]
    y_composer_train_full = train_data["y_composer"]
    y_era_train_full = train_data["y_era"]

    test_data = np.load(folder_path+"test.npz")
    X_test = test_data["X"]
    y_composer_test = test_data["y_composer"]
    y_era_test = test_data["y_era"]

    # split training to train and val sets
    X_train, X_val, y_composer_train, y_composer_val, y_era_train, y_era_val = train_test_split(
    X_train_full, 
    y_composer_train_full, 
    y_era_train_full, 
    test_size=0.20,
    random_state=15, 
    stratify=y_composer_train_full

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_composer_train_tensor = torch.from_numpy(y_composer_train).long()
    y_era_train_tensor = torch.from_numpy(y_era_train).long()

    X_val_tensor = torch.from_numpy(X_val).float()
    y_composer_val_tensor = torch.from_numpy(y_composer_val).long()
    y_era_val_tensor = torch.from_numpy(y_era_val).long()

    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_composer_test).long()
    y_era_test_tensor = torch.from_numpy(y_era_test).long()

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_composer_train_tensor, y_era_train_tensor)
    val_dataset   = TensorDataset(X_val_tensor, y_composer_val_tensor, y_era_val_tensor)
    test_dataset  = TensorDataset(X_test_tensor, y_test_tensor, y_era_test_tensor)

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SymphonyClassifier(
          input_size=X_train.shape[2],
          n_embedding=128,
          num_heads=8,
          num_layers=2,
          dropout=0.2
      ).to(device)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    n_epochs = 5

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, n_epochs, task="composer_era")

if __name__ == "__main__":
    if not (len(sys.argv) == 3 or len(sys.argv) == 4):
        raise Exception('Include the data path (containing train and val folders) and model_name as argument, e.g., python training.py ML/dataset best_model.')
    data_folder = sys.argv[1]
    model_name = sys.argv[2]
    training(data_folder, model_name)
