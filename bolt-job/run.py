import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import turibolt as bolt
from tqdm import tqdm
import random

# Move Training To CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Dataset ----------------
class CrimeDataset(Dataset):
    def __init__(self, features, targets):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
# --------------- Params ----------------
def get_parameters():
    if bolt.get_current_task_id() is None:
        return {
            'learning_rate': 0.001,
            'epochs': 100,
        }
    else:
        config = bolt.get_current_config()['parameters']
        return {
            'learning_rate': float(config['learning_rate']),
            'epochs': int(config['epochs']),
        }

# ---------------- Model ----------------
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1d = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.pool(self.relu(self.conv1d(x))).squeeze(-1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # Return raw logits

# ---------------- Main ----------------
def main():
    params = get_parameters()
    learning_rate = params['learning_rate']

    df = pd.read_csv("Clean_Combined.csv")
    df.drop(columns=["mo_codes"], inplace=True)

    # Split
    test_size = int(0.2 * len(df))
    train_df = df[:-test_size].reset_index(drop=True)
    test_df = df[-test_size:].reset_index(drop=True)

    # Features and targets
    X_train, y_train = train_df.drop(columns="Status"), train_df["Status"]
    X_test, y_test = test_df.drop(columns="Status"), test_df["Status"]

    # Scale
    scaler = PowerTransformer(method='yeo-johnson')
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = CrimeDataset(X_train_scaled, y_train)
    test_dataset = CrimeDataset(X_test_scaled, y_test)

    # Weighted sampler
    class_counts = y_train.value_counts().to_dict()
    weights = {k: 1.0 / v for k, v in class_counts.items()}
    sample_weights = y_train.map(weights)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model, loss, optimizer
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # Use BCEWithLogitsLoss with pos_weight
    neg_weight = class_counts[0]
    pos_weight = class_counts[1]
    pos_weight_tensor = torch.tensor([neg_weight / pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Early Stopping Parameters
    max_epochs = params['epochs']
    patience = 15
    epochs_no_improve = 0
    best_loss = float('inf')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

    # Training loop
    for epoch in tqdm(range(max_epochs)):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        bolt.send_metrics(
            metrics={'loss': avg_loss},
            context={'epoch': epoch}
        )
        scheduler.step(avg_loss)

        # Check For Early Stopping
        if loss < best_loss:
            best_loss = loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered at epoch #", epoch)
                break

    # Save model
    torch.save(model.state_dict(), f"{bolt.ARTIFACT_DIR}/model_final.pt")
    print("✅ Model saved as model_final.pt")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad(): 
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = torch.sigmoid(model(X_batch))  # Apply sigmoid for evaluation
            preds = (outputs > 0.5).long().squeeze(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    print("\n📊 Classification Report:\n")
    print(classification_report(all_labels, all_preds, digits=4))

    print("\nSample Predictions")
    for _ in range(5):
        rand_index = random.randint(0, len(all_preds) - 1)
        print("Prediction: ", all_preds[rand_index], "\tActual: ", all_labels[rand_index])

if __name__ == "__main__":
    main()