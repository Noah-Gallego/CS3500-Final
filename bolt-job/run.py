# Ensemble version: trains both TabTransformer and MLP and averages outputs
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
import turibolt as bolt
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter

# ---------------- Device ----------------
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

# ---------------- Focal Loss ----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# ---------------- Models ----------------
class TabTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, hidden_dim=128):
        super(TabTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*2, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.head(x)

class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.selu = nn.SELU()
        self.dropout = nn.AlphaDropout(0.1)

    def forward(self, x):
        x = self.dropout(self.selu(self.fc1(x)))
        x = self.dropout(self.selu(self.fc2(x)))
        return self.fc3(x)

# ---------------- Main ----------------
def main():
    learning_rate = 0.001
    epochs = 500
    batch_size = 1024
    patience = 15

    df = pd.read_csv("Clean_Combined.csv")
    df.drop(columns=["mo_codes"], inplace=True)
    df = df[df["arrest_type"].isin([0, 1])].reset_index(drop=True)

    test_size = int(0.2 * len(df))
    train_df = df[:-test_size].reset_index(drop=True)
    test_df = df[-test_size:].reset_index(drop=True)

    # Split features and labels before scaling
    X_train = train_df.drop(columns="arrest_type")
    y_train = train_df["arrest_type"]
    X_test = test_df.drop(columns="arrest_type")
    y_test = test_df["arrest_type"]

    # Scale training and test data
    scaler = PowerTransformer(method='yeo-johnson')
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE OVersampling (Only On Training Data So performance is not affected)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Wrap datasets
    train_dataset = CrimeDataset(X_train_resampled, y_train_resampled)
    test_dataset = CrimeDataset(X_test_scaled, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Before SMOTE:", Counter(y_train))
    print("After SMOTE:", Counter(y_train_resampled))

    input_dim = X_train.shape[1]
    trans_model = TabTransformer(input_dim).to(device)
    mlp_model = MLPModel(input_dim).to(device)

    models = [trans_model, mlp_model]
    optimizers = [torch.optim.Adam(m.parameters(), lr=learning_rate, weight_decay=1e-4) for m in models]
    criterion = FocalLoss(alpha=0.8, gamma=1.0)
    schedulers = [ReduceLROnPlateau(opt, mode='min', patience=10) for opt in optimizers]

    for i, model in enumerate(models):
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        best_loss, epochs_no_improve = float('inf'), 0
        for epoch in tqdm(range(epochs), desc=f"Training model {i+1}"):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizers[i].zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1).float())
                loss.backward()
                optimizers[i].step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            bolt.send_metrics(metrics={"loss": avg_loss}, context={"model": f"model_{i+1}", "epoch": epoch})
            schedulers[i].step(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
                torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            f"{bolt.ARTIFACT_DIR}/model_{i+1}.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

    trans_model.load_state_dict(torch.load(f"{bolt.ARTIFACT_DIR}/model_1.pt"))
    mlp_model.load_state_dict(torch.load(f"{bolt.ARTIFACT_DIR}/model_2.pt"))
    trans_model.eval().to(device)
    mlp_model.eval().to(device)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            t_out = torch.sigmoid(trans_model(X_batch))
            m_out = torch.sigmoid(mlp_model(X_batch))
            probs = ((t_out + m_out) / 2).squeeze(1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    
    # Flip labels and probs to treat class 0 as the "positive" class
    flipped_labels = 1 - all_labels
    flipped_probs = 1 - all_probs

    precision, recall, thresholds = precision_recall_curve(flipped_labels, flipped_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-6)

    # Best threshold for class 0 F1
    best_idx = np.argmax(f1_scores)
    best_thresh = 1 - thresholds[best_idx]  # flip back

    print(f"📍 Best threshold to maximize F1 for class 0: {best_thresh:.4f}")

    preds = (np.array(all_probs) > best_thresh).astype(int)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    plt.figure(figsize=(8, 5))
    sns.histplot(all_probs[all_labels == 0], color='red', bins=50, kde=True, stat='density', label='Class 0')
    sns.histplot(all_probs[all_labels == 1], color='blue', bins=50, kde=True, stat='density', label='Class 1')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold = 0.5')
    plt.title("Predicted Probability Distribution by Class")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{bolt.ARTIFACT_DIR}/probability_histogram.png")
    plt.close()

    report = classification_report(all_labels, preds, digits=4, output_dict=True)
    pd.DataFrame(report).to_csv(f"{bolt.ARTIFACT_DIR}/ensemble_classification_report.csv")

    cm = confusion_matrix(all_labels, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Ensemble Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{bolt.ARTIFACT_DIR}/ensemble_confusion_matrix.png")
    plt.close()

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(f"{bolt.ARTIFACT_DIR}/ensemble_roc_curve.png")
    plt.close()

    bolt.send_metrics(
        metrics={
            "best_threshold": float(best_thresh),
            "f1_macro": float(report["macro avg"]["f1-score"]),
            "recall_class_0": float(report["0"]["recall"]),
            "recall_class_1": float(report["1"]["recall"]),
            "roc_auc": float(roc_auc)
        },
        context={"stage": "final_eval"}
    )

    print(f"\n📀 Best Threshold by F1: {best_thresh:.3f}")
    print("\n📊 Classification Report:\n")
    print(classification_report(all_labels, preds, digits=4))

    print("\n🔍 Sample Predictions")
    for _ in range(5):
        i = random.randint(0, len(preds)-1)
        print(f"Predicted: {preds[i]} | Actual: {all_labels[i]} | Prob: {all_probs[i]:.4f}")

if __name__ == "__main__":
    main()