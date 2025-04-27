import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: CUDA not available, running on CPU. This will be slower.")

# Custom Dataset for PyTorch
class PAMAP2Dataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data[['x', 'y', 'z']].values, dtype=torch.float32)
        self.labels = torch.tensor(data['activity_id'].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Data Augmentation Functions
def jitter(data, sigma=0.01):
    return data + torch.randn_like(data) * sigma

def time_warp(data, factor=1.2):
    return data * factor

# SimCLR Model with 1D Convolutions
class SimCLR(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=32):
        super(SimCLR, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, x):
        # Add a dummy sequence dimension
        x = x.unsqueeze(2)  # Shape: [batch_size, input_dim, 1]
        return self.encoder(x)

# TFC Model with 1D Convolutions
class TFC(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=64):
        super(TFC, self).__init__()
        self.time_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, x):
        # Add a dummy sequence dimension
        x = x.unsqueeze(2)  # Shape: [batch_size, input_dim, 1]
        return self.time_encoder(x)

# SimCLR Contrastive Loss with Dynamic Temperature
def nt_xent_loss(out_1, out_2, temperature=1.0):
    out_1 = nn.functional.normalize(out_1, dim=1)
    out_2 = nn.functional.normalize(out_2, dim=1)
    batch_size = out_1.size(0)
    out = torch.cat([out_1, out_2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.T) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = -torch.log(pos_sim / sim_matrix.sum(dim=-1))
    return loss.mean()

# Simulate LLM annotation with metrics
def annotate_with_embeddings(test_emb, walking_emb, running_emb, true_labels):
    predictions = []
    for i, emb in enumerate(test_emb):
        min_walk_dist = min([euclidean(emb, w) for w in walking_emb])
        min_run_dist = min([euclidean(emb, r) for r in running_emb])
        pred = 4 if min_walk_dist < min_run_dist else 5
        predictions.append(pred)
        print(f"Sample {i+1}: Embedding {emb}, Walking Dist: {min_walk_dist:.2f}, Running Dist: {min_run_dist:.2f}, Predicted: {'Walking' if pred == 4 else 'Running'}, True: {'Walking' if true_labels[i] == 4 else 'Running'}")

    accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)
    f1 = f1_score(true_labels, predictions, pos_label=5)
    cm = confusion_matrix(true_labels, predictions, labels=[4, 5])

    return accuracy, f1, cm, predictions

# Load and preprocess data
all_subjects_data = []
for subject_id in range(101, 110):
    file_path = r'D:\Activity Recognition\Activity_Detection\pamap2+physical+activity+monitoring\PAMAP2_Dataset\Protocol\subject{subject_id}.dat'
    try:
        data = pd.read_csv(file_path.format(subject_id=subject_id), sep=' ', header=None, na_values='NaN')
        subject_data = data[[1, 4, 5, 6]].dropna()
        subject_data.columns = ['activity_id', 'x', 'y', 'z']
        subject_data['subject_id'] = subject_id
        all_subjects_data.append(subject_data)
    except FileNotFoundError:
        print(f"Warning: Could not find file for subject {subject_id}")
        continue

combined_data = pd.concat(all_subjects_data, ignore_index=True)
combined_data = combined_data[combined_data['activity_id'].isin([4, 5])]

# Use a subset for faster execution and reset index
combined_data = combined_data.sample(frac=0.1, random_state=42).reset_index(drop=True)

# Standardize the data
scaler = StandardScaler()
combined_data[['x', 'y', 'z']] = scaler.fit_transform(combined_data[['x', 'y', 'z']])

# Prepare dataset and dataloader
dataset = PAMAP2Dataset(combined_data)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Train SimCLR with 200 epochs and dynamic temperature
simclr_model = SimCLR().to(device)
optimizer = optim.SGD(simclr_model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(200):
    simclr_model.train()
    total_loss = 0
    temperature = max(0.5, 1.0 - epoch * 0.005)  # Dynamic temperature
    for data, _ in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        aug_data_1 = jitter(data, sigma=0.05)
        aug_data_2 = time_warp(data, factor=1.1)
        out_1 = simclr_model(aug_data_1)
        out_2 = simclr_model(aug_data_2)
        loss = nt_xent_loss(out_1, out_2, temperature=temperature)
        if torch.isnan(loss):
            print(f"NaN loss at epoch {epoch+1}, skipping batch")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(simclr_model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"SimCLR Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# Train TFC with 100 epochs
tfc_model = TFC().to(device)
optimizer = optim.Adam(tfc_model.parameters(), lr=3e-4, betas=(0.9, 0.99))

for epoch in range(100):
    tfc_model.train()
    total_loss = 0
    for data, _ in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        aug_data_1 = jitter(data, sigma=0.05)
        aug_data_2 = time_warp(data, factor=1.1)
        out_1 = tfc_model(aug_data_1)
        out_2 = tfc_model(aug_data_2)
        loss = nt_xent_loss(out_1, out_2, temperature=1.0)
        if torch.isnan(loss):
            print(f"NaN loss at epoch {epoch+1}, skipping batch")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tfc_model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    print(f"TFC Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# Generate embeddings
simclr_model.eval()
tfc_model.eval()
with torch.no_grad():
    all_data = torch.tensor(combined_data[['x', 'y', 'z']].values, dtype=torch.float32).to(device)
    simclr_embeddings = simclr_model(all_data).cpu().numpy()
    tfc_embeddings = tfc_model(all_data).cpu().numpy()

# Reduce dimensionality to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42)
simclr_embeddings_2d = tsne.fit_transform(simclr_embeddings)
tfc_embeddings_2d = tsne.fit_transform(tfc_embeddings)

# Visualize embeddings
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(simclr_embeddings_2d[:, 0], simclr_embeddings_2d[:, 1], c=combined_data['activity_id'], cmap='viridis', s=5)
plt.title("SimCLR Embeddings (t-SNE)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(tfc_embeddings_2d[:, 0], tfc_embeddings_2d[:, 1], c=combined_data['activity_id'], cmap='viridis', s=5)
plt.title("TFC Embeddings (t-SNE)")
plt.colorbar()

plt.show()

# Prepare samples for evaluation
walking_samples = combined_data[combined_data['activity_id'] == 4].sample(25, random_state=42)
running_samples = combined_data[combined_data['activity_id'] == 5].sample(25, random_state=42)
test_samples = combined_data.sample(10, random_state=42)

# Get corresponding embeddings using integer positions
walking_simclr_emb = simclr_embeddings_2d[walking_samples.index]
running_simclr_emb = simclr_embeddings_2d[running_samples.index]
test_simclr_emb = simclr_embeddings_2d[test_samples.index]

walking_tfc_emb = tfc_embeddings_2d[walking_samples.index]
running_tfc_emb = tfc_embeddings_2d[running_samples.index]
test_tfc_emb = tfc_embeddings_2d[test_samples.index]

# Evaluate SimCLR
print("\nSimCLR Annotations:")
simclr_accuracy, simclr_f1, simclr_cm, simclr_preds = annotate_with_embeddings(
    test_simclr_emb, walking_simclr_emb, running_simclr_emb, test_samples['activity_id'].values
)
print(f"SimCLR Accuracy: {simclr_accuracy:.2f}")
print(f"SimCLR F1-Score: {simclr_f1:.2f}")
print("SimCLR Confusion Matrix:")
print(simclr_cm)
disp = ConfusionMatrixDisplay(confusion_matrix=simclr_cm, display_labels=['Walking', 'Running'])
disp.plot(cmap=plt.cm.Blues)
plt.title("SimCLR Confusion Matrix")
plt.show()

# Evaluate TFC
print("\nTFC Annotations:")
tfc_accuracy, tfc_f1, tfc_cm, tfc_preds = annotate_with_embeddings(
    test_tfc_emb, walking_tfc_emb, running_tfc_emb, test_samples['activity_id'].values
)
print(f"TFC Accuracy: {tfc_accuracy:.2f}")
print(f"TFC F1-Score: {tfc_f1:.2f}")
print("TFC Confusion Matrix:")
print(tfc_cm)
disp = ConfusionMatrixDisplay(confusion_matrix=tfc_cm, display_labels=['Walking', 'Running'])
disp.plot(cmap=plt.cm.Blues)
plt.title("TFC Confusion Matrix")
plt.show()