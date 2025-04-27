import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# SimCLR model definition
class SimCLRModel(nn.Module):
    def __init__(self, input_channels=9, output_dim=64):
        super(SimCLRModel, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, time, channels] -> [batch, channels, time]
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.projection(h)
        return h, z

# TFC model definition
class TFCModel(nn.Module):
    def __init__(self, input_channels=9, output_dim=64):
        super(TFCModel, self).__init__()

        # Time encoder
        self.time_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Frequency encoder
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Projections
        self.time_projection = nn.Linear(64, output_dim)
        self.freq_projection = nn.Linear(64, output_dim)

    def forward(self, x):
        # Time domain
        x_time = x.permute(0, 2, 1)  # [batch, time, channels] -> [batch, channels, time]
        h_time = self.time_encoder(x_time)
        h_time = h_time.view(h_time.size(0), -1)
        z_time = self.time_projection(h_time)

        # Frequency domain - fixed to handle each channel separately
        batch_size, seq_len, channels = x.shape
        x_freq_list = []

        for c in range(channels):
            # Process each channel separately
            channel_data = x[:, :, c]
            # Compute FFT for this channel
            channel_fft = torch.fft.rfft(channel_data).abs()
            x_freq_list.append(channel_fft)

        # Stack along channel dimension
        x_freq = torch.stack(x_freq_list, dim=1)  # [batch, channels, fft_size]

        h_freq = self.freq_encoder(x_freq)
        h_freq = h_freq.view(h_freq.size(0), -1)
        z_freq = self.freq_projection(h_freq)

        return h_time, h_freq, z_time, z_freq

# Dataset class
class PAMAP2Dataset(Dataset):
    def __init__(self, data_path, subject_ids=None):
        self.data = []
        self.labels = []

        # Load data (simplified)
        if subject_ids is None:
            subject_ids = range(1, 10)

        for subject_id in subject_ids:
            try:
                file_path = os.path.join(data_path, f"subject10{subject_id}.dat")

                # Check if file exists
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    continue

                print(f"Loading file: {file_path}")
                data = pd.read_csv(file_path, sep=' ', header=None)
                print(f"File loaded with shape: {data.shape}")

                # Extract accelerometer data and labels
                acc_cols = list(range(4, 7)) + list(range(21, 24)) + list(range(38, 41))

                # Check if columns exist
                if max(acc_cols) >= data.shape[1]:
                    print(f"File {file_path} doesn't have enough columns. Expected at least {max(acc_cols)+1}, got {data.shape[1]}")
                    continue

                X = data.iloc[:, acc_cols].values
                y = data.iloc[:, 1].values

                # Remove NaN values
                mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
                X = X[mask]
                y = y[mask]

                print(f"After removing NaNs: {X.shape[0]} samples")

                if X.shape[0] == 0:
                    print(f"No valid data in {file_path} after removing NaNs")
                    continue

                # Create windows
                window_size = 128
                stride = 64
                windows_count = 0

                for i in range(0, len(X) - window_size + 1, stride):
                    window = X[i:i + window_size]
                    label = int(y[i + window_size // 2])
                    if label > 0:  # Skip label 0 (no activity)
                        self.data.append(window)
                        self.labels.append(label)
                        windows_count += 1

                print(f"Created {windows_count} windows from subject {subject_id}")

            except Exception as e:
                print(f"Error loading subject {subject_id}: {str(e)}")

        # Check if we have any data
        if len(self.data) == 0:
            raise ValueError("No valid data was loaded. Check your data path and file format.")

        self.data = np.array(self.data, dtype=np.float32)  # Explicitly use float32
        self.labels = np.array(self.labels, dtype=np.int64)

        # Normalize data safely
        mean = np.nanmean(self.data, axis=0)
        std = np.nanstd(self.data, axis=0)
        # Replace NaNs with 0 in mean and 1 in std
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=1.0)
        # Avoid division by zero
        std[std == 0] = 1.0

        self.data = (self.data - mean) / std

        print(f"Dataset created with {len(self.data)} samples and {len(np.unique(self.labels))} unique activities")
        print(f"Data shape: {self.data.shape}")  # Print data shape for debugging

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Load the dataset
data_path = r"D:\Activity Recognition\Activity_Detection\pamap2+physical+activity+monitoring\PAMAP2_Dataset\Protocol"
dataset = PAMAP2Dataset(data_path)

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Load the trained models
simclr_model = SimCLRModel().to(device)
simclr_model.load_state_dict(torch.load("simclr_model.pth"))

tfc_model = TFCModel().to(device)
tfc_model.load_state_dict(torch.load("tfc_model.pth"))

def generate_embeddings(model, data_loader, model_type="SimCLR"):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device).float()

            if model_type == "SimCLR":
                embeddings, _ = model(data)
            else:  # TFC
                time_embeddings, freq_embeddings, _, _ = model(data)
                embeddings = torch.cat([time_embeddings, freq_embeddings], dim=1)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)

    print(f"Generated {len(all_labels)} embeddings using {model_type} model")
    return all_embeddings, all_labels

# Generate embeddings for both models
simclr_embeddings, labels = generate_embeddings(simclr_model, test_loader, "SimCLR")
tfc_embeddings, _ = generate_embeddings(tfc_model, test_loader, "TFC")

# Visualize embeddings with t-SNE
tsne = TSNE(n_components=2, random_state=42)
simclr_2d = tsne.fit_transform(simclr_embeddings)

plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    idx = labels == label
    plt.scatter(simclr_2d[idx, 0], simclr_2d[idx, 1], label=f"Activity {label}")

plt.legend()
plt.title("SimCLR Embeddings Visualization")
plt.savefig("simclr_embeddings.png")
plt.close()

# Simulate LLM annotation using embeddings
def simulate_llm_annotation(embeddings, labels):
    """
    Simulate LLM annotation based on embedding similarity
    """
    # Create a dictionary to store embeddings by class
    class_embeddings = {}
    for label in np.unique(labels):
        class_embeddings[label] = embeddings[labels == label]

    # Calculate class centroids
    class_centroids = {label: np.mean(embs, axis=0) for label, embs in class_embeddings.items()}

    # Function to predict class based on closest centroid
    def predict_class(embedding):
        distances = {label: np.linalg.norm(embedding - centroid)
                    for label, centroid in class_centroids.items()}
        return min(distances, key=distances.get)

    # Make predictions
    predictions = np.array([predict_class(emb) for emb in embeddings])

    # Calculate accuracy
    accuracy = np.mean(predictions == labels)

    # Calculate F1 score
    f1 = f1_score(labels, predictions, average='weighted')

    # Create confusion matrix
    cm = confusion_matrix(labels, predictions)

    print(f"LLM Annotation Simulation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('LLM Annotation Confusion Matrix')
    plt.savefig("llm_annotation_confusion_matrix.png")
    plt.close()

    return accuracy, f1, predictions

# Simulate LLM annotation using SimCLR embeddings
simclr_acc, simclr_f1, simclr_preds = simulate_llm_annotation(simclr_embeddings, labels)

# Simulate LLM annotation using TFC embeddings
tfc_acc, tfc_f1, tfc_preds = simulate_llm_annotation(tfc_embeddings, labels)

# Compare results
print("\nComparison of Models for LLM Annotation:")
print(f"SimCLR: Accuracy = {simclr_acc:.4f}, F1 Score = {simclr_f1:.4f}")
print(f"TFC: Accuracy = {tfc_acc:.4f}, F1 Score = {tfc_f1:.4f}")

# Prepare for integration with a real LLM
def prepare_for_llm(embeddings, activity_labels, activity_names=None):
    """
    Convert embeddings to a format that can be used by an LLM
    """
    if activity_names is None:
        # Default activity names
        activity_names = {
            1: "lying",
            2: "sitting",
            3: "standing",
            4: "walking",
            5: "running",
            6: "cycling",
            7: "Nordic walking",
            9: "watching TV",
            10: "computer work",
            11: "car driving",
            12: "ascending stairs",
            13: "descending stairs",
            16: "vacuum cleaning",
            17: "ironing",
            18: "folding laundry",
            19: "house cleaning",
            20: "playing soccer",
            24: "rope jumping"
        }

    # Normalize embeddings to be between 0 and 1
    normalized_embeddings = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min())

    # Convert to a simpler representation (e.g., 3 decimal places)
    simplified_embeddings = np.round(normalized_embeddings, 3)

    # Create a dictionary mapping from embedding to activity name
    embedding_to_activity = {}
    for i, embedding in enumerate(simplified_embeddings):
        # Convert embedding to a tuple so it can be used as a dictionary key
        embedding_tuple = tuple(embedding)
        embedding_to_activity[embedding_tuple] = activity_names.get(activity_labels[i], f"Unknown activity {activity_labels[i]}")

    print(f"Prepared {len(embedding_to_activity)} embeddings for LLM integration")
    return embedding_to_activity

# Prepare embeddings for LLM integration
embedding_to_activity = prepare_for_llm(simclr_embeddings, labels)

# Save embeddings and labels
np.save('simclr_embeddings.npy', simclr_embeddings)
np.save('tfc_embeddings.npy', tfc_embeddings)
np.save('activity_labels.npy', labels)

# Save the embedding to activity mapping
import pickle
with open('embedding_to_activity.pkl', 'wb') as f:
    pickle.dump(embedding_to_activity, f)

print("\nSaved all embeddings and mappings for future use")