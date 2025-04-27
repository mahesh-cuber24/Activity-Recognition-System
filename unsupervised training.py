import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from torch.nn import functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simple dataset class with better error handling
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

# Simple augmentation function
def augment_time_series(x):
    # Add jitter
    return x + np.random.normal(loc=0, scale=0.1, size=x.shape)

# SimCLR model (simplified)
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

# TFC model (simplified and fixed)
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

        # Frequency encoder - modified to handle variable length FFT
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # This will handle variable length inputs
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

# Fixed contrastive loss function
def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Simplified and more robust NT-Xent loss implementation
    """
    batch_size = z1.size(0)
    
    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate embeddings from both augmentations
    z = torch.cat([z1, z2], dim=0)
    
    # Compute pairwise similarity
    sim = torch.mm(z, z.t()) / temperature
    
    # Create mask for positive pairs
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    
    # Create labels for positive pairs
    positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2*batch_size, 1)
    
    # Create mask to exclude self-similarity
    mask = torch.ones_like(sim)
    mask.fill_diagonal_(0)
    
    # Get negative samples using the mask
    negative_samples = sim[mask.bool()].reshape(2*batch_size, -1)
    
    # Concatenate positive and negative samples
    logits = torch.cat([positive_samples, negative_samples], dim=1)
    
    # Create labels (positive samples are the target)
    labels = torch.zeros(2*batch_size, dtype=torch.long, device=z1.device)
    
    # Compute cross entropy loss
    loss = F.cross_entropy(logits, labels)
    
    return loss

# Training function for SimCLR
def train_unsupervised_simclr(model, train_loader, optimizer, epochs=10):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device).float()  # Explicitly convert to float32
            
            # Skip small batches that might cause issues
            if data.size(0) <= 1:
                print(f"Skipping batch {batch_idx} with size {data.size(0)}")
                continue
                
            batch_count += 1

            # Create augmented views
            batch_size = data.shape[0]
            augmented_data1 = torch.tensor(np.array([augment_time_series(data[i].cpu().numpy())
                                                    for i in range(batch_size)]), device=device).float()
            augmented_data2 = torch.tensor(np.array([augment_time_series(data[i].cpu().numpy())
                                                    for i in range(batch_size)]), device=device).float()

            # Forward pass
            _, z1 = model(augmented_data1)
            _, z2 = model(augmented_data2)

            # Compute loss
            loss = nt_xent_loss(z1, z2)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        if batch_count > 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/batch_count:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}], No valid batches processed")

    return model

# Training function for TFC
def train_unsupervised_tfc(model, train_loader, optimizer, epochs=10):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device).float()  # Explicitly convert to float32
            
            # Skip small batches that might cause issues
            if data.size(0) <= 1:
                print(f"Skipping batch {batch_idx} with size {data.size(0)}")
                continue
                
            batch_count += 1

            # Create augmented views
            batch_size = data.shape[0]
            augmented_data1 = torch.tensor(np.array([augment_time_series(data[i].cpu().numpy())
                                                    for i in range(batch_size)]), device=device).float()
            augmented_data2 = torch.tensor(np.array([augment_time_series(data[i].cpu().numpy())
                                                    for i in range(batch_size)]), device=device).float()

            try:
                # Forward pass
                h_time1, h_freq1, z_time1, z_freq1 = model(augmented_data1)
                h_time2, h_freq2, z_time2, z_freq2 = model(augmented_data2)

                # Compute losses
                time_loss = nt_xent_loss(z_time1, z_time2)
                freq_loss = nt_xent_loss(z_freq1, z_freq2)
                cross_loss = nt_xent_loss(z_time1, z_freq1)

                # Total loss
                loss = 0.5 * (time_loss + freq_loss) + 0.5 * cross_loss

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                print(f"Data shape: {data.shape}")
                continue

        if batch_count > 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/batch_count:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}], No valid batches processed")

    return model

def main():
    # Data path - FIXED PATH STRING
    data_path = r"D:\Activity Recognition\Activity_Detection\pamap2+physical+activity+monitoring\PAMAP2_Dataset\Protocol"  # Using raw string

    print(f"Looking for data in: {os.path.abspath(data_path)}")

    # Check if directory exists
    if not os.path.exists(data_path):
        print(f"ERROR: Data directory not found: {data_path}")
        print("Available directories in current location:")
        print(os.listdir('.'))
        return

    # List files in directory
    print("Files in data directory:")
    for file in os.listdir(data_path):
        print(f"  - {file}")

    try:
        # Create dataset
        dataset = PAMAP2Dataset(data_path)

        # Check dataset size
        if len(dataset) == 0:
            print("Dataset is empty. Cannot proceed with training.")
            return

        # Split dataset
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        # Create data loaders with smaller batch size
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

        # Initialize models
        simclr_model = SimCLRModel().to(device)
        tfc_model = TFCModel().to(device)

        # Initialize optimizers
        simclr_optimizer = optim.Adam(simclr_model.parameters(), lr=0.001)
        tfc_optimizer = optim.Adam(tfc_model.parameters(), lr=0.001)

        # Train models
        print("Training SimCLR model...")
        train_unsupervised_simclr(simclr_model, train_loader, simclr_optimizer, epochs=5)

        print("Training TFC model...")
        train_unsupervised_tfc(tfc_model, train_loader, tfc_optimizer, epochs=5)

        # Save models
        torch.save(simclr_model.state_dict(), "simclr_model.pth")
        torch.save(tfc_model.state_dict(), "tfc_model.pth")

        print("Training completed!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()