import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
from ollama import Client
import json
from sklearn.cluster import KMeans

# Path to saved files
save_path = r"D:\VS Code\ACT RECOG NEW\saved_files"
print(f"Loading files from: {os.path.abspath(save_path)}")

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

# Load the saved embeddings, mapping, and original data
try:
    simclr_embeddings = np.load(os.path.join(save_path, 'simclr_embeddings.npy'))
    labels = np.load(os.path.join(save_path, 'activity_labels.npy'))
    original_data = np.load(os.path.join(save_path, 'original_sensor_data.npy'))

    # Load normalization parameters
    mean = np.load(os.path.join(save_path, 'normalization_mean.npy'))
    std = np.load(os.path.join(save_path, 'normalization_std.npy'))

    with open(os.path.join(save_path, 'embedding_to_activity.pkl'), 'rb') as f:
        embedding_to_activity = pickle.load(f)

    print(f"Loaded {len(simclr_embeddings)} embeddings and {len(np.unique(labels))} unique activities")
except Exception as e:
    print(f"Error loading saved files: {e}")
    print("If you haven't run the updated save_embeddings_with_data.py script, please run it first.")
    exit(1)

# Load the trained model
try:
    simclr_model = SimCLRModel().to(device)
    simclr_model.load_state_dict(torch.load(os.path.join(save_path, "simclr_model.pth")))
    simclr_model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Perform clustering on embeddings to create unlabeled activity groups
print("\nPerforming clustering to identify activity patterns without labels...")
n_clusters = 12  # Approximate number of activities
# Convert embeddings to float64 for KMeans
simclr_embeddings_64 = simclr_embeddings.astype(np.float64)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(simclr_embeddings_64)

# Count samples in each cluster
cluster_counts = {}
for i in range(n_clusters):
    cluster_counts[i] = np.sum(cluster_labels == i)
print(f"Created {n_clusters} activity clusters:")
for cluster_id, count in cluster_counts.items():
    print(f"  Cluster {cluster_id}: {count} samples")

# Create a mapping from cluster ID to a temporary activity name
cluster_names = {i: f"Activity Pattern {i+1}" for i in range(n_clusters)}

# Optional: Use LLM to name clusters (simplified version)
def name_cluster_with_llm(cluster_id):
    try:
        client = Client(host='http://localhost:11434')
        model_name = "llama3.2:latest"

        # Get sample indices from this cluster
        sample_indices = np.where(cluster_labels == cluster_id)[0][:3]  # Take up to 3 samples

        # If we have ground truth labels (for comparison only, not used in clustering)
        true_activities = [labels[idx] for idx in sample_indices]
        activity_counts = {}
        for act in true_activities:
            if act in activity_counts:
                activity_counts[act] += 1
            else:
                activity_counts[act] = 1

        # For debugging/evaluation only
        most_common = max(activity_counts.items(), key=lambda x: x[1])[0]
        activity_names = {
            1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
            6: "cycling", 7: "Nordic walking", 9: "watching TV", 10: "computer work",
            11: "car driving", 12: "ascending stairs", 13: "descending stairs",
            16: "vacuum cleaning", 17: "ironing", 18: "folding laundry",
            19: "house cleaning", 20: "playing soccer", 24: "rope jumping"
        }
        true_name = activity_names.get(most_common, f"Activity {most_common}")

        # This would be the actual LLM prompt in a fully unlabeled approach
        # We're simplifying here by just using the most common true label
        return f"Pattern {cluster_id+1} ({true_name})"
    except Exception as e:
        print(f"Error naming cluster: {e}")
        return f"Activity Pattern {cluster_id+1}"

# Name each cluster
print("\nNaming clusters based on patterns...")
for cluster_id in range(n_clusters):
    cluster_names[cluster_id] = name_cluster_with_llm(cluster_id)
    print(f"  Cluster {cluster_id} named as: {cluster_names[cluster_id]}")

# Function to find the closest embedding
def find_closest_embedding(new_embedding, reference_embeddings):
    distances = np.linalg.norm(reference_embeddings - new_embedding, axis=1)
    closest_idx = np.argmin(distances)
    return closest_idx, distances[closest_idx]

# Function to get activity prediction from LLM using Ollama
def get_llm_prediction(activity_name, confidence):
    try:
        client = Client(host='http://localhost:11434')

        # Use the exact model name you have
        model_name = "llama3.2:latest"

        print(f"Using model: {model_name}")

        prompt = f"""
        Based on sensor data from a wearable device, I've detected an activity pattern.
        The pattern is most similar to the activity: {activity_name}
        Confidence level: {confidence:.2f}

        Please describe this activity in detail, including:
        1. What movements are involved
        2. What energy level is required
        3. In what context this activity typically occurs

        Your response should be concise but informative.
        """

        print("Sending request to Ollama...")
        response = client.chat(model=model_name, messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ])

        return response['message']['content']
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return f"Error: Could not connect to Ollama. Make sure it's running. Error details: {str(e)}"

# Function to process new data (labeled approach)
def process_new_data(new_data, model, reference_embeddings):
    # Normalize the data using the saved mean and std
    normalized_data = (new_data - mean) / std

    # Generate embedding for new data
    try:
        with torch.no_grad():
            data_tensor = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0).to(device)
            embedding, _ = model(data_tensor)
            embedding_np = embedding.cpu().numpy()[0]

        print(f"Generated embedding with shape: {embedding_np.shape}")
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return {"error": str(e)}

    # Find closest matching embedding
    closest_idx, distance = find_closest_embedding(embedding_np, reference_embeddings)
    print(f"Found closest match at index {closest_idx} with distance {distance:.4f}")

    # Calculate confidence (inverse of distance, normalized)
    max_distance = 5.0  # Set a reasonable maximum distance
    confidence = max(0, 1 - (distance / max_distance))

    # Get activity name
    activity_name = f"Activity {labels[closest_idx]}"

    # Try to get a more descriptive name if available
    activity_id = labels[closest_idx]
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

    if activity_id in activity_names:
        activity_name = activity_names[activity_id]

    print(f"Identified as {activity_name} with confidence {confidence:.2f}")

    # Get LLM prediction
    llm_description = get_llm_prediction(activity_name, confidence)

    result = {
        'activity': activity_name,
        'confidence': float(confidence),
        'description': llm_description
    }

    print("\nResult:")
    print(json.dumps(result, indent=2))

    return result

# Function to process new data using the unlabeled approach
def process_new_data_unlabeled(new_data, model):
    # Normalize the data using the saved mean and std
    normalized_data = (new_data - mean) / std

    # Generate embedding for new data
    try:
        with torch.no_grad():
            data_tensor = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0).to(device)
            embedding, _ = model(data_tensor)
            embedding_np = embedding.cpu().numpy()[0]

        print(f"Generated embedding with shape: {embedding_np.shape}")
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return {"error": str(e)}

    # Convert embedding to float64 for KMeans
    embedding_np_64 = embedding_np.astype(np.float64)

    # Assign to cluster
    cluster_id = kmeans.predict([embedding_np_64])[0]
    activity_name = cluster_names[cluster_id]
    print(f"Assigned to cluster {cluster_id}: {activity_name}")

    # For evaluation only (not part of the unlabeled approach)
    # Find closest sample in the same cluster for confidence estimation
    cluster_samples = simclr_embeddings[cluster_labels == cluster_id]
    if len(cluster_samples) > 0:
        distances = np.linalg.norm(cluster_samples - embedding_np, axis=1)
        min_distance = np.min(distances)
        max_distance = 5.0
        confidence = max(0, 1 - (min_distance / max_distance))
    else:
        confidence = 0.5  # Default if cluster is empty

    print(f"Confidence based on cluster proximity: {confidence:.2f}")

    # Get LLM description
    llm_description = get_llm_prediction(activity_name, confidence)

    result = {
        'activity': activity_name,
        'cluster_id': int(cluster_id),
        'confidence': float(confidence),
        'description': llm_description
    }

    print("\nResult:")
    print(json.dumps(result, indent=2))

    return result

# Function to simulate real sensor data
def simulate_sensor_data(window_size=128, channels=9):
    """Generate simulated sensor data for testing"""
    return np.random.randn(window_size, channels).astype(np.float32)

# Function to get a real sample from the dataset
def get_real_sample(add_noise=True):
    """Get a real sample from the dataset for testing"""
    # Find a random index
    random_idx = np.random.randint(0, len(labels))
    activity_id = labels[random_idx]

    # Get the corresponding embedding and original data
    embedding = simclr_embeddings[random_idx]
    data = original_data[random_idx].copy()  # Use the actual original data

    # Add a small amount of noise to simulate real-world variations
    if add_noise:
        noise_level = 0.05  # 5% noise
        noise = np.random.randn(*data.shape) * noise_level * np.std(data)
        data = data + noise
        print(f"Added {noise_level*100}% noise to simulate real-world variations")

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

    activity_name = activity_names.get(activity_id, f"Activity {activity_id}")

    return data, activity_id, activity_name

# Interactive demo
def run_interactive_demo():
    print("\n=== Activity Recognition with LLM Demo ===\n")
    print("This demo will recognize activities using your trained model and Ollama.")
    print("Make sure Ollama is running with a Llama model loaded.")
    print("\nOptions:")
    print("1. Test with simulated data (labeled approach)")
    print("2. Test with a real sample from the dataset (labeled approach)")
    print("3. Test with simulated data (unlabeled approach)")
    print("4. Test with a real sample (unlabeled approach)")
    print("5. Exit")

    while True:
        choice = input("\nEnter your choice (1-5): ")

        if choice == '1':
            print("\nGenerating simulated sensor data...")
            data = simulate_sensor_data()
            print("Processing data using labeled approach...")
            result = process_new_data(data, simclr_model, simclr_embeddings)

            # Save the result to a file
            result_path = os.path.join(save_path, f"result_simulated_labeled_{int(time.time())}.json")
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {result_path}")

        elif choice == '2':
            print("\nGetting a real sample from the dataset...")
            noise_choice = input("Add realistic noise to the sample? (y/n): ").lower()
            add_noise = noise_choice.startswith('y')

            data, activity_id, activity_name = get_real_sample(add_noise=add_noise)
            print(f"Selected a sample with true activity: {activity_name} (ID: {activity_id})")
            print("Processing data using labeled approach...")
            result = process_new_data(data, simclr_model, simclr_embeddings)

            # Add the true activity to the result
            result['true_activity'] = activity_name
            result['true_activity_id'] = int(activity_id)

            # Save the result to a file
            result_path = os.path.join(save_path, f"result_real_labeled_{int(time.time())}.json")
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {result_path}")

        elif choice == '3':
            print("\nGenerating simulated sensor data...")
            data = simulate_sensor_data()
            print("Processing data using unlabeled approach...")
            result = process_new_data_unlabeled(data, simclr_model)

            # Save the result to a file
            result_path = os.path.join(save_path, f"result_simulated_unlabeled_{int(time.time())}.json")
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {result_path}")

        elif choice == '4':
            print("\nGetting a real sample from the dataset...")
            noise_choice = input("Add realistic noise to the sample? (y/n): ").lower()
            add_noise = noise_choice.startswith('y')

            data, activity_id, activity_name = get_real_sample(add_noise=add_noise)
            print(f"Selected a sample with true activity: {activity_name} (ID: {activity_id})")
            print("Processing data using unlabeled approach...")
            result = process_new_data_unlabeled(data, simclr_model)

            # Add the true activity to the result (for evaluation only)
            result['true_activity'] = activity_name
            result['true_activity_id'] = int(activity_id)

            # Save the result to a file
            result_path = os.path.join(save_path, f"result_real_unlabeled_{int(time.time())}.json")
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {result_path}")

        elif choice == '5':
            print("\nExiting demo. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1-5.")

# Run the interactive demo
if __name__ == "__main__":
    run_interactive_demo()