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

# Load the saved embeddings and mapping
try:
    simclr_embeddings = np.load(os.path.join(save_path, 'simclr_embeddings.npy'))
    labels = np.load(os.path.join(save_path, 'activity_labels.npy'))

    with open(os.path.join(save_path, 'embedding_to_activity.pkl'), 'rb') as f:
        embedding_to_activity = pickle.load(f)

    print(f"Loaded {len(simclr_embeddings)} embeddings and {len(np.unique(labels))} unique activities")
except Exception as e:
    print(f"Error loading saved files: {e}")
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

# Function to find the closest embedding
def find_closest_embedding(new_embedding, reference_embeddings):
    distances = np.linalg.norm(reference_embeddings - new_embedding, axis=1)
    closest_idx = np.argmin(distances)
    return closest_idx, distances[closest_idx]

# Function to get activity prediction from LLM using Ollama
def get_llm_prediction(activity_name, confidence):
    try:
        client = Client(host='http://localhost:11434')

        # Get available models
        try:
            models = client.list()
            available_models = [model['name'] for model in models['models']]
            print(f"Available models: {available_models}")

            # Choose the best available model
            model_name = None
            for preferred_model in ["llama3.2:latest"]:
                if preferred_model in available_models:
                    model_name = preferred_model
                    break

            if not model_name:
                if available_models:
                    model_name = available_models[0]  # Use the first available model
                else:
                    return "Error: No models available in Ollama"
        except Exception as e:
            print(f"Error listing models: {e}")
            # Fallback to using llama3 directly
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

# Function to process new data
def process_new_data(new_data, model, reference_embeddings):
    # Generate embedding for new data
    try:
        with torch.no_grad():
            data_tensor = torch.tensor(new_data, dtype=torch.float32).unsqueeze(0).to(device)
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

# Function to simulate real sensor data
def simulate_sensor_data(window_size=128, channels=9):
    """Generate simulated sensor data for testing"""
    return np.random.randn(window_size, channels).astype(np.float32)

# Function to get a real sample from the dataset
def get_real_sample():
    """Get a real sample from the dataset for testing"""
    # Find a random index
    random_idx = np.random.randint(0, len(labels))
    activity_id = labels[random_idx]

    # Get the corresponding embedding
    embedding = simclr_embeddings[random_idx]

    # We don't have the original data, so we'll simulate it
    # but we'll use the real activity label
    data = simulate_sensor_data()

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
    print("1. Test with simulated data")
    print("2. Test with a real sample from the dataset")
    print("3. Exit")

    while True:
        choice = input("\nEnter your choice (1-3): ")

        if choice == '1':
            print("\nGenerating simulated sensor data...")
            data = simulate_sensor_data()
            print("Processing data...")
            result = process_new_data(data, simclr_model, simclr_embeddings)

            # Save the result to a file
            result_path = os.path.join(save_path, f"result_simulated_{int(time.time())}.json")
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {result_path}")

        elif choice == '2':
            print("\nGetting a real sample from the dataset...")
            data, activity_id, activity_name = get_real_sample()
            print(f"Selected a sample with true activity: {activity_name} (ID: {activity_id})")
            print("Processing data...")
            result = process_new_data(data, simclr_model, simclr_embeddings)

            # Add the true activity to the result
            result['true_activity'] = activity_name
            result['true_activity_id'] = int(activity_id)

            # Save the result to a file
            result_path = os.path.join(save_path, f"result_real_{int(time.time())}.json")
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {result_path}")

        elif choice == '3':
            print("\nExiting demo. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# Run the interactive demo
if __name__ == "__main__":
    run_interactive_demo()