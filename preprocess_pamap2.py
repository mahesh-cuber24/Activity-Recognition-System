import pandas as pd
import json
import os

# Full activity mapping from datasheet
activity_map = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "Nordic walking",
    9: "watching TV", 10: "computer work", 11: "car driving",
    12: "ascending stairs", 13: "descending stairs",
    16: "vacuum cleaning", 17: "ironing", 18: "folding laundry",
    19: "house cleaning", 20: "playing soccer", 24: "rope jumping"
}

def process_file(file_path):
    # Load all 54 columns
    data = pd.read_csv(file_path, sep=" ", header=None, na_values="NaN")
    data.columns = [f"col_{i}" for i in range(54)]
    # Filter out transient activities (ID 0) and NaN activity IDs
    data = data[data["col_1"].isin(activity_map.keys())].dropna(subset=["col_1"])
    return data

# Dataset path
data_dir = r"D:\Activity Recognition\Activity_Detection\pamap2+physical+activity+monitoring\PAMAP2_Dataset\Protocol"
processed_data = []
for file in os.listdir(data_dir):
    if file.endswith(".dat"):
        file_path = os.path.join(data_dir, file)
        df = process_file(file_path)
        for _, row in df.iterrows():
            # Hand IMU: ±16g acc (4-6), gyro (8-10)
            prompt = (
                f"Hand Accelerometer X: {row['col_4']:.2f}, Y: {row['col_5']:.2f}, Z: {row['col_6']:.2f}; "
                f"Hand Gyroscope X: {row['col_8']:.2f}, Y: {row['col_9']:.2f}, Z: {row['col_10']:.2f} -> Activity:"
            )
            completion = activity_map[int(row["col_1"])]
            processed_data.append({"text": f"{prompt} {completion}"})

# Subsample to 20k for 8 GB VRAM
processed_data = processed_data[:20000]

# Save to JSONL in project folder
with open("pamap2_processed.jsonl", "w") as f:
    for entry in processed_data:
        f.write(json.dumps(entry) + "\n")
print(f"Processed {len(processed_data)} samples.")