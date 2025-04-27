import pandas as pd

# List to store data from all subjects
all_subjects_data = []

# Load data for subjects 101 to 109
for subject_id in range(101, 110):
    file_path = f'D:\Activity Recognition\Activity_Detection\pamap2+physical+activity+monitoring\PAMAP2_Dataset\Protocol\subject{subject_id}.dat'
    try:
        # Read the data
        data = pd.read_csv(file_path, sep=' ', header=None, na_values='NaN')
        
        # Filter for hand accelerometer (columns 4-6) and activity ID (column 1)
        subject_data = data[[1, 4, 5, 6]].dropna()
        subject_data.columns = ['activity_id', 'x', 'y', 'z']
        
        # Add subject ID column
        subject_data['subject_id'] = subject_id
        
        all_subjects_data.append(subject_data)
    except FileNotFoundError:
        print(f"Warning: Could not find file for subject {subject_id}")
        continue

# Combine all subjects' data
combined_data = pd.concat(all_subjects_data, ignore_index=True)

# Get 50 samples each for walking and running across all subjects
walking = combined_data[combined_data['activity_id'] == 4].sample(50, random_state=42)
running = combined_data[combined_data['activity_id'] == 5].sample(50, random_state=42)
unlabeled = combined_data[combined_data['activity_id'].isin([4, 5])].sample(2, random_state=42)

# Print labeled examples
print("Labeled Examples:")
print("\nWalking Samples:")
for i, row in walking.iterrows():
    print(f"Subject {int(row['subject_id'])} - Walking: [{row['x']:.2f}, {row['y']:.2f}, {row['z']:.2f}]")

print("\nRunning Samples:")
for i, row in running.iterrows():
    print(f"Subject {int(row['subject_id'])} - Running: [{row['x']:.2f}, {row['y']:.2f}, {row['z']:.2f}]")

# Print unlabeled samples
print("\nUnlabeled Samples:")
for i, row in unlabeled.iterrows():
    print(f"Sample {i+1} (Subject {int(row['subject_id'])}): [{row['x']:.2f}, {row['y']:.2f}, {row['z']:.2f}]")