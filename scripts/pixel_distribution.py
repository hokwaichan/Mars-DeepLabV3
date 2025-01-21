import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image

# Path to the directory containing label masks
LABELS_DIR = ""

# List of class names corresponding to class IDs
CLASS_NAMES = [
    "hole", "trace", "rover", "rock", "bedrock",
    "sand", "soil", "ridge", "sky", "NULL"  # Update this list if needed
]

# Total number of classes
NUM_CLASSES = len(CLASS_NAMES)

# Initialize an array to store pixel counts for each class
class_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

# Iterate through all label files in the directory
for label_file in os.listdir(LABELS_DIR):
    # Construct the full path to the label file
    label_path = os.path.join(LABELS_DIR, label_file)
    
    # Load the label mask as a NumPy array
    label_mask = np.array(Image.open(label_path))
    
    # Count the pixels for each class ID in the label mask
    for class_id in range(NUM_CLASSES):
        class_pixel_counts[class_id] += np.sum(label_mask == class_id)

# Print the pixel counts for each class
print("Pixel counts per class:")
for class_id, pixel_count in enumerate(class_pixel_counts):
    print(f"{CLASS_NAMES[class_id]}: {pixel_count}")

# Plot the class distribution as a bar chart
plt.figure(figsize=(10, 6))  # Set figure size
plt.bar(CLASS_NAMES, class_pixel_counts, color="skyblue")  # Bar chart with class names
plt.xlabel("Classes")  # Label for the x-axis
plt.ylabel("Pixel Count")  # Label for the y-axis
plt.title("Class Distribution in Dataset")  # Chart title
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()  # Display the chart
