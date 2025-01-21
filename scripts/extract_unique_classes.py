import os
import numpy as np
from PIL import Image

# Path to the folder containing label images
labels_folder = ''

# Set to store all unique class values from label images
unique_classes = set()

# Loop through each label image in the Labels folder
for label_file in os.listdir(labels_folder):
    label_path = os.path.join(labels_folder, label_file)
    
    # Check if the file is a valid image file (jpg/png)
    if label_file.endswith(('.jpg', '.png')):
        # Open the label image
        label_image = Image.open(label_path)
        
        # Convert the image to a numpy array for easy processing
        label_array = np.array(label_image)
        
        # Add the unique pixel values (class IDs) to the set
        unique_classes.update(np.unique(label_array))

# Convert the set of unique classes to a sorted list
unique_classes = sorted(list(unique_classes))

# Print out the unique classes detected in the dataset
print("The unique classes are:", unique_classes)
