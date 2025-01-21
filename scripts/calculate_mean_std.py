import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Path to the image folder
image_folder = ''

# Transformation to convert images to tensors
transform = transforms.ToTensor()

# Collect all image file paths
image_files = [
    os.path.join(image_folder, fname) 
    for fname in os.listdir(image_folder) 
    if fname.lower().endswith(('png', 'jpg', 'jpeg'))
]

# Initialize variables for mean and standard deviation calculation
mean = 0.0  # Cumulative mean across all images
std = 0.0   # Cumulative standard deviation across all images
nb_samples = 0  # Total number of samples

# Loop through all images in the folder
for img_path in image_files:
    # Open the image and ensure it is in RGB format
    img = Image.open(img_path).convert("RGB")
    
    # Apply the transformation (convert to tensor)
    img_tensor = transform(img)  # Resulting shape: [C, H, W] with values in [0, 1]
    
    # Calculate the per-channel mean and standard deviation
    mean += img_tensor.mean(dim=(1, 2))  # Mean for each channel
    std += img_tensor.std(dim=(1, 2))   # Standard deviation for each channel
    
    # Increment the sample count
    nb_samples += 1

# Compute the average mean and standard deviation across all samples
mean /= nb_samples
std /= nb_samples

# Print the results
print(f"Mean: {mean}")
print(f"Std: {std}")
