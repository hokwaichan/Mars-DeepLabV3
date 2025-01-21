import os
from PIL import Image

def convert_grayscale_to_rgb(dataset_dir, output_dir):
    """
    Convert grayscale images in the dataset to RGB and save them to an output directory.

    Args:
        dataset_dir (str): Path to the dataset directory containing images.
        output_dir (str): Path to the output directory for saving converted images.
    """
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Traverse the dataset directory
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Define output file path by maintaining the directory structure
            output_path = os.path.join(output_dir, os.path.relpath(file_path, dataset_dir))
            
            # Create any necessary subdirectories for the output path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                # Open the image file
                with Image.open(file_path) as img:
                    # Check if the image is grayscale
                    if img.mode == "L":  # Grayscale
                        # Convert grayscale image to RGB
                        rgb_img = img.convert("RGB")
                        rgb_img.save(output_path)
                    else:  # Image is already in RGB mode
                        img.save(output_path)
            except Exception as e:
                # Print error if the image cannot be processed
                print(f"Error processing {file_path}: {e}")

# Example usage
dataset_dir = ""  # Path to the dataset
output_dir = ""  # Path to save converted images
convert_grayscale_to_rgb(dataset_dir, output_dir)
