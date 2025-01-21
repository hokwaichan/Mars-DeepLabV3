import os
from PIL import Image

def check_image_modes(dataset_dir):
    """
    Check the modes (grayscale or RGB) of images in a dataset.

    Args:
        dataset_dir (str): Path to the dataset directory containing images.
    """
    # Initialize counters for grayscale, RGB, and invalid images
    grayscale_count = 0
    rgb_count = 0
    invalid_count = 0
    invalid_files = []

    # Traverse through all files in the dataset directory
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Checking file: {file_path}")  # Print the file being checked

            try:
                # Open the image file
                with Image.open(file_path) as img:
                    # Check the image mode
                    if img.mode == "L":  # Grayscale image
                        grayscale_count += 1
                    elif img.mode == "RGB":  # RGB image
                        rgb_count += 1
                    else:
                        # Count invalid files (neither Grayscale nor RGB)
                        invalid_count += 1
                        invalid_files.append(file_path)
            except Exception as e:
                # Handle files that are not valid images (e.g., corrupted files)
                invalid_count += 1
                invalid_files.append(file_path)

    # Print the dataset summary
    print("\nDataset Summary:")
    print(f"Grayscale images: {grayscale_count}")
    print(f"RGB images: {rgb_count}")
    print(f"Invalid or unsupported files: {invalid_count}")

    # List invalid files (optional)
    if invalid_files:
        print("\nInvalid or unsupported files:")
        for file_path in invalid_files:
            print(file_path)


# Example usage
dataset_dir = ""  # Replace with your dataset path
check_image_modes(dataset_dir)
