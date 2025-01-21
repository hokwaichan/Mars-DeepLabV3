import os

def rename_images_in_folder(folder_path, start_number=1):
    """
    Renames all image files in the specified folder to a sequential pattern (e.g., 000001).
    
    Parameters:
        folder_path (str): Path to the folder containing the images.
        start_number (int): Starting number for renaming (default is 1).
    """
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    # Get all files in the folder
    files = os.listdir(folder_path)

    # Filter for image files based on the extensions
    images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    
    # Optional: Sort files alphabetically (can be useful if filenames are not ordered)
    images.sort()

    # Rename each image to a sequential name starting from 'start_number'
    for i, image in enumerate(images, start=start_number):
        # Old path (current location of the file)
        old_path = os.path.join(folder_path, image)
        
        # New name with sequential numbering (e.g., 000001.jpg)
        new_name = f"{i:06d}{os.path.splitext(image)[1].lower()}"  # Preserve file extension
        
        # New path (location where the file will be renamed)
        new_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {image} -> {new_name}")

# Example usage
folder_path = ""  # Replace with the actual folder path
rename_images_in_folder(folder_path)
