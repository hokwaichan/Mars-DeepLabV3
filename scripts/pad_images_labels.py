from PIL import Image, ImageOps
import os

def pad_to_size(image, target_size=(1200, 1200), fill_value=0):
    """
    Pads the input image or label to the target size.

    Args:
        image (PIL.Image): The input image or label.
        target_size (tuple): The desired size (height, width).
        fill_value (int or tuple): The padding value (default 0 for labels, or RGB value for images).

    Returns:
        PIL.Image: The padded image or label.
    """
    width, height = image.size
    target_width, target_height = target_size

    # Calculate padding for each side
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    right = target_width - width - left
    bottom = target_height - height - top

    # Adjust fill_value based on image mode
    if image.mode in ("L", "I"):  # Grayscale or single-channel
        fill_value = 0  # Ensure it's an integer for grayscale
    elif image.mode in ("RGB", "RGBA"):  # RGB or RGBA
        fill_value = (0, 0, 0)  # Ensure it's a tuple for RGB images

    # Pad the image with the calculated padding and fill value
    return ImageOps.expand(image, border=(left, top, right, bottom), fill=fill_value)

def pad_images_and_labels(path, output_path, target_size=(1200, 1200)):
    """
    Processes images and labels from a single input path, pads them, and saves to output_path.

    Args:
        path (str): Path to the directory containing images and labels.
        output_path (str): Path to save the padded images and labels.
        target_size (tuple): Desired size (height, width).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist.")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Loop through files in the directory
    for file_name in os.listdir(path):
        if file_name.endswith(('.jpg', '.png', '.jpeg')):  # Check if the file is an image
            full_path = os.path.join(path, file_name)
            try:
                # Open the image file
                image = Image.open(full_path)

                # Determine padding value based on the file name
                if "label" in file_name.lower():  # Assuming "label" in filename identifies labels
                    fill_value = 0  # Padding value for labels (grayscale)
                else:
                    fill_value = (0, 0, 0)  # Padding value for RGB images

                # Pad the image or label
                padded_image = pad_to_size(image, target_size, fill_value)

                # Save the padded image to the output directory
                padded_output_path = os.path.join(output_path, file_name)
                padded_image.save(padded_output_path)
                print(f"Padded and saved: {padded_output_path}")

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

# Example usage
input_path = ''  # Input directory containing images and labels
output_path = ''  # Directory to save padded files
pad_images_and_labels(input_path, output_path, target_size=(1200, 1200))
