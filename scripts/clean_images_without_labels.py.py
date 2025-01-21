import os

# Define the paths for the images and labels directories
images_dir = ''
labels_dir = ''

# Function to extract the relevant prefix from filenames
def extract_prefix(filename):
    """
    Extracts the prefix from a filename by splitting at 'EDR' or 'RNG'.
    Returns the part of the filename before these keywords.
    If neither keyword is found, returns the entire filename.
    """
    if 'EDR' in filename:
        return filename.split('EDR')[0]
    elif 'RNG' in filename:
        return filename.split('RNG')[0]
    return filename

# Get the set of prefixes from image and label files
image_files = set(extract_prefix(file) for file in os.listdir(images_dir))
label_files = set(extract_prefix(file) for file in os.listdir(labels_dir))

# Identify images without a corresponding label
images_without_labels = image_files - label_files

# Find the full list of image files to remove (those without matching labels)
unmatched_images = [file for file in os.listdir(images_dir) if extract_prefix(file) in images_without_labels]

# Remove unmatched images
for file in unmatched_images:
    file_path = os.path.join(images_dir, file)
    print(f"Removing unmatched image: {file_path}")
    os.remove(file_path)

# Print the total number of files removed
total_removed = len(unmatched_images)
print(f"Total unmatched images removed: {total_removed}")
