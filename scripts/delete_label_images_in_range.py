import os

# Define the folder where the images are stored
folder_path = ''  # Replace with the actual folder path

# Define the start and end image numbers for the range
start_num = 15415
end_num = 15766

# Loop through the range of image numbers
for num in range(start_num, end_num + 1):
    # Format the number to have leading zeros (e.g., 006001.png)
    filename = f"{num:06d}.png"
    file_path = os.path.join(folder_path, filename)
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Delete the file if it exists
        os.remove(file_path)
        print(f"Deleted: {filename}")
    else:
        # Print a message if the file was not found
        print(f"File not found: {filename}")
        