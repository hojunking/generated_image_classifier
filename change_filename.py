import os

# Define the directory containing the files to be renamed
directory = './data/imagen_dataset'  # Adjust this path to your specific directory

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file name contains whitespace
    if ' ' in filename:
        # Construct the new file name by replacing spaces with underscores
        new_filename = filename.replace(' ', '_')
        # Construct the full old and new file paths
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{filename}' to '{new_filename}'")
