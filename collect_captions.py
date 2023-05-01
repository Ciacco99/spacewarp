import os

# Define the folder path containing the image files
folder_path = "data/out/unplash-512-lite/00002/"

# Get the last folder name from the folder path
last_folder_name = os.path.basename(os.path.normpath(folder_path))


# Create an empty list to store the descriptions
descriptions = []

# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    # Check if the file has a .txt extension
    if filename.endswith(".txt"):
        # Read the textual description
        with open(os.path.join(folder_path, filename), "r") as f:
            description = f.read().strip()
            # Remove the .txt extension from the filename
            filename_without_extension = os.path.splitext(filename)[0]
            descriptions.append((filename_without_extension, description))

# Sort the list based on filenames just to be a bit more organized
descriptions.sort(key=lambda x: x[0])


# Save all descriptions and filenames in a single file
output_file = f"{last_folder_name}_image_descriptions.txt"
with open(output_file, "w") as f:
    for filename, description in descriptions:
        f.write(f"{filename}: {description}\n")
