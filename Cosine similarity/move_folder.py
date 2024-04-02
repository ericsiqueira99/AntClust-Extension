import os
import shutil

# Path to the folder containing the files
current_directory = os.path.dirname(os.path.realpath(__file__))

source = current_directory+"/image_test/"
dest = current_directory+"/data/"
# Get list of files in the folder
files = os.listdir(source)

# Create a folder for each first letter and move the file into it
for file_name in files:
    if os.path.isfile(os.path.join(source, file_name)):
        match = file_name.split("_")[0]
        new_folder_path = os.path.join(dest, match)
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        shutil.move(os.path.join(source, file_name), os.path.join(new_folder_path, file_name))
