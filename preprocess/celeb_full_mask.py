import os
import cv2
import numpy as np
from utils import make_folder

# Define the list of common labels
label_list = {
    'iris': ['l_iris', 'r_iris'],
    'brow': ['l_brow', 'r_brow'],
    'sclera': ['l_sclera', 'r_sclera'],
    'caruncle': ['l_caruncle', 'r_caruncle'],
    'lid': ['l_lid', 'r_lid']
}

# Directories
folder_base = 'full_celeb_annotations'
folder_save = 'full_celeb_annotations-SAVE'

# Create save directory if it doesn't exist
make_folder(folder_save)

# Dictionary to store combined masks
dim_dict = {}

# Initialize the masks in dim_dict
for root, dirs, files in os.walk(folder_base):
    for file in files:
        if 'checkpoint' not in file and file != '.DS_Store':
            filename = '_'.join(file.split('_')[:2])
            if filename not in dim_dict:
                im = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    dim_dict[filename] = np.zeros_like(im)

# Populate the combined masks
for root, dirs, files in os.walk(folder_base):
    for file in files:
        if 'checkpoint' not in file and file != '.DS_Store':
            filename = '_'.join(file.split('_')[:2])
            if filename in dim_dict:
                im_base = dim_dict[filename]
                for idx, (common_label, labels) in enumerate(label_list.items()):
                    for label in labels:
                        struct_filename = os.path.join(folder_base, f'{filename}_crop_{label}.png')
                        # print(struct_filename)
                        if os.path.exists(struct_filename):
                            im = cv2.imread(struct_filename, cv2.IMREAD_GRAYSCALE)
                            if im is not None:
                                im_base[im != 0] = idx + 1

# Save the combined masks
for filename, im_base in dim_dict.items():
    filename_save = os.path.join(folder_save, f'{filename}.png')
    cv2.imwrite(filename_save, im_base)

print("Combined masks have been saved successfully.")
