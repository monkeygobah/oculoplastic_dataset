import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the list of common labels
label_list = ['iris', 'brow', 'sclera', 'caruncle', 'lid']

# Directories
folder_save = 'full_celeb_annotations-SAVE'
img_num = 1000

# Function to normalize and colorize the labels for visualization
def visualize_annotation(image, num_classes):
    # Scale pixel values to the range [0, 255] for visualization
    image = (image / num_classes * 255).astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_JET)

visualize_me = [0,100,101,102,103]
for k in visualize_me:
    filename = os.path.join(folder_save, f'{str(k)}_celeb.png')  # Update this path to one of your actual mask files
    if os.path.exists(filename):
        im_base = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # Read the combined mask as grayscale

        # Check the unique values in the mask to ensure it's read correctly
        print(f"Unique values in mask {filename}: {np.unique(im_base)}")

        # Visualize a few images
        im_colored = visualize_annotation(im_base, len(label_list))
        plt.figure(figsize=(5, 5))
        plt.imshow(im_colored)
        plt.title(f"Annotation for image {k}")
        plt.axis('off')
        plt.savefig(f'{k}_vis.jpg')
        plt.show()