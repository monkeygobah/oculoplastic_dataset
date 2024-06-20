import os
import numpy as np

from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

import re
from utils import *
from PIL import Image, ImageOps
from plotting import *

import pandas as pd
import csv


from sam_model import get_bounding_boxes, SAM





def extract_features_from_mask(mask, idx, gt=False, split_face=False):
    if gt:
        features_array = {
            'right_iris': (mask == 6), 
            'left_iris': (mask == 5),   
            'right_sclera': np.logical_or(mask == 2, mask == 6),
            'left_sclera': np.logical_or(mask == 1, mask == 5),
            'right_brow': (mask == 4),
            'left_brow': (mask == 3)     
        }

    if split_face and not gt:
        # The new output mask initialized to zeros, same shape as input mask
        new_mask = np.zeros_like(mask)
        
        # Midpoint of the image to split left and right
        mid = mask.shape[1] // 2
        unique_3 = np.unique(new_mask)

        # Update left side
        new_mask[:, :mid][mask[:, :mid] == 1] = 2  # Left Sclera
        new_mask[:, :mid][mask[:, :mid] == 2] = 4  # Left Brow
        new_mask[:, :mid][mask[:, :mid] == 3] = 6  # Left Iris

        # Update right side
        new_mask[:, mid:][mask[:, mid:] == 1] = 1  # Right Sclera
        new_mask[:, mid:][mask[:, mid:] == 2] = 3  # Right Brow
        new_mask[:, mid:][mask[:, mid:] == 3] = 5  # Right Iris

        features_array = {
            'right_iris': (new_mask == 6), 
            'left_iris': (new_mask == 5),   
            'right_sclera': np.logical_or(new_mask == 2, new_mask == 6),
            'left_sclera': np.logical_or(new_mask == 1, new_mask == 5),
            'right_brow': (new_mask == 4),
            'left_brow': (new_mask == 3)     
        }

    features_transposed = {key: np.transpose(np.nonzero(value)) for key, value in features_array.items()}

    return features_transposed, features_array



def calculate_mae_for_all_images(names, gt_measurements_list, gt_landmarks_list, pred_measurements_list, pred_landmarks_list):
    # Initialize a list to store MAE results for each image
    image_mae_results = []

    # Iterate over all images
    for name, gt_measurements, gt_landmarks, measurements, landmarks in zip(names, gt_measurements_list, gt_landmarks_list, pred_measurements_list, pred_landmarks_list):
        # Initialize a dictionary for this image's MAE results
        image_mae = {'image_name': name}

        # Scaling factors based on iris diameters
        gt_cf = 11.71 / ((gt_landmarks['right_iris_diameter'] + gt_landmarks['left_iris_diameter']) / 2)
        pred_cf = 11.71 / ((landmarks['right_iris_diameter'] + landmarks['left_iris_diameter']) / 2)

        excluded_keys = ['left_vd_plot_point', 'right_vd_plot_point']
        special_keys = ['right_canthal_tilt', 'left_canthal_tilt', 'right_scleral_area', 'left_scleral_area']

        # Calculate MAE for each measurement and landmark, and store it in the dictionary
        for key in measurements.keys():
            if key not in excluded_keys:
                gt_val = gt_measurements.get(key, 0)  # Default to 0 if key not found in gt
                pred_val = measurements.get(key, 0)  # Default to 0 if key not found in predictions

                # Apply scaling if not a special key
                if key not in special_keys:
                    gt_val *= gt_cf
                    pred_val *= pred_cf
                
                # Calculate the MAE for this key
                image_mae[key] = abs(gt_val - pred_val)

        # if landmarks['brow_error']:
        #     brow_error_message = 'brow_error'

        #     image_mae['sup_left_medial_bh'] = brow_error_message
        #     image_mae['sup_left_central_bh'] = brow_error_message
        #     image_mae['sup_left_lateral_bh'] = brow_error_message

        #     image_mae['sup_right_medial_bh'] = brow_error_message
        #     image_mae['sup_right_central_bh'] = brow_error_message
        #     image_mae['sup_right_lateral_bh'] = brow_error_message
            
        #     image_mae['left_medial_bh'] = brow_error_message
        #     image_mae['left_central_bh'] = brow_error_message
        #     image_mae['left_lateral_bh'] = brow_error_message

        #     image_mae['right_medial_bh'] = brow_error_message
        #     image_mae['right_central_bh'] = brow_error_message
        #     image_mae['right_lateral_bh'] = brow_error_message
        #     # pass

        # if not landmarks['brow_error'] and gt_landmarks['brow_error']:
        #     pass

        # if landmarks['brow_error'] and not gt_landmarks['brow_error']:
        #     pass

        # if not landmarks['brow_error'] and not gt_landmarks['brow_error']:
        #     pass 
        #     image_mae['sup_left_medial_bh'] = sup_left_mc_bh
        #     image_mae['sup_left_central_bh'] = sup_left_central_bh
        #     image_mae['sup_left_lateral_bh'] = sup_left_lc_bh

        #     image_mae['sup_right_medial_bh'] = sup_right_mc_bh
        #     image_mae['sup_right_central_bh'] = sup_right_central_bh
        #     image_mae['sup_right_lateral_bh'] = sup_right_lc_bh
            
        #     image_mae['left_medial_bh'] = left_mc_bh
        #     image_mae['left_central_bh'] = left_central_bh
        #     image_mae['left_lateral_bh'] = left_lc_bh

        #     image_mae['right_medial_bh'] = right_mc_bh
        #     image_mae['right_central_bh'] = right_central_bh
        #     image_mae['right_lateral_bh'] = right_lc_bh


        # Calculate MAE for the selected landmarks
        for landmark in ['right_iris_diameter', 'left_iris_diameter']:
            gt_land = gt_landmarks.get(landmark, 0)
            pred_land = landmarks.get(landmark, 0)
            image_mae[landmark] = abs(gt_land - pred_land)

        # Append this image's MAE results to the list
        image_mae_results.append(image_mae)

    # Create a DataFrame from the list of MAE results
    df_mae = pd.DataFrame(image_mae_results)
    print(df_mae.head())
    df_mae.set_index('image_name', inplace=True)

    return df_mae

def get_temp_iris_mask(csv_path, split_path, path, dataset, sam, transformer):
        # 1. find bounding box for l and r iris from csv file based on name
    l_iris, r_iris = get_bounding_boxes(csv_path, split_path, dataset)
    # 2. submit bounding box to SAM
    # 3. return mask of l and r iris
    img = cv2.imread(path)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes = np.array([r_iris,l_iris])
    masks_dict = sam.segment_no_jitter(img, boxes)

    # 4. make a temporary numpy array of 512 x 512
    temp_iris_mask = np.zeros((512, 512), dtype=np.uint8)

    # 5. resize the returned binary mask using the same logic for transform
    # mask_transform = transformer(dynamic_resize_and_pad=True, totensor=False, normalize=False, centercrop=False, imsize=512)
    l_iris_transform = transformer(Image.fromarray(masks_dict['left_iris']))
    r_iris_transform = transformer(Image.fromarray(masks_dict['right_iris']))


    # 6. Paste the iris mask into the temp array w labels 5 and 6 for l and r
    # 7. Store in iria masks list
    # Convert transformed PIL Images back to numpy arrays for further processing
    l_iris_mask_np = np.array(l_iris_transform)
    r_iris_mask_np = np.array(r_iris_transform)

    temp_iris_mask[l_iris_mask_np > 0] = 3  
    temp_iris_mask[r_iris_mask_np > 0] = 3

    return temp_iris_mask


def extract_numeric_id(file_path, dataset):
    if dataset == 'ted':
    # Extracts numeric part from the filename, assuming it follows 't_' and precedes '_crop'
        match = re.search(r't_(\d+)_crop', file_path)
        return int(match.group(1)) if match else None
    elif dataset == 'md':
        match = re.search(r'md_(\d+)_crop', file_path)
        return int(match.group(1)) if match else None 
    elif dataset == 'cfd':
        # The ID may include uppercase/lowercase letters, digits, and hyphens
        match = re.search(r'CFD-([A-Za-z0-9-]+)-N?_crop', file_path)
        return match.group(1) if match else None 


def align_images_and_labels(test_image_paths, gt_test_paths,dataset):
    # Create dictionaries with the numeric ID as the key
    test_images_dict = {extract_numeric_id(path, dataset): path for path in test_image_paths}
    gt_images_dict = {extract_numeric_id(path, dataset): path for path in gt_test_paths}
    
    aligned_test_images = []
    aligned_gt_images = []
    
    # Align based on numeric IDs
    for numeric_id in test_images_dict.keys():
        if numeric_id in gt_images_dict:
            aligned_test_images.append(test_images_dict[numeric_id])
            aligned_gt_images.append(gt_images_dict[numeric_id])
    
    return aligned_test_images, aligned_gt_images






def crop_and_resize(img):
    # Crop the image into left and right halves
    mid = img.width // 2
    left_half = img.crop((0, 0, mid, img.height))
    # Adjust the start of the right half if the width is not divisible by 2
    right_half_start = mid if img.width % 2 == 0 else mid + 1
    right_half = img.crop((right_half_start, 0, img.width, img.height))
    # Resize each half to 256x256
    left_resized = left_half.resize((256, 256))
    right_resized = right_half.resize((256, 256))
    return left_resized, right_resized


def transform_img_split(resize, totensor, normalize):
    options = []

    if resize:
        options.append(transforms.Lambda(crop_and_resize))

    if totensor:
        # Adjust to handle a pair of images (left and right halves)
        options.append(transforms.Lambda(lambda imgs: (transforms.ToTensor()(imgs[0]), transforms.ToTensor()(imgs[1]))))
        
    if normalize:
        # Normalize each image in the pair
        options.append(transforms.Lambda(lambda imgs: (transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgs[0]), 
                                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgs[1]))))
        
    transform = transforms.Compose(options)
    return transform



def make_dataset(dir, gt=False, custom=True):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    f = dir.split('/')[-1].split('_')[-1]
    print (dir, len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]))
    if custom:
        for root, dirs, files in os.walk(dir):
            for file in files:
                if 'checkpoint' not in file:
                    path = os.path.join(dir, file)
                    images.append(path)

    else:
        for i in range(len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])):
            if not gt:
                i = i
                img = str(i) + '.jpg'
            else:
                img = str(i) + '.png'
            path = os.path.join(dir, img)
            images.append(path)
       
    return images




def write_dice_scores_to_csv(storage, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row, assuming all dictionaries have the same keys
        if storage:
            headers = ['Batch Index'] + [f'Dice Score Class {cls}' for cls in storage[0].keys()]
            writer.writerow(headers)
            
            # Write each set of Dice scores to the CSV file
            for i, dice_scores in enumerate(storage):
                row = [i] + list(dice_scores.values())
                writer.writerow(row)


def apply_colormap(image, color_map):
    print("Unique pixel values in image:", np.unique(image))

    """Apply a colormap to a single-channel image."""
    colored_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for cls, color in color_map.items():
        colored_image[image == cls] = color
    return colored_image


# 2. 'l_eye', 'r_eye', 'l_brow', 'r_brow', = 1,2,3,4\
def relabel_classes(mask):
    mask[(mask == 3) | (mask == 2)] = 1
    mask[(mask == 5) | (mask == 6)] = 1

    mask[(mask == 3) | (mask == 4)] = 2    
    return mask
    
def visualize_and_dice(predictions, targets, batch_size, storage, corrected_masks, custom=False,dataset=None):
    """Visualize and save predicted and ground truth segmentation maps."""

    for idx in range(len(predictions)):
        pred_image = predictions[idx]
        if dataset != 'ted_long':
            gt_image = targets[idx]

            gt_image = np.squeeze(gt_image)
            gt_image = gt_image*255
            

            # New class mappings:
            # both eye and iris class should just be eye
            new_label_np = np.zeros_like(gt_image)
            new_label_np[(gt_image == 1) | (gt_image == 2)] = 1
            new_label_np[(gt_image == 3) | (gt_image == 4)] = 2
            new_label_np[(gt_image == 5) | (gt_image == 6)] = 1

            dice = dice_coefficient(pred_image, new_label_np, custom)

            # corrected_mask = mask_corrector(pred_image)
            storage.append(dice)
        corrected_masks.append(pred_image)
        
    return storage

def dice_coefficient(pred, target, custom=True, num_classes=3):
    """Compute the Dice score, combining left and right brow and sclera into single labels."""
    
    temp_pred = np.copy(pred)
    temp_target = np.copy(target) 
    

    dice_scores = {}
    for cls in range(num_classes):  # Now, only iterating over the combined classes
        pred_cls = (temp_pred == cls)
        target_cls = (temp_target == cls)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        dice_score = 2 * intersection / (union + 1e-6)
        dice_scores[cls] = dice_score
        
    return dice_scores

