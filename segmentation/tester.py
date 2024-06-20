
import os
import time
import torch
import datetime
import numpy as np
import pickle
from scipy.ndimage import label

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import PIL

from test_utils import *

import re
from unet import unet
from utils import *
from PIL import Image, ImageOps
from plotting import *

import pandas as pd
import csv

from maskExtraction import EyeFeatureExtractor
from measureAnatomy import EyeMetrics
from distance_plot import Plotter
from sam_model import get_bounding_boxes, SAM
from torchvision.models.segmentation import deeplabv3_resnet101

# SAM_CHECKPOINT_PATH = os.path.join('..','..', 'SAM_WEIGHTS', 'sam_vit_h_4b8939.pth')
# SAM_ENCODER_VERSION = "vit_h"


def transformer(dynamic_resize_and_pad, totensor, normalize, centercrop, imsize, is_mask=False):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if dynamic_resize_and_pad:
        options.append(ResizeAndPad(output_size=(imsize, imsize)))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(options)


class ResizeAndPad:
    def __init__(self, output_size=(512, 512), fill=0, padding_mode='constant'):
        self.output_size = output_size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # Calculate new height maintaining aspect ratio
        original_width, original_height = img.size
        new_height = int(original_height * (self.output_size[0] / original_width))
        img = img.resize((self.output_size[0], new_height), Image.NEAREST)

        # Calculate padding
        padding_top = (self.output_size[1] - new_height) // 2
        padding_bottom = self.output_size[1] - new_height - padding_top

        # Apply padding
        img = ImageOps.expand(img, (0, padding_top, 0, padding_bottom), fill=self.fill)
        
        return img
    

class Tester(object):
    def __init__(self, config, device):
        # exact model and loss
        self.model = config.model

        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel

        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.img_path = config.img_path
        self.label_path = config.label_path 
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        self.dataset = config.dataset
        self.device = device
        
        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        
        self.dlv3 = config.dlv3
        
        if self.dataset != 'ted_long':
            self.test_label_path = config.test_label_path
            self.test_label_path_gt = config.test_label_path_gt
            self.test_color_label_path = config.test_color_label_path
            
        self.test_image_path = config.test_image_path

        self.get_sam_iris = config.get_sam_iris

        # Test size and model
        self.test_size = config.test_size
        self.model_name = config.model_name
        
        self.train_limit = config.train_limit
        
        self.csv_path = config.csv_path
        
        self.pickle_in = config.pickle_in
        self.split_face = config.split_face

        self.build_model()

    def test(self):
        sam = SAM()
        plotter = Plotter()

        # if not self.pickle_in:
        if self.split_face:
            transform = transform_img_split(resize=True, totensor=True, normalize=True)
            transform_plotting = transform_img_split(resize=True, totensor=False, normalize=False)
            transform_plotting_sam = transformer(dynamic_resize_and_pad=True, totensor=False, normalize=False,centercrop=False, imsize=512)
            transform_gt = transformer(dynamic_resize_and_pad=True, totensor=True, normalize=False,centercrop=False, imsize=512)

        else:
            transform = transformer(dynamic_resize_and_pad=True, totensor=True, normalize=True, centercrop=False, imsize=512)
            transform_plotting = transformer(dynamic_resize_and_pad=True, totensor=False, normalize=False, centercrop=False, imsize=512)
            
        if self.dataset != 'celeb':
            custom = True
        else:
            custom = False
            
        # if custom, return list of paths, otherwise return list of paths for celeb that are numbers.jpg/png
        test_paths = make_dataset(self.test_image_path, custom=custom)
        if self.dataset != 'ted_long':
            gt_test_paths = make_dataset(self.test_label_path_gt, gt=True, custom=custom)
            print(f'length of test path is {len(test_paths)} and gt is {len(gt_test_paths)}')
        
        #align the paths so the indices match if custom dataset
        if custom and self.dataset != 'ted_long':
            test_paths, gt_test_paths = align_images_and_labels(test_paths, gt_test_paths, self.dataset)
            
        # make_folder(self.test_label_path, '')
        
        # make_folder(self.test_color_label_path, '') 
        
        # load model
        self.G.load_state_dict(torch.load(os.path.join(self.model_name)))
        self.G.eval() 
        
        batch_num = int(self.test_size / self.batch_size)
        storage = []
        corrected_masks = []
        iris_masks = []
        images_plotting = []
        names = []
        gt_for_storage = []

        # start prediction process
        for i in range(batch_num):
            
            imgs = []
            gt_labels = []
            l_imgs = []
            r_imgs = []
            original_sizes = []
            
            for j in range(self.batch_size):
                current_idx = i * self.batch_size + j
                if current_idx < len(test_paths):
                    path = test_paths[current_idx]
                    name = path.split('/')[-1][:-4]
                    names.append(name)
                    print(name)

                    # if splitting the face down the midline, use the appropriate transforms
                    if self.split_face:
                        original_sizes.append(Image.open(path).size)
                        l_img, r_img = transform(Image.open(path))
                        l_imgs.append(l_img)
                        r_imgs.append(r_img)
                        images_plotting.append(transform_plotting_sam(Image.open(path)))
                    
                    # otherwise use the original transform
                    else:
                        img = transform(Image.open(path))
                        imgs.append(img)
                        images_plotting.append(transform_plotting(Image.open(path)))
            
                    # when doing distance prediction and need sam to get the iris mask
                    if self.get_sam_iris:
                        temp_iris_mask = get_temp_iris_mask(self.csv_path, path.split('/')[-1], path, self.dataset, sam, transform_plotting_sam)
                        iris_masks.append(temp_iris_mask)

                    # if data has no labels (longitudinal TED), use this
                    if self.dataset != 'ted_long':
                        gt_path = gt_test_paths[current_idx]
                        gt_img = Image.open(gt_path)
                        gt_label = transform_plotting_sam(gt_img)
                        gt_labels.append(transform_gt(gt_img).numpy())
                        gt_for_storage.append(np.array(gt_label))

                else:
                    break 
                
                
            if len(imgs) != 0 or len(l_imgs)!=0:
                print('PREDICTING IMAGES NOW ')
                
                # get predictions when using split face method
                if self.split_face:
                    labels_predict_plain = predict_split_face(l_imgs,r_imgs, self.imsize, transform_plotting_sam, self.device, original_sizes, self.G, dlv3=self.dlv3)                        
    

                # otherwise predict using the original method
                # else:
                #     imgs = torch.stack(imgs) 
                #     imgs = imgs.to(self.device)

                #     # predict
                #     # labels_predict = self.G(imgs)
                #     if self.dlv3:
                #         labels_predict = self.G(imgs)['out'] 
                #     else:
                #         labels_predict = self.G(imgs)  
                #     # # # After obtaining predictions
                #     labels_predict_plain = generate_label_plain(labels_predict, self.imsize)
                # # if self.dataset != 'ted_long':

                # visualize and get the dice scores for plotting later
                visualize_and_dice(labels_predict_plain, np.array(gt_labels), self.batch_size, storage, corrected_masks, custom=custom, dataset = self.dataset)


        if self.dataset != 'ted_long':
            if self.train_limit is None:
                limit = 'all'
            else:
                limit = self.train_limit
            title = f'{self.dataset}_{limit}_test'
            if self.dlv3:
                csv_file_path = f'{title}_dlv3_dice_scores.csv'
            else:
                csv_file_path = f'{title}_dice_scores.csv'            
            write_dice_scores_to_csv(storage, csv_file_path)
            plot_dice_boxplots(storage, title)
        
        
        # plot_dice_histograms(storage, title)
        
        if self.get_sam_iris:
            # 1. replace the pixels in corrected masks with the iris masks
            # 2. Make a transposed dict of all of the masks so that we can reuse the old shitty measurement path
            # 3. Once everything is in right format, send down measurements without VD / canthal tilt as they need the midline points
            # 4. Once verified that these are correct, send down measurement pathway 
            integrated_masks = []

            for corrected_mask, iris_mask in zip(corrected_masks, iris_masks):
                if self.split_face:
                    corrected_mask[iris_mask == 3] = 3 
                # else:
                #     # Replace the left iris region in the corrected mask
                #     corrected_mask[iris_mask == 5] = 5
        
                #     # Replace the right iris region in the corrected mask
                #     corrected_mask[iris_mask == 6] = 6
                integrated_masks.append(corrected_mask)
            
            features_list = [extract_features_from_mask(mask,idx, split_face=self.split_face) for idx, mask in enumerate(integrated_masks)]
            
            if self.dataset != 'ted_long':
                gt_features_list = [extract_features_from_mask(mask,idx, gt=True, split_face=self.split_face) for idx,mask in enumerate(gt_for_storage)]
            
            
        pred_measurements = []
        pred_landmarks = []
        gt_measurements = []
        gt_landmarks = []

        bad_indices_pred = []
        bad_indices_gt = []
        print(len(features_list))
        # measurments for predictions
        print('ANALYZING AI PREDICTIONS NOW')

        for idx, features in enumerate(features_list):
            try:
                _, features_array = features  
                
                extractor = EyeFeatureExtractor(features_array, images_plotting[idx],idx)
                landmarks = extractor.extract_features()
                pred_landmarks.append(landmarks)

                # Create an instance of EyeMetrics with the landmarks
                eye_metrics = EyeMetrics(landmarks, features_array) 
                measurements = eye_metrics.run()
                pred_measurements.append(measurements)
                
                # store marked up images to see if this is working
                plotter.create_plots(images_plotting[idx], features_array, landmarks, names[idx], measurements, self.dlv3)
            except (ValueError, KeyError):
                bad_indices_pred.append(idx)

        if self.train_limit == None:
            save_to_csv(self.dataset, names, pred_measurements, pred_landmarks, dlv3=self.dlv3)
                

        #measurements for gt 
        if self.dataset != 'ted_long':
            print('ANALYZING GT NOW')
            for idx, features in enumerate(gt_features_list):
                try:
                    _, features_array_gt = features           

                    extractor_gt = EyeFeatureExtractor(features_array_gt, images_plotting[idx],idx, gt=True)
                    landmarks_gt = extractor_gt.extract_features()
                    gt_landmarks.append(landmarks_gt)

                    # Create an instance of EyeMetrics with the landmarks
                    eye_metrics_gt = EyeMetrics(landmarks_gt, features_array_gt) 
                    measurements_gt = eye_metrics_gt.run()
                    gt_measurements.append(measurements_gt)
                    plotter.create_plots(images_plotting[idx], features_array_gt, landmarks_gt, names[idx], measurements_gt, gt=True)

                except (ValueError, KeyError):
                    bad_indices_gt.append(idx)
                    
            if self.train_limit == None:
                save_to_csv(self.dataset, names, gt_measurements, gt_landmarks, gt=True)

                    
            print(f'PRINTING BAD INDICES PRED:{bad_indices_pred} and GT: {bad_indices_gt}. REMOVING PRED FROM GT ONLY')
            # all_bad_indices = set(bad_indices_pred) | set(bad_indices_gt)

            # # Remove bad indices from the names list
            # names = [name for i, name in enumerate(names) if i not in all_bad_indices]

            # # Remove bad indices from gt_measurements and gt_landmarks if they are in bad_indices_pred
            # gt_measurements = [m for i, m in enumerate(gt_measurements) if i not in bad_indices_pred]
            # gt_landmarks = [l for i, l in enumerate(gt_landmarks) if i not in bad_indices_pred]

            # # Remove bad indices from pred_measurements and pred_landmarks if they are in bad_indices_gt
            # pred_measurements = [m for i, m in enumerate(pred_measurements) if i not in bad_indices_gt]
            # pred_landmarks = [l for i, l in enumerate(pred_landmarks) if i not in bad_indices_gt]
            
            title = f'{self.dataset}_{self.train_limit}_dlv3_{self.dlv3}test'
            mae_df = calculate_mae_for_all_images(names, gt_measurements, gt_landmarks, pred_measurements, pred_landmarks)        
            mae_df.to_csv(f'{title}_mae.csv')



    # def build_model(self):
    #     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     print(f"Using device: {self.device}")
    #     self.G = unet().to(self.device)
    #     if self.parallel:
    #         self.G = nn.DataParallel(self.G)


    def build_model(self):
        if not self.dlv3:
            self.G = unet().to(self.device)
        elif self.dlv3:
            self.G = deeplabv3_resnet101(num_classes=3).to(self.device)
 
        if self.parallel:
            self.G = nn.DataParallel(self.G)



def save_to_csv(dataset, names, measurements, landmarks, gt=False, dlv3=False):
    # Save measurements to CSV
    if gt:
        title = f'{dataset}_dlv3_{dlv3}_measurements_GROUND_TRUTH.csv'
        with open(title, 'w', newline='') as file:
            writer = csv.writer(file)
            
            # Write header row
            header = ['Name'] + list(measurements[0].keys()) + ['right_iris_diameter', 'left_iris_diameter']
            writer.writerow(header)
            
            # Write data rows
            for name, measurement, landmark in zip(names, measurements, landmarks):
                row = [name] + list(measurement.values()) + [landmark['right_iris_diameter'], landmark['left_iris_diameter']]
                writer.writerow(row)
    else:
        with open(f'{dataset}_dlv3_{dlv3}_measurements.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            
            # Write header row
            header = ['Name'] + list(measurements[0].keys()) + ['right_iris_diameter', 'left_iris_diameter']
            writer.writerow(header)
            
            # Write data rows
            for name, measurement, landmark in zip(names, measurements, landmarks):
                row = [name] + list(measurement.values()) + [landmark['right_iris_diameter'], landmark['left_iris_diameter']]
                writer.writerow(row)
        
 






def predict_split_face(l_imgs,r_imgs, imsize, transform_plotting_sam, device, original_sizes, G, dlv3=False):
    l_imgs = torch.stack(l_imgs) 
    r_imgs = torch.stack(r_imgs) 
    
    print(l_imgs.size())
    
    l_imgs = l_imgs.to(device)
    r_imgs = r_imgs.to(device)
    

    if dlv3:
        l_labels_predict = G(l_imgs)['out'] 
        r_labels_predict = G(r_imgs)['out']  
        
    else:
        l_labels_predict = G(l_imgs)
        r_labels_predict = G(r_imgs)
    
    l_labels_predict_plain = generate_label_plain(l_labels_predict, imsize)
    r_labels_predict_plain = generate_label_plain(r_labels_predict, imsize)
    
    labels_predict_plain = []

    for idx, (left_pred, right_pred) in enumerate(zip(l_labels_predict_plain, r_labels_predict_plain)):
        original_width, original_height = original_sizes[idx]
        mid = original_width // 2
        
        # Calculate dimensions for left and right halves based on the original sizes
        left_width = mid  # Since mid is the midpoint
        right_width = original_width - mid  # Width from midpoint to right edge

        # Resize predictions to match these dimensions
        left_pred_resized = cv2.resize(left_pred, (left_width, original_height), interpolation=cv2.INTER_NEAREST)
        right_pred_resized = cv2.resize(right_pred, (right_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Create a new empty array for the stitched prediction
        stitched = np.zeros((original_height, original_width), dtype=np.uint8)

        # Place the resized predictions onto the stitched canvas
        stitched[:, :mid] = left_pred_resized
        stitched[:, mid:] = right_pred_resized
    
        
        # Resize stitched prediction to 512x512
        resized_stitched = transform_plotting_sam(Image.fromarray(stitched))

        labels_predict_plain.append(np.array(resized_stitched))

    print('converting labels')
    labels_predict_plain = np.array(labels_predict_plain)
    print(len(labels_predict_plain))
    
    return labels_predict_plain