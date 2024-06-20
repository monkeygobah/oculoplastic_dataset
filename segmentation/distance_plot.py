import cv2
import matplotlib.pyplot as plt
import numpy as np 
import os
from sam_model import SAM




# abse class to get everything started and allow child classes to inherit 
class BasePlotter:
    def __init__(self, image, mask_dict, landmark_dict, image_name, measurements,dlv3,gt=False):
        config = {
            'outputs': {
                'iris_outline': True,
                'pupil_outline': False,
                'distance_annotation': True,
                'masks': {
                    'individual': False,
                    'total': True
                },
            },
            'paths': {
                'annotation': f'distance_annotations/lines/',
                'total' : f'distance_annotations/masks/',
            }
        }
        self.image = image
        self.masks = mask_dict
        self.landmarks = landmark_dict
        self.name = image_name
        self.measurements = measurements
        self.config_dict = config
        self.gt=gt
        self.dlv3 = dlv3
        
    def plot(self):
        raise NotImplementedError
    
        
# draw the lines on the face        
class LineAnnotator(BasePlotter):
    def __init__(self, image, mask_dict, landmark_dict, image_name, measurements, dlv3,gt=False):
        super().__init__(image, mask_dict, landmark_dict, image_name, measurements, dlv3,gt=gt)

    def _draw_line(self, point1, point2, color='black', linewidth=1, force_horizontal = False):
        if force_horizontal:
            plt.plot([point1[0], point2[0]], [point2[1], point2[1]], color=color, linewidth=linewidth)
        else:        
            """Helper function to draw a line between two points."""
            plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color=color, linewidth=linewidth)
        
    def draw_landmarks(self):
        if not self.config_dict['outputs']['distance_annotation']:
            return

        output_path = self.config_dict['paths']['annotation']
        
        plt.imshow(self.image)

        try:    
            # Left Brow Heights
            brow_data = {
                'l_medial_eyebrow': ['left_medial_canthus', 'l_medial_eyebrow', 'purple',2],
                'l_center_eyebrow': ['left_iris_centroid', 'l_center_eyebrow', 'purple',2],
                'l_lat_eyebrow'   : ['left_lateral_canthus', 'l_lat_eyebrow', 'purple',2],
                'r_medial_eyebrow': ['right_medial_canthus', 'r_medial_eyebrow', 'purple',2],
                'r_center_eyebrow': ['right_iris_centroid', 'r_center_eyebrow', 'purple',2],
                'r_lat_eyebrow'   : ['right_lateral_canthus', 'r_lat_eyebrow', 'purple',2],
                
                'sup_l_medial_eyebrow': ['left_medial_canthus', 'sup_l_medial_eyebrow', 'black',1],
                'sup_l_center_eyebrow': ['left_iris_centroid', 'sup_l_center_eyebrow', 'black',1],
                'sup_l_lat_eyebrow'   : ['left_lateral_canthus', 'sup_l_lat_eyebrow', 'black',1],
                'sup_r_medial_eyebrow': ['right_medial_canthus', 'sup_r_medial_eyebrow', 'black',1],
                'sup_r_center_eyebrow': ['right_iris_centroid', 'sup_r_center_eyebrow', 'black',1],
                'sup_r_lat_eyebrow'   : ['right_lateral_canthus', 'sup_r_lat_eyebrow', 'black',1]
            }
 
            for key, (start, end, color, thickness) in brow_data.items():
        
                if not np.array_equal(self.landmarks[key], np.array([0,0])):
                    
                    self._draw_line(self.landmarks[start], self.landmarks[end], color=color, linewidth=thickness)
                    
        except KeyError:
            pass
        scleral_show_data = {'left_SSS' : ['left_sclera_superior', 'left_iris_superior', 'left_iris_centroid', 'lightblue' ], 
                             'left_ISS' : ['left_sclera_inferior', 'left_iris_inferior','left_iris_centroid', 'orange' ],
                             'right_SSS': ['right_sclera_superior', 'right_iris_superior', 'right_iris_centroid', 'lightblue'] ,
                             'right_ISS': ['right_sclera_inferior', 'right_iris_inferior','right_iris_centroid', 'orange' ]
                             }
        thickness = 1
        for key, (sclera, iris_dir, center, color) in scleral_show_data.items():
            if self.measurements[key] != 0:
                plt.plot([self.landmarks[sclera][0],self.landmarks[iris_dir][0]], [self.landmarks[sclera][1], self.landmarks[iris_dir][1]], color='blue',linewidth=thickness)
                plt.plot([self.landmarks[center][0], self.landmarks[iris_dir][0]], [self.landmarks[center][1], self.landmarks[iris_dir][1]], color=color,linewidth=thickness)
       
            else:
                plt.plot([self.landmarks[center][0], self.landmarks[sclera][0]], [self.landmarks[center][1], self.landmarks[sclera][1]], color=color,linewidth=thickness)


        # Direct plot scenarios (i.e., ones without conditions or needing the helper function)
        self._draw_line(self.landmarks['left_medial_canthus'], self.landmarks['left_lateral_canthus'], color='lightblue', linewidth=thickness, force_horizontal=True) # horiz fisure
        self._draw_line(self.landmarks['right_medial_canthus'], self.landmarks['right_lateral_canthus'], color='lightblue', linewidth=thickness, force_horizontal=True) #horiz fissure
        self._draw_line(self.landmarks['left_iris_centroid'], self.landmarks['right_iris_centroid'], color='black', linewidth=.5, force_horizontal=True) #IPD
        self._draw_line(self.landmarks['left_medial_canthus'], self.landmarks['right_medial_canthus'], color='gray', linewidth=.5, force_horizontal=True ) #ICD
        self._draw_line(self.landmarks['left_lateral_canthus'], self.landmarks['right_lateral_canthus'], color='green', linewidth=.5, force_horizontal=True) #ICD

        # self._draw_line(self.landmarks['right_sclera_superior'], self.landmarks['right_sclera_inferior'], color='red', linewidth=5) 
        # self._draw_line(self.landmarks['left_sclera_superior'], self.landmarks['left_sclera_inferior'], color='red', linewidth=5) 
        
        plt.axis('off')

        # Save the annotated image
        if self.gt:
            plt.savefig(os.path.join(output_path, f"{self.name}_GT_landmarks_dlv3_{self.dlv3}.jpg"), dpi=300)
        else:
            plt.savefig(os.path.join(output_path, f"{self.name}_landmarks_dlv3_{self.dlv3}.jpg"), dpi=300)
        plt.close()

    def plot(self):
        self.draw_landmarks()
        
class MaskDrawer(BasePlotter):
    def __init__(self, image, mask_dict, landmark_dict, image_name, measurements, dlv3, gt=False):
        super().__init__(image, mask_dict, landmark_dict, image_name,measurements,dlv3, gt=gt)
    
    def total_masker(self):
        alpha=.2
        colors = [np.array([0,1,0,alpha]), np.array([1,0,0,alpha]),np.array([0,0,1,alpha]),  np.array([1,1,0,alpha]), np.array([1,0,1,alpha]),  np.array([1,0,1,alpha]) ]
        if self.config_dict['outputs']['masks']['total']:
            output_path = self.config_dict['paths']['total']
            # plt.figure()
            plt.imshow(self.image)

            for idx, mask in enumerate(['left_brow', 'right_brow', 'left_sclera', 'right_sclera', 'left_iris', 'right_iris']):
                SAM.show_mask(self.masks[mask], plt, colors[idx])

            # plt.show()
            plt.axis('off')
            if self.gt:
                plt.savefig(os.path.join(output_path, f"{self.name}_GT_total_dlv3_{self.dlv3}.jpg"), dpi=300)

            else:
                plt.savefig(os.path.join(output_path, f"{self.name}_total_dlv3_{self.dlv3}.jpg"), dpi=300)
            plt.close()

    def plot(self):
        self.total_masker()
        
# class BoxDrawer(BasePlotter):
#     def __init__(self, image, mask_dict, landmark_dict, image_name, boxes,measurements):
#         super().__init__(image, mask_dict, landmark_dict, image_name, boxes,measurements)
    
#     def total_boxer(self):
#         if self.config_dict['outputs']['boxes']:
#             output_path = self.config_dict['paths']['boxes']
#             fig, ax = plt.subplots()
#             ax.imshow(self.image)
#             # skip the last box as it is the midline
#             for box in self.boxes:
#                 SAM.show_box(box, ax)
#             ax.axis('off')
#             plt.savefig(os.path.join(output_path, f"{self.name}_bbs.jpg"), dpi=300)
#             plt.close()

#     def plot(self):
#         self.total_boxer()

class Plotter:
    def __init__(self):
        pass
    
    def create_plots(self, image, mask_dict, landmark_dict, image_name, measurements,dlv3=False, gt=False):  
        if gt:

            line_annotator = LineAnnotator(image, mask_dict, landmark_dict, image_name, measurements,dlv3,gt=True)
            mask_drawer = MaskDrawer(image, mask_dict, landmark_dict, image_name, measurements,dlv3, gt=True)
            
            mask_drawer.plot()
            line_annotator.plot()
                   

        else:
            # Create an instance for each plotter
            line_annotator = LineAnnotator(image, mask_dict, landmark_dict, image_name,measurements,dlv3)
            mask_drawer = MaskDrawer(image, mask_dict, landmark_dict, image_name, measurements,dlv3)
            
            # Call the plot() method for each plotter
            line_annotator.plot()
            mask_drawer.plot()
            
        # This is optional, but you can return the modified image if needed
        return image


