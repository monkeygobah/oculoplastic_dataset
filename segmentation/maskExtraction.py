import numpy as np
from operator import itemgetter
from scipy.optimize import leastsq
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import cv2

class ExtractorTools:
    @staticmethod
    def circle_residuals(params, points):
        h, k, r = params
        x, y = points.T
        return (x - h)**2 + (y - k)**2 - r**2
    
    @staticmethod
    def fit_circle(points):
        # Initial guess: centroid of points as circle center & average distance to center as radius
        x, y = np.mean(points, axis=0)
        r = np.mean(np.sqrt((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2))
        params, _ = leastsq(ExtractorTools.circle_residuals, [x, y, r], args=(points,))
        return params  # Returns circle parameters: h, k, r

    # TODO probably can use Darvins better method here 
    @staticmethod
    # gets border points from 
    def get_border_points(mask):
        # Getting top and bottom points for each column
        top_points = []
        bottom_points = []
        
        for x in range(mask.shape[1]):
            column = mask[:, x]
            
            # Get the top and bottom non-zero points
            non_zeros = np.where(column == 1)[0]
            
            if len(non_zeros) > 0:
                top_points.append((x, non_zeros[0]))
                bottom_points.append((x, non_zeros[-1]))

        return top_points, bottom_points


    @staticmethod
    def filter_iris_points(iris_mask, sclera_mask, threshold=10):
        iris_top, iris_bottom = ExtractorTools.get_border_points(iris_mask)
        sclera_top, sclera_bottom = ExtractorTools.get_border_points(sclera_mask)


        # Convert points list to dictionary with x as key for easy lookup
        sclera_top_dict = {x:y for x,y in sclera_top}
        sclera_bottom_dict = {x:y for x,y in sclera_bottom}
        
        new_iris_top = []
        new_iris_bottom = []
        # Iterate over iris_top and check the corresponding sclera_top
        for x, y in iris_top:
            if x in sclera_top_dict and abs(y - sclera_top_dict[x]) > threshold:
                new_iris_top.append((x,y))
        
        # Iterate over iris_bottom and check the corresponding sclera_bottom
        for x, y in iris_bottom:
            if x in sclera_bottom_dict and abs(y - sclera_bottom_dict[x]) > threshold:
                new_iris_bottom.append((x,y))
        
        return new_iris_top, new_iris_bottom
        
    @staticmethod
    def visualize_boundaries(iris_mask, sclera_mask, filtered_top, filtered_bottom, circle_center=None, circle_diameter=None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(iris_mask, cmap='gray')
        
        iris_top, iris_bottom = ExtractorTools.get_border_points(iris_mask)
        sclera_top, sclera_bottom = ExtractorTools.get_border_points(sclera_mask)
        
        ax.scatter(*zip(*iris_top), c='yellow', s=2, label="Iris Top")
        ax.scatter(*zip(*iris_bottom), c='orange', s=2, label="Iris Bottom")
        ax.scatter(*zip(*sclera_top), c='blue', s=2, label="Sclera Top")
        ax.scatter(*zip(*sclera_bottom), c='cyan', s=2, label="Sclera Bottom")
        
        # Check if lists are empty before plotting
        if filtered_top:
            ax.scatter(*zip(*filtered_top), c='red', s=2, label="Filtered Iris Top")
        if filtered_bottom:
            ax.scatter(*zip(*filtered_bottom), c='magenta', s=2, label="Filtered Iris Bottom")
        
        # Plot the circle if center and diameter are provided
        if circle_center is not None and circle_diameter is not None:
            radius = circle_diameter / 2
            circle_patch = Circle(circle_center, radius, fill=False, color='green', linewidth=2, label='Fitted Circle')
            ax.add_patch(circle_patch)
        
        ax.legend()
  
        
    @staticmethod
    def get_lateral_iris_circle(iris_mask, sclera_mask, dim, method=1):
        # iris_mask = ExtractorTools.mask_coords_to_bin(iris_data, dim)
        # sclera_mask = ExtractorTools.mask_coords_to_bin(sclera_data, dim)

        #was .0196
        threshold = .0196 * dim[1]
        
        filtered_top, filtered_bottom = ExtractorTools.filter_iris_points(iris_mask, sclera_mask,threshold=threshold)
        
        if method == 1:
            try:
                if len(filtered_top)==0:
                    h, k, r = ExtractorTools.fit_circle(np.array(filtered_bottom))        
        
                elif len(filtered_bottom) == 0:
                    h, k, r = ExtractorTools.fit_circle(np.array(filtered_top))
                else:
                    filtered_iris = np.vstack((filtered_top, filtered_bottom))
                    h, k, r = ExtractorTools.fit_circle(filtered_iris)    
                    
                center = np.array([h,k])
                diameter = 2*r
            except:
                iris_top, iris_bottom = ExtractorTools.get_border_points(iris_mask)
                combined_iris = np.vstack((iris_top, iris_bottom))
                center, diameter = ExtractorTools.iris_circle_inscribed(combined_iris)    
                
        if method == 2:
            iris_top, iris_bottom = ExtractorTools.get_border_points(iris_mask)
            combined_iris = np.vstack((iris_top, iris_bottom))
            center, diameter = ExtractorTools.iris_circle_inscribed(combined_iris)

        # ExtractorTools.visualize_boundaries(iris_mask, sclera_mask, filtered_top, filtered_bottom, center, diameter)

        return center, diameter



    @staticmethod
    def iris_circle_inscribed(points):
        max_distance = 0
        point1, point2 = None, None
        
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                if distance > max_distance:
                    max_distance = distance
                    point1, point2 = points[i], points[j]
                    
        midpoint = [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]
        return midpoint, max_distance


    @staticmethod
    def find_closest_if_no_match(candidates, eyebrow_mask, reference_point, orientation):
        # Check if there are direct matches (candidates) on the same x-coordinate
        if len(candidates) > 0:
            if orientation == 'inferior':
                # Find the most inferior (maximum y-value) among the candidates
                selected_point = candidates[np.argmax(candidates)]
            elif orientation == 'superior':
                # Find the most superior (minimum y-value) among the candidates
                selected_point = candidates[np.argmin(candidates)]
            return (reference_point[0], selected_point)
        else:
            # No direct matches, find the closest point in the specified orientation
            # Compute distances from the reference point to all points in the eyebrow mask
            distances = np.sqrt((eyebrow_mask[:, 0] - reference_point[1])**2 + (eyebrow_mask[:, 1] - reference_point[0])**2)
            min_distance_idx = distances.argmin()
            target_point = eyebrow_mask[min_distance_idx]
            return (target_point[0], target_point[1])

    @staticmethod
    def cluster_masks(points, img_shape):
        """
        Get the largest cluster from a set of points.
        """
 
        # Ensure points is a numpy array
        points = np.array(points)

        # Label connected components
        labeled, num_features = label(points)
        
        # Find the label with the maximum count (excluding background which is label 0)
        if num_features > 0:
            largest_label = np.bincount(labeled.ravel())[1:].argmax() + 1

            # Create a mask for only this largest component
            largest_cluster_mask = (labeled == largest_label).astype(np.uint8)
            return largest_cluster_mask
        else:
            return np.array([])  
    
    @staticmethod
    def combine_masks(sclera_mask, iris_mask):
        # Ensure both masks are binary (0 or 1)
        sclera_mask_binary = np.where(sclera_mask > 0, 1, 0)
        iris_mask_binary = np.where(iris_mask > 0, 1, 0)
        
        # Combine the masks
        combined_mask = cv2.bitwise_or(sclera_mask_binary, iris_mask_binary)
        return combined_mask    
    
    @staticmethod
    def get_all_clusters(detections_dict, crop_img_rgb, gt = False):

        right_sclera_mask = ExtractorTools.cluster_masks(detections_dict['right_sclera'], (crop_img_rgb.size[1], crop_img_rgb.size[0]))
        left_sclera_mask = ExtractorTools.cluster_masks(detections_dict['left_sclera'], (crop_img_rgb.size[1], crop_img_rgb.size[0]))

        right_iris_mask = ExtractorTools.cluster_masks(detections_dict['right_iris'], (crop_img_rgb.size[1], crop_img_rgb.size[0]))
        left_iris_mask = ExtractorTools.cluster_masks(detections_dict['left_iris'], (crop_img_rgb.size[1], crop_img_rgb.size[0]))
        
        right_brow_mask = ExtractorTools.cluster_masks(detections_dict['right_brow'], (crop_img_rgb.size[1], crop_img_rgb.size[0]))
        left_brow_mask = ExtractorTools.cluster_masks(detections_dict['left_brow'], (crop_img_rgb.size[1], crop_img_rgb.size[0]))

        
        # # Combine sclera and iris masks for both right and left
        # right_sclera_mask = ExtractorTools.combine_masks(right_sclera_mask, right_iris_mask)
        # left_sclera_mask = ExtractorTools.combine_masks(left_sclera_mask, left_iris_mask)
   
        # right_pupil_mask = ExtractorTools.cluster_masks(detections_dict['right_pupil'], (crop_img_rgb.size[1], crop_img_rgb.size[0]))
        # left_pupil_mask = ExtractorTools.cluster_masks(detections_dict['left_pupil'], (crop_img_rgb.size[1], crop_img_rgb.size[0]))       
        
        return right_sclera_mask, left_sclera_mask, right_iris_mask, left_iris_mask, right_brow_mask, left_brow_mask #, right_pupil_mask, left_pupil_mask

class PupilIris:
    # TODO: make this work for both eyes
    def __init__(self):
        pass
    
    
    # TODO: also include smallest inscribed circle for iris and implement some form of comparison     
    # get iris center, superior/ inferior from center (same x coordinate),and iris diameter
    def get_iris_info(self, iris_mask, sclera_mask, img_dim,crop_image, method=1):  
        method=2
        centroid, diameter = ExtractorTools.get_lateral_iris_circle(iris_mask, sclera_mask, img_dim,method)
        
        # ExtractorTools.visualize_boundaries_and_circles( crop_image,iris_coordinates, sclera_mask, img_dim, combined_iris, combined_sclera)
   
        centroid = [round(centroid[0],0), round(centroid[1],0)]
        
        superior = [centroid[0], centroid[1]-(diameter/2)]
        inferior = [centroid[0], centroid[1]+(diameter/2)]

        return np.array(centroid), diameter, np.array(superior), np.array(inferior) #[::-1], inferior[::-1]



# TODO this needs some serious love- two get_medial_lateral points methods and sclera points
class Sclera:
    def __init__(self):
        pass

    # def fit_polynomial(border_points, degree=4):
    #     """
    #     Fit a polynomial to the border points.
    
    #     :param border_points: Array of border points (x, y).
    #     :param degree: Degree of the polynomial.
    #     :return: Polynomial coefficients.
    #     """
    #     border_points = np.array(border_points)
    #     return np.polyfit(border_points[:, 0], border_points[:, 1], degree)


    # def find_vertical_points(poly_coeffs, iris_center_x, mask_shape):
    #     """
    #     Find the vertical points on the polynomial curve from the iris center.
    
    #     :param poly_coeffs: Coefficients of the polynomial.
    #     :param iris_center_x: x-coordinate of the iris center.
    #     :param mask_shape: Shape of the mask (to ensure points are within the mask).
    #     :return: Superior and inferior points (y-coordinates).
    #     """
    #     poly = np.poly1d(poly_coeffs)
    #     superior_point = max(0, int(poly(iris_center_x)))  # Ensure within the image
    #     inferior_point = min(mask_shape[0] - 1, int(poly(iris_center_x)))  # Ensure within the image
    #     return superior_point, inferior_point


    # @staticmethod
    # def plot_border_points(mask, top_points, bottom_points, superior_point=None, inferior_point=None, title='Sclera Border Points'):
    #     plt.figure(figsize=(8, 8))
    #     plt.title(title)
    
    #     # Convert points to arrays for easy plotting
    #     top_points_array = np.array(top_points)
    #     bottom_points_array = np.array(bottom_points)
    
    #     # Plot top and bottom points
    #     plt.scatter(top_points_array[:, 0], top_points_array[:, 1], color='blue', label='Top Border Points')
    #     plt.scatter(bottom_points_array[:, 0], bottom_points_array[:, 1], color='red', label='Bottom Border Points')
    
    #     # If superior and inferior points are provided, plot them
    #     if superior_point is not None:
    #         plt.scatter(superior_point[0], superior_point[1], color='green', marker='^', label='Superior Point')
    #     if inferior_point is not None:
    #         plt.scatter(inferior_point[0], inferior_point[0], color='orange', marker='v', label='Inferior Point')
    
    #     plt.xlabel('X-axis')
    #     plt.ylabel('Y-axis')
    #     plt.legend()
    #     plt.show()


    # @staticmethod
    # def find_point_on_polynomial(poly_coeffs, x_value):
    #     poly = np.poly1d(poly_coeffs)
    #     y_val =  poly(x_value)
    #     return np.array([x_value, y_val])
        

 
    def plot_sclera_masks_with_key_points(l_sclera_mask, r_sclera_mask, l_key_points, r_key_points, current_index):
        """
        Plots left and right sclera masks with key points.

        Args:
        - l_sclera_mask: 2D array for left sclera mask.
        - r_sclera_mask: 2D array for right sclera mask.
        - l_key_points: Tuple of key points (sup, inf, med, lat) for the left sclera.
        - r_key_points: Tuple of key points (sup, inf, med, lat) for the right sclera.
        """

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot left sclera mask
        axs[0].imshow(l_sclera_mask, cmap='gray')
        axs[0].set_title('Left Sclera Mask')
        # Plot right sclera mask
        axs[1].imshow(r_sclera_mask, cmap='gray')
        axs[1].set_title('Right Sclera Mask')

        # Define colors for different key points for clarity
        colors = {'sup': 'red', 'inf': 'blue', 'med': 'green', 'lat': 'yellow'}
        point_labels = {'sup': 'Superior', 'inf': 'Inferior', 'med': 'Medial', 'lat': 'Lateral'}

        # Helper function to plot key points
        def plot_points(ax, key_points, color_map):
            for label, point in zip(color_map.keys(), key_points):
                if point is not None:
                    ax.scatter(point[0], point[1], c=color_map[label], label=point_labels[label], s=40, edgecolor='black')
        
        # Plot key points for left and right sclera
        plot_points(axs[0], l_key_points, colors)
        plot_points(axs[1], r_key_points, colors)

        # Legend
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4)

        plt.tight_layout()
        plt.savefig(f'{current_index}_sclera_points.jpg')
        plt.close
        
    def get_sclera_key_points(self, sclera_mask, iris_centroid_x, direction='r'):
        # Identify all y (rows) and x (columns) coordinates where the mask is non-zero
        
        column = sclera_mask[:, int(iris_centroid_x)]
        y_indices = np.where(column > 0)[0]
    
        # The superior point is the minimum y index, and the inferior point is the maximum y index
        superior_point_y = y_indices.min()
        inferior_point_y = y_indices.max()
        
        # Return points as (x, y) pairs
        superior_point = (iris_centroid_x, superior_point_y)
        inferior_point = (iris_centroid_x, inferior_point_y)
        
        y_coords, x_coords = np.where(sclera_mask > 0)
        
        if direction == 'r':
            lateral_point = (x_coords.min(), y_coords[x_coords.argmin()])  # Min x
            medial_point = (x_coords.max(), y_coords[x_coords.argmax()])  # Max x
        else:
            medial_point = (x_coords.min(), y_coords[x_coords.argmin()])  # Min x
            lateral_point = (x_coords.max(), y_coords[x_coords.argmax()])  # Max x  

        return superior_point, inferior_point, medial_point, lateral_point

    def sclera_points(self, l_sclera, r_sclera, l_iris_centroid, r_iris_centroid, dim, image):

        l_sup, l_inf, l_med, l_lat =  Sclera.get_sclera_key_points(self, l_sclera, l_iris_centroid[0], direction='l')
        r_sup, r_inf, r_med, r_lat =  Sclera.get_sclera_key_points(self, r_sclera, r_iris_centroid[0])

        # Sclera.plot_sclera_masks_with_key_points(l_sclera, r_sclera,
        #                                 (l_sup, l_inf, l_med, l_lat), 
        #                                 (r_sup, r_inf, r_med, r_lat),current_idx)


        return l_sup, l_inf, r_sup, r_inf, r_med, r_lat, l_med, l_lat


class Brows:
    def __init__(self):
        pass

    @staticmethod
    def plot_points_with_labels(reference_points, eyebrow_points):
        # Create a new figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot reference points with specific colors
        for key, point in reference_points.items():
            ax.plot(point[0], point[1], 'o', label=key)
        
        # Plot eyebrow points with a different marker
        eyebrow_labels = ['l_lat_eyebrow', 'l_center_eyebrow', 'l_medial_eyebrow', 'r_lat_eyebrow', 'r_center_eyebrow', 'r_medial_eyebrow',
                        'sup_l_lat_eyebrow', 'sup_l_center_eyebrow', 'sup_l_medial_eyebrow', 'sup_r_lat_eyebrow', 'sup_r_center_eyebrow', 'sup_r_medial_eyebrow']
        for point, label in zip(eyebrow_points, eyebrow_labels):
            ax.plot(point[0], point[1], 'x', label=label)
        
        ax.invert_yaxis()

        ax.axis('equal')
        ax.legend()
        
        # Show grid for better visualization
        ax.grid(True)
        
        plt.show()

    
    def _find_eyebrow_point(self, eyebrow, reference_x, reference_point, dir='inf'):
        
        candidates = np.where(eyebrow[:, int(reference_point[0])] == 1)[0]
        if dir == 'inf':
            return ExtractorTools.find_closest_if_no_match(candidates, eyebrow, reference_point, 'inferior')
        elif dir == 'sup':
            return ExtractorTools.find_closest_if_no_match(candidates, eyebrow, reference_point, 'superior')
        
    def get_eyebrow_points(self, l_eyebrow, r_eyebrow, l_lateral_canthus, r_lateral_canthus, l_medial_canthus, r_medial_canthus, r_iris_center, l_iris_center):    

        l_lat_eyebrow = self._find_eyebrow_point(l_eyebrow, l_lateral_canthus[0], l_lateral_canthus)
        l_medial_eyebrow = self._find_eyebrow_point(l_eyebrow, l_medial_canthus[0], l_medial_canthus)
        l_center_eyebrow = self._find_eyebrow_point(l_eyebrow, int(l_iris_center[0]), l_iris_center)
        
        r_lat_eyebrow = self._find_eyebrow_point(r_eyebrow, r_lateral_canthus[0], r_lateral_canthus)
        r_medial_eyebrow = self._find_eyebrow_point(r_eyebrow, r_medial_canthus[0], r_medial_canthus)
        r_center_eyebrow = self._find_eyebrow_point(r_eyebrow, int(r_iris_center[0]), r_iris_center)
        
        sup_l_lat_eyebrow = self._find_eyebrow_point(l_eyebrow, l_lateral_canthus[0], l_lateral_canthus, dir='sup')
        sup_l_medial_eyebrow = self._find_eyebrow_point(l_eyebrow, l_medial_canthus[0], l_medial_canthus, dir='sup')
        sup_l_center_eyebrow = self._find_eyebrow_point(l_eyebrow, int(l_iris_center[0]), l_iris_center, dir='sup')
        
        sup_r_lat_eyebrow = self._find_eyebrow_point(r_eyebrow, r_lateral_canthus[0], r_lateral_canthus, dir='sup')
        sup_r_medial_eyebrow = self._find_eyebrow_point(r_eyebrow, r_medial_canthus[0], r_medial_canthus, dir='sup')
        sup_r_center_eyebrow = self._find_eyebrow_point(r_eyebrow, int(r_iris_center[0]), r_iris_center, dir='sup')
        
        reference_points = {
            'l_lateral_canthus': l_lateral_canthus, 'r_lateral_canthus': r_lateral_canthus,
            'l_medial_canthus': l_medial_canthus, 'r_medial_canthus': r_medial_canthus,
            'l_iris_center': l_iris_center, 'r_iris_center': r_iris_center
        }
        
        eyebrow_points = (l_lat_eyebrow, l_center_eyebrow, l_medial_eyebrow, r_lat_eyebrow, r_center_eyebrow, r_medial_eyebrow,\
                            sup_l_lat_eyebrow, sup_l_center_eyebrow, sup_l_medial_eyebrow, sup_r_lat_eyebrow, sup_r_center_eyebrow, sup_r_medial_eyebrow
                            )

        # Brows.plot_points_with_labels(reference_points, eyebrow_points)

        return l_lat_eyebrow, l_center_eyebrow, l_medial_eyebrow, r_lat_eyebrow, r_center_eyebrow, r_medial_eyebrow, \
            sup_l_lat_eyebrow, sup_l_medial_eyebrow, sup_l_center_eyebrow, sup_r_lat_eyebrow, sup_r_medial_eyebrow, sup_r_center_eyebrow

class EyeFeatureExtractor:
    def __init__(self,detections_dict,crop_image, gt=False):
        self.pupil_iris_getter = PupilIris()
        self.sclera_getter = Sclera()
        self.brow_getter = Brows()
        self.landmarks = {}
        self.image_size = (crop_image.size[1], crop_image.size[0])
        self.detections_dict = detections_dict
        self.crop_image = crop_image
        # self.current_idx = current_idx
        self.gt = gt

    def _cluster(self):
        self.right_sclera_mask, self.left_sclera_mask, self.right_iris_mask, self.left_iris_mask, self.right_brow_mask, self.left_brow_mask\
                = ExtractorTools.get_all_clusters(self.detections_dict, self.crop_image, self.gt)
    
    def _iris(self):
        right_iris_centroid, right_iris_diameter, right_iris_superior, right_iris_inferior \
            = self.pupil_iris_getter.get_iris_info(self.right_iris_mask, self.right_sclera_mask, self.image_size, self.crop_image, method=1)
        
        left_iris_centroid, left_iris_diameter, left_iris_superior, left_iris_inferior \
            = self.pupil_iris_getter.get_iris_info(self.left_iris_mask, self.left_sclera_mask, self.image_size,self.crop_image, method=1)

        if abs(left_iris_diameter/right_iris_diameter) < .975 :
            self.landmarks['iris_discrepancy'] = 'iris_discrepancy'
            
            # Try to recompute using iris_circle_inscribed
            right_iris_centroid, right_iris_diameter, right_iris_superior, right_iris_inferior \
                = self.pupil_iris_getter.get_iris_info(self.right_iris_mask, self.right_sclera_mask, self.image_size, self.crop_image, method=2)
            
            left_iris_centroid, left_iris_diameter, left_iris_superior, left_iris_inferior \
                = self.pupil_iris_getter.get_iris_info(self.left_iris_mask, self.left_sclera_mask, self.image_size,self.crop_image, method=2)

        else:
            self.landmarks['iris_discrepancy'] = ''

        self.landmarks['right_iris_centroid'] = right_iris_centroid
        self.landmarks['right_iris_diameter'] = right_iris_diameter
        self.landmarks['right_iris_superior'] = right_iris_superior
        self.landmarks['right_iris_inferior'] = right_iris_inferior

        self.landmarks['left_iris_centroid'] = left_iris_centroid
        self.landmarks['left_iris_diameter'] = left_iris_diameter
        self.landmarks['left_iris_superior'] = left_iris_superior
        self.landmarks['left_iris_inferior'] = left_iris_inferior

    # def _pupil(self):
    #     right_pupil_centroid, right_pupil_diameter, r_medial_pupil, r_lateral_pupil, r_inferior_pupil\
    #         = self.pupil_iris_getter.get_pupil_info(self.right_pupil_mask, 'right')
    #     left_pupil_centroid, left_pupil_diameter, l_medial_pupil, l_lateral_pupil, l_inferior_pupil \
    #         = self.pupil_iris_getter.get_pupil_info(self.left_pupil_mask, 'left')       
        
    #     self.landmarks['right_pupil_centroid'] = right_pupil_centroid
    #     self.landmarks['right_pupil_diameter'] = right_pupil_diameter
    #     self.landmarks['r_medial_pupil'] = r_medial_pupil
    #     self.landmarks['r_lateral_pupil'] = r_lateral_pupil
    #     self.landmarks['r_inferior_pupil'] = r_inferior_pupil
        
    #     self.landmarks['left_pupil_centroid'] = left_pupil_centroid
    #     self.landmarks['left_pupil_diameter'] = left_pupil_diameter
    #     self.landmarks['l_medial_pupil'] = l_medial_pupil
    #     self.landmarks['l_lateral_pupil'] = l_lateral_pupil
    #     self.landmarks['l_inferior_pupil'] = l_inferior_pupil

    def _sclera(self):
        # Your            
        left_sclera_superior, left_sclera_inferior, right_sclera_superior, right_sclera_inferior, \
        right_medial_canthus, right_lateral_canthus, left_medial_canthus,left_lateral_canthus\
            = self.sclera_getter.sclera_points(self.left_sclera_mask,self.right_sclera_mask,self.landmarks['left_iris_centroid'],self.landmarks['right_iris_centroid'],self.image_size, self.crop_image )

        self.landmarks['left_sclera_superior'] = left_sclera_superior
        self.landmarks['left_sclera_inferior'] = left_sclera_inferior
        self.landmarks['right_sclera_superior'] = right_sclera_superior
        self.landmarks['right_sclera_inferior'] = right_sclera_inferior
        
        self.landmarks['right_medial_canthus'] = right_medial_canthus
        self.landmarks['right_lateral_canthus'] = right_lateral_canthus
        self.landmarks['left_medial_canthus'] = left_medial_canthus
        self.landmarks['left_lateral_canthus'] = left_lateral_canthus
        
    def _eyebrow(self):

        l_lat_eyebrow, l_center_eyebrow, l_medial_eyebrow, r_lat_eyebrow, r_center_eyebrow, r_medial_eyebrow, \
            sup_l_lat_eyebrow, sup_l_medial_eyebrow, sup_l_center_eyebrow, sup_r_lat_eyebrow, sup_r_medial_eyebrow, sup_r_center_eyebrow  = \
            self.brow_getter.get_eyebrow_points(self.left_brow_mask, self.right_brow_mask, \
                                                self.landmarks['left_lateral_canthus'], self.landmarks['right_lateral_canthus'],\
                                                self.landmarks['left_medial_canthus'], self.landmarks['right_medial_canthus'], self.landmarks['right_iris_centroid'], self.landmarks['left_iris_centroid']) 
        
        self.landmarks['l_lat_eyebrow'] = l_lat_eyebrow
        self.landmarks['l_center_eyebrow'] = l_center_eyebrow
        self.landmarks['l_medial_eyebrow'] = l_medial_eyebrow
        
        self.landmarks['r_lat_eyebrow'] = r_lat_eyebrow
        self.landmarks['r_center_eyebrow'] = r_center_eyebrow
        self.landmarks['r_medial_eyebrow'] = r_medial_eyebrow

        self.landmarks['sup_l_lat_eyebrow'] = sup_l_lat_eyebrow
        self.landmarks['sup_l_center_eyebrow'] = sup_l_center_eyebrow
        self.landmarks['sup_l_medial_eyebrow'] = sup_l_medial_eyebrow
        
        self.landmarks['sup_r_lat_eyebrow'] = sup_r_lat_eyebrow
        self.landmarks['sup_r_center_eyebrow'] = sup_r_center_eyebrow
        self.landmarks['sup_r_medial_eyebrow'] = sup_r_medial_eyebrow



    def extract_features(self):
        self._cluster()
        
        # try:
        self._iris()
            # self.landmarks['iris_error'] = False

        # except IndexError:
        #     iris_error = 'no_iris_detected'
        #     self.landmarks['right_iris_centroid'] = iris_error
        #     self.landmarks['right_iris_diameter'] = iris_error
        #     self.landmarks['right_iris_superior'] = iris_error
        #     self.landmarks['right_iris_inferior'] = iris_error

        #     self.landmarks['left_iris_centroid'] = iris_error
        #     self.landmarks['left_iris_diameter'] = iris_error
        #     self.landmarks['left_iris_superior'] = iris_error
        #     self.landmarks['left_iris_inferior'] = iris_error

        #     self.landmarks['iris_error'] = True

        
        # self._pupil()
        # try:
        self._sclera()
        # self.landmarks['sclera_error'] = False

        # except IndexError:
        #     sclera_error = 0
        #     self.landmarks['left_sclera_superior'] = sclera_error
        #     self.landmarks['left_sclera_inferior'] = sclera_error
        #     self.landmarks['right_sclera_superior'] = sclera_error
        #     self.landmarks['right_sclera_inferior'] = sclera_error
            
        #     self.landmarks['right_medial_canthus'] = sclera_error
        #     self.landmarks['right_lateral_canthus'] = sclera_error
        #     self.landmarks['left_medial_canthus'] = sclera_error
        #     self.landmarks['left_lateral_canthus'] = sclera_error

        #     self.landmarks['sclera_error'] = True
            
            
        
        try:
            self._eyebrow()
            self.landmarks['brow_error'] = False

        except IndexError:
            pass
            # brow_error = np.array([0,0])
            # self.landmarks['l_lat_eyebrow'] = brow_error
            # self.landmarks['l_center_eyebrow'] = brow_error
            # self.landmarks['l_medial_eyebrow'] = brow_error
            
            # self.landmarks['r_lat_eyebrow'] = brow_error
            # self.landmarks['r_center_eyebrow'] = brow_error
            # self.landmarks['r_medial_eyebrow'] = brow_error

            # self.landmarks['sup_l_lat_eyebrow'] = brow_error
            # self.landmarks['sup_l_center_eyebrow'] = brow_error
            # self.landmarks['sup_l_medial_eyebrow'] = brow_error
            
            # self.landmarks['sup_r_lat_eyebrow'] = brow_error
            # self.landmarks['sup_r_center_eyebrow'] = brow_error
            # self.landmarks['sup_r_medial_eyebrow'] = brow_error

            # self.landmarks['brow_error'] = True
        
        
        return self.landmarks





