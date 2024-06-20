import numpy as np
import os
import math

class AnatomyTools:
    @staticmethod
    def distance(point1, point2, stack=False):
        if stack:
            return np.linalg.norm(np.array([point1[0], point2[1]]) - np.array(point2)) 
        else:
            return np.linalg.norm(np.array(point1) - np.array(point2))

    @staticmethod
    def conditional_distance(cond, pointA, pointB, pointC=None):
        if cond:
            return AnatomyTools.distance(pointA, pointB), 0
        else:
            return AnatomyTools.distance(pointA, pointC), AnatomyTools.distance(pointB, pointC)

    #: TODO probably can just use np dot here? cant remember why i did this 
    @staticmethod
    # Get dot product of two vectors. Could use something else here probably?
    def dot(vA, vB):
        return vA[0]*vB[0]+vA[1]*vB[1]

    @staticmethod
    # get angle between two lines for canthal tilt
    def ang(lineA, lineB):
        # Get nicer vector form
        vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
        vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
        # Get dot prod
        dot_prod =AnatomyTools.dot(vA, vB)
        # Get magnitudes
        magA = AnatomyTools.dot(vA, vA)**0.5
        magB = AnatomyTools.dot(vB, vB)**0.5
        # Get cosine value
        cos_ = dot_prod/magA/magB
        # Get angle in radians and then convert to degrees
        angle = math.acos(dot_prod/magB/magA)
        return angle

class EyeMetrics:
    def __init__(self, landmarks, mask_dict):
        self.landmarks = landmarks
        self.mask_dict = mask_dict
        self.measurements= {}
    
    def horiz_fissure(self):
        left_horiz_pf =AnatomyTools.distance(self.landmarks['left_medial_canthus'], self.landmarks['left_lateral_canthus'], True)
        right_horiz_pf = AnatomyTools.distance(self.landmarks['right_medial_canthus'], self.landmarks['right_lateral_canthus'], True)
        self.measurements['left_horiz_fissure'] = left_horiz_pf
        self.measurements['right_horiz_fissure'] = right_horiz_pf

    def brow_heights(self):
        left_mc_bh = AnatomyTools.distance(self.landmarks['left_medial_canthus'], self.landmarks['l_medial_eyebrow'])
        left_central_bh = AnatomyTools.distance(self.landmarks['left_iris_centroid'], self.landmarks['l_center_eyebrow'])
        left_lc_bh = AnatomyTools.distance(self.landmarks['left_lateral_canthus'], self.landmarks['l_lat_eyebrow'])

        right_mc_bh = AnatomyTools.distance(self.landmarks['right_medial_canthus'], self.landmarks['r_medial_eyebrow'])
        right_central_bh = AnatomyTools.distance(self.landmarks['right_iris_centroid'], self.landmarks['r_center_eyebrow'])
        right_lc_bh = AnatomyTools.distance(self.landmarks['right_lateral_canthus'], self.landmarks['r_lat_eyebrow'])
        
        sup_left_mc_bh = AnatomyTools.distance(self.landmarks['left_medial_canthus'], self.landmarks['sup_l_medial_eyebrow'])
        sup_left_central_bh = AnatomyTools.distance(self.landmarks['left_iris_centroid'], self.landmarks['sup_l_center_eyebrow'])
        sup_left_lc_bh = AnatomyTools.distance(self.landmarks['left_lateral_canthus'], self.landmarks['sup_l_lat_eyebrow'])

        sup_right_mc_bh = AnatomyTools.distance(self.landmarks['right_medial_canthus'], self.landmarks['sup_r_medial_eyebrow'])
        sup_right_central_bh = AnatomyTools.distance(self.landmarks['right_iris_centroid'], self.landmarks['sup_r_center_eyebrow'])
        sup_right_lc_bh = AnatomyTools.distance(self.landmarks['right_lateral_canthus'], self.landmarks['sup_r_lat_eyebrow'])
        
            
        self.measurements['sup_left_medial_bh'] = sup_left_mc_bh
        self.measurements['sup_left_central_bh'] = sup_left_central_bh
        self.measurements['sup_left_lateral_bh'] = sup_left_lc_bh

        self.measurements['sup_right_medial_bh'] = sup_right_mc_bh
        self.measurements['sup_right_central_bh'] = sup_right_central_bh
        self.measurements['sup_right_lateral_bh'] = sup_right_lc_bh
        
        self.measurements['left_medial_bh'] = left_mc_bh
        self.measurements['left_central_bh'] = left_central_bh
        self.measurements['left_lateral_bh'] = left_lc_bh

        self.measurements['right_medial_bh'] = right_mc_bh
        self.measurements['right_central_bh'] = right_central_bh
        self.measurements['right_lateral_bh'] = right_lc_bh


    def scleral_show(self):
        l_mrd_1, left_SSS = AnatomyTools.conditional_distance((self.landmarks['left_sclera_superior'][1] > self.landmarks['left_iris_superior'][1]), self.landmarks['left_iris_centroid'], self.landmarks['left_sclera_superior'], self.landmarks['left_iris_superior'])
        l_mrd_2, left_ISS = AnatomyTools.conditional_distance((self.landmarks['left_sclera_inferior'][1] < self.landmarks['left_iris_inferior'][1]), self.landmarks['left_iris_centroid'], self.landmarks['left_sclera_inferior'], self.landmarks['left_iris_inferior'])
        l_mrd_1 = l_mrd_1 + left_SSS
        l_mrd_2 = l_mrd_2 + left_ISS
        r_mrd_1, right_SSS = AnatomyTools.conditional_distance((self.landmarks['right_sclera_superior'][1] > self.landmarks['right_iris_superior'][1]) ,self.landmarks['right_iris_centroid'], self.landmarks['right_sclera_superior'], self.landmarks['right_iris_superior'])
        r_mrd_2, right_ISS = AnatomyTools.conditional_distance((self.landmarks['right_sclera_inferior'][1] < self.landmarks['right_iris_inferior'][1]), self.landmarks['right_iris_centroid'], self.landmarks['right_sclera_inferior'], self.landmarks['right_iris_inferior'])
        r_mrd_1 = r_mrd_1 + right_SSS
        r_mrd_2 = r_mrd_2 + right_ISS
        
        self.measurements['left_mrd_1'] = l_mrd_1
        self.measurements['left_SSS'] = left_SSS
        
        self.measurements['left_mrd_2'] = l_mrd_2
        self.measurements['left_ISS'] = left_ISS
        
        self.measurements['right_mrd_1'] = r_mrd_1
        self.measurements['right_SSS'] = right_SSS
        
        self.measurements['right_mrd_2'] = r_mrd_2
        self.measurements['right_ISS'] = right_ISS
        
        r_vpf =AnatomyTools.distance(self.landmarks['right_sclera_superior'], self.landmarks['right_sclera_inferior'])
        l_vpf = AnatomyTools.distance(self.landmarks['left_sclera_superior'], self.landmarks['left_sclera_inferior'])
        
        
        self.measurements['right_vert_pf'] = r_vpf
        self.measurements['left_vert_pf'] = l_vpf
                
        # self.measurements['right_vert_pf'] = r_mrd_2 + r_mrd_1
        # self.measurements['left_vert_pf'] = l_mrd_2 + l_mrd_1
    
    def canthal_height(self):
        rmc = self.landmarks['right_medial_canthus']
        lmc = self.landmarks['left_medial_canthus']
        rlc = self.landmarks['right_lateral_canthus']
        llc = self.landmarks['left_lateral_canthus']

        r_canthal_height =  np.linalg.norm(np.array(rlc[0], rmc[1]), rlc)
        l_canthal_height = np.linalg.norm(np.array(llc[0], lmc[1]), llc)


        self.measurements['r_canthal_height'] = r_canthal_height
        self.measurements['l_canthal_height'] = l_canthal_height
                                
    def icd(self):
        icd = AnatomyTools.distance(self.landmarks['left_medial_canthus'], self.landmarks['right_medial_canthus'], True)
        ipd = AnatomyTools.distance(self.landmarks['left_iris_centroid'], self.landmarks['right_iris_centroid'], True)
        ocd = AnatomyTools.distance(self.landmarks['left_lateral_canthus'], self.landmarks['right_lateral_canthus'], True)

        self.measurements['icd'] = icd
        self.measurements['ipd'] = ipd
        self.measurements['ocd'] = ocd
         
    def canthal_tilt(self):
        # "detect canthal tilt using angle between various lines of interest"
        #### do left eye first
        ## define line from medial canthus to lateral canthus using two xy points
        lmc_llc = [self.landmarks['left_medial_canthus'], self.landmarks['left_lateral_canthus']]

        ## define line from medial canthus to midline (use nasion x point) using two xy points
        point_1 = [self.landmarks['midline'][0],self.landmarks['midline'][1]]
        mid_lmc = [[point_1[0], self.landmarks['left_medial_canthus'][1]], self.landmarks['left_medial_canthus']]

        # use ang function to get angle of two lines in radians and then convert to degrees
        lct_rad = AnatomyTools.ang(lmc_llc, mid_lmc)
        
        lct_deg = math.degrees(lct_rad)
        
        #### do right eye now
        ## define line from medial canthus to lateral canthus using two xy points
        rmc_rlc = [self.landmarks['right_medial_canthus'], self.landmarks['right_lateral_canthus']]

        ## define line from medial canthus to midline (use apex of nose x point) using two xy points
        mid_rmc = [self.landmarks['right_medial_canthus'], [point_1[0], self.landmarks['right_medial_canthus'][1]]]

        # use ang function to get angle of two lines in radians and then convert to degrees
        rct_rad = AnatomyTools.ang(mid_rmc, rmc_rlc)
        rct_deg = math.degrees(rct_rad)
        #subtract to get acute angle
        rct_deg = 180-rct_deg
        
        self.measurements['left_canthal_tilt'] = lct_deg
        self.measurements['right_canthal_tilt'] = rct_deg
    
    
    def vert_dystop(self):
        l_mc = [self.landmarks['midline'][0], self.landmarks['left_iris_centroid'][1]]
        ### define xy coord of R medial canthus to midline
        r_mc = [self.landmarks['midline'][0], self.landmarks['right_iris_centroid'][1]]

        vert_dystop = AnatomyTools.distance(l_mc, r_mc)
        self.measurements['vert_dystopia'] = vert_dystop
        self.measurements['left_vd_plot_point'] = l_mc
        self.measurements['right_vd_plot_point'] = r_mc
    
    def scleral_area(self):
        right_iris_area = np.sum(self.mask_dict['right_iris'])
        right_sclera_area = np.sum(self.mask_dict['right_sclera'])

        left_iris_area = np.sum(self.mask_dict['left_iris'])
        left_sclera_area = np.sum(self.mask_dict['left_sclera'])
        
        right_area = right_sclera_area / right_iris_area if right_iris_area > 0 else 0
        left_area = left_sclera_area / left_iris_area if left_iris_area > 0 else 0

        self.measurements['right_scleral_area'] = right_area
        self.measurements['left_scleral_area'] = left_area
        
    def run(self, midline=False):
        self.horiz_fissure()
        try:
            self.brow_heights()
        except KeyError:
            print('eyebrow pass meauring')
            pass
        self.scleral_show()
        self.canthal_height()
        self.icd()
        if midline:
            self.canthal_tilt()
            self.vert_dystop()
        self.scleral_area()

        return self.measurements


