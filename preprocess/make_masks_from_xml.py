import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt

def extract_rle_from_xml(xml_file_path, image_name, structure, max_id=1000):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    # Iterate through all 'image' elements in the XML
    for image in root.findall('image'):
        image_id = int(image.get('id'))
        if image.get('name') == image_name and image_id <= max_id:
            width = int(image.get('width'))
            height = int(image.get('height'))
            # If the image name matches, search for the structure
            for mask in image.findall('mask'):
                if mask.get('label') == structure:
                    mask_top = int(mask.get('top'))
                    mask_width = int(mask.get('width'))
                    mask_left = int(mask.get('left'))
                    return mask.get('rle'), mask_left, mask_top, mask_width, height, width
    return None


def cvat_rle_to_binary_image_mask(rle, left, top, width, img_h, img_w):
    # convert CVAT tight object RLE to COCO-style whole image mask
    rle = [int(x) for x in rle.split(',')]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    value = 0
    offset = 0
    for rle_count in rle:
        while rle_count > 0:
            y, x = divmod(offset, width)
            mask[y + top][x + left] = value
            rle_count -= 1
            offset += 1
        value = 1 - value
    return mask


def create_image_list(xml_file_path, max_id, output_file):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    with open(output_file, 'w') as f:
        for image in root.findall('image'):
            image_id = int(image.get('id'))
            if image_id <= max_id:
                f.write(image.get('name') + '\n')



if __name__ == '__main__':

    # paths and structs
    xml_path = 'celeb_full_annot_1_1000.xml'
    structures = ['l_iris', 'r_iris', 'l_brow', 'r_brow', 'l_sclera', 'r_sclera', 'l_caruncle', 'r_caruncle', 'left_lid', 'right_lid']
    out_path = 'full_celeb_annotations'
    # data = 'CELEBA-HQ-CROP'
    data = 'small_img'
    utils.make_folder(out_path)

    MAX_ID = 1000  
    no_img = []

    image_file = 'image_list.txt'

    create_image_list(xml_path, MAX_ID, image_file)

    no_img = []

    with open(image_file, 'r') as f:
        image_list = f.read().splitlines()

    # read in image name from list
    for image_name in image_list:
        print(image_name)
        for struct in structures:

            mask_data = extract_rle_from_xml(xml_path, image_name, struct)
            if mask_data:
                mask = cvat_rle_to_binary_image_mask(*mask_data)


                #prep file name
                letter = struct[0]
                anat = struct.split('_')[1]

                # save each struct mask
                name = f'{image_name[:-4]}_{letter}_{anat}.png'
                cv2.imwrite(os.path.join(out_path, name), mask)


    print(f'images that were not found: {no_img}')
