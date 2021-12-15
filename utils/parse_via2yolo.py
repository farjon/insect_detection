import os
import cv2
import json
import shutil
import argparse
import numpy as np

# this script assumes that the annotations are arranged in a via style
# but without a train-val-test split!
# there is only a single annotations file and the images folder

def via_label_to_yolo_style(via_style, im_width, im_height):
    '''
    convert via labels style to yolo style: bbox[x_min, y_min, w, h] -> x_center, y_center, w, h
    :param via_style: a dictionary with x, y, w, h, normalized
    :return: yolo style labels in a list
    '''
    x_center_norm = (via_style['shape_attributes']['x'] + int(via_style['shape_attributes']['width']/2))/im_width
    y_center_norm = (via_style['shape_attributes']['y'] + int(via_style['shape_attributes']['height']/2))/im_height
    w_norm = via_style['shape_attributes']['width'] / im_width
    h_norm = via_style['shape_attributes']['height'] /im_height
    yolo_style = [x_center_norm, y_center_norm, w_norm, h_norm]
    return yolo_style


def parse_args():
    parser = argparse.ArgumentParser(description='Annotations parser from VOC to COCO')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-s', '--stage', type=str, default='1', help='for understanding this go to the ReadMe file')

    args = parser.parse_args()
    return args


def main(args):
    np.random.seed(10)
    args.ROOT_DIR = os.path.join(args.ROOT_DIR, 'stage_' + args.stage)
    # set up paths
    path_to_via_file = os.path.join(args.ROOT_DIR, 'annotations', 'insects_combined_annotations.json')
    path_to_yolo = os.path.join(args.ROOT_DIR, 'yolo_no_test')

    with open(path_to_via_file, 'r') as annotations:
        via_json = json.load(annotations)

    images_name = np.array(via_json['_via_image_id_list'])
    random_ids = np.random.permutation(images_name.size)
    train_names = images_name[random_ids[:int(len(images_name)*0.8)]]
    val_names = images_name[random_ids[int(len(images_name)*0.8):]]


    sets = ['train', 'val']
    for current_set, set_names in zip(sets, [train_names, val_names]):
        path_to_set_yolo = os.path.join(path_to_yolo, current_set)
        os.makedirs(path_to_set_yolo, exist_ok=True)
        set_label_path = os.path.join(path_to_yolo, current_set + '.txt')
        set_label_txt = open(set_label_path, 'w+')

        # run over images, collect annotations
        for img_name in via_json['_via_img_metadata']:
            if img_name not in set_names:
                continue
            img_via = via_json['_via_img_metadata'][img_name]
            img_name = img_via['filename']
            open(set_label_path, 'a').write(current_set + '\\' + img_name +'\n')
            path_to_im_src = os.path.join(args.ROOT_DIR, 'images', img_name)
            img_cv = cv2.imread(path_to_im_src)
            img_height, img_width, _ = img_cv.shape
            path_to_im_dest = os.path.join(path_to_set_yolo, img_name)
            shutil.copy(path_to_im_src, path_to_im_dest)
            img_label_path = os.path.join(path_to_set_yolo, img_name.split('.')[0] + '.txt')
            img_label_txt = open(img_label_path, 'w+')
            annotations_for_im = img_via['regions']
            for annot in annotations_for_im:
                yolo_style_annot = via_label_to_yolo_style(annot, img_width, img_height)
                yolo_style_annot.insert(0, '0')
                open(img_label_path, 'a').write(' '.join([str(i) for i in yolo_style_annot]) + '\n')
            img_label_txt.close()
        set_label_txt.close()

if __name__ == '__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\insect_detection\\Data'
    main(args)


