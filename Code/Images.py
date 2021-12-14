# Read The Data
import numpy as np
import cv2
import os

classes = {"cl": 0, "ra": 1, 'sh': 2, 'su': 3}


def import_and_resize_all(path, desired_dimensions):
    img_r_list = []
    targets =[]
    for file in os.listdir(path):
        try:
            img = cv2.imread(os.path.join(path, file))
            img = img[:, :, ::-1]
            img_r = cv2.resize(img, desired_dimensions)
            img_r_list.append(img_r)
            targets.append(classes[file[0:2]])
        except Exception as e:
            print('Could not Load: ', file)
    return img_r_list, targets
