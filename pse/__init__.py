import subprocess
import os
import numpy as np
import cv2
from .pse import pse_cpp

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
#     raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))


def pse(kernals, min_area=5):
    """PSE algorithm
    Args:
        kernals: 6 kernals, from big to small.
        min_area: min area to be preserved.
    Returns:
        pred: pred seg map.
        label_values: label values of pred seg map.
    """
    kernal_num = len(kernals)
    if not kernal_num:
        return np.array([]), []
    kernals = np.array(kernals)
    label_num, label = cv2.connectedComponents(kernals[kernal_num - 1].astype(np.uint8), connectivity=4)
    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0
            continue
        label_values.append(label_idx)
    pred = pse_cpp(label, kernals, c=6)

    return pred, label_values
