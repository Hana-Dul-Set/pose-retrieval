import numpy as np
import cv2

from typing import Tuple

def black_image(size):
    return np.zeros((size[0], size[1], 3), dtype = np.uint8)

def predict_image_border_after_resize(image, target_size):
    target_width, target_height = target_size

    original_height, original_width = image.shape[:2]
    
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    expanding = target_width > original_width or target_height > original_height

    if expanding:
        if target_aspect_ratio > original_aspect_ratio:
            new_width = target_width
            new_height = int(target_width / original_aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * original_aspect_ratio)
    else:
        # Shrink the image and add black borders
        if target_aspect_ratio > original_aspect_ratio:
            new_height = target_height
            new_width = int(target_height * original_aspect_ratio)
        else:
            new_width = target_width
            new_height = int(target_width / original_aspect_ratio)

    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left
    return (left, top, right, bottom)

def resize_with_black_borders(image, target_size):
    target_width, target_height = target_size

    original_height, original_width = image.shape[:2]
    
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    expanding = target_width > original_width or target_height > original_height

    if expanding:
        if target_aspect_ratio < original_aspect_ratio:
            new_width = target_width
            new_height = int(target_width / original_aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * original_aspect_ratio)
        resized_image = cv2.resize(image, (new_width, new_height))
    else:
        # Shrink the image and add black borders
        if target_aspect_ratio > original_aspect_ratio:
            new_height = target_height
            new_width = int(target_height * original_aspect_ratio)
        else:
            new_width = target_width
            new_height = int(target_width / original_aspect_ratio)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left

    resized_image_with_borders = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return resized_image_with_borders
