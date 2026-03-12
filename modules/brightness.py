import numpy as np
import cv2


def align_brightness(images_list):
    if not images_list:
        return images_list
    brightness_values = [np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]) for img in images_list]
    median_brightness = np.median(brightness_values)
    ref_img = images_list[np.argmin(np.abs(np.array(brightness_values) - median_brightness))]
    ref_mean = np.mean(cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB)[:, :, 0])
    aligned_images = []
    for img in images_list:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        l_adjusted = np.clip(l_channel + (ref_mean - np.mean(l_channel)), 0, 255).astype(np.uint8)
        lab[:, :, 0] = l_adjusted
        aligned_images.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
    return aligned_images
