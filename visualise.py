import numpy as np
import cv2
import random as rnd
from skimage.measure import find_contours

def draw_shapes(image, shapes, thickness):
    for i, shape in enumerate(shapes):
        shape[:, [0, 1]] = shape[:, [1, 0]]
        shape = np.array([shape], np.int32)
        color = (rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255))
        cv2.polylines(image, [shape], True, color, thickness)

    return image


def draw_dots(image, shapes, radius=5, thickness=3):
    for shape in shapes:
        x_centre = np.mean(shape[:, 1], axis=0)
        y_centre = np.mean(shape[:, 0], axis=0)
        cv2.circle(image, center=(int(y_centre), int(x_centre)), radius=radius, color=(0, 0, 255), thickness=thickness)

    return image


def draw_circles(image, masks):
    for i in range(masks.shape[2]):
        contours = find_contours(masks[:, :, i], level=i)
        for countour in contours:
            x_centre = int(np.mean(countour[:, 1], axis=0))
            y_centre = int(np.mean(countour[:, 0], axis=0))
            cv2.circle(image, (x_centre, y_centre), 20, (255, 40, 47), 2)

    return image