import numpy as np
import cv2
import random as rnd
from skimage.measure import find_contours


def draw_shapes(image, shapes, thickness):
    """Draw crowns

    :param image: numpy array
        image on which need to draw crowns
    :param shapes: list of numpy array
        crowns which need to draw
    :param thickness: int
        thickness of line border of crown
    :return:
        image with drew shapes
    """

    for i, shape in enumerate(shapes):
        shape[:, [0, 1]] = shape[:, [1, 0]]
        shape = np.array([shape], np.int32)
        color = (rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255))
        cv2.polylines(image, [shape], True, color, thickness)

    return image


def draw_dots(image, shapes, radius=5, thickness=3):
    """Draw dots on centre crown

    :param image: numpy array
        image on which need to draw point on centre crowns
    :param shapes: list of numpy array
        crowns for which need to draw point on centre
    :param radius: int
         radius of dots for drawing
    :param thickness: int
        thickness of line border of dot
    :return:
        image with drew dots
    """

    for shape in shapes:
        x_centre = np.mean(shape[:, 1], axis=0)
        y_centre = np.mean(shape[:, 0], axis=0)
        cv2.circle(image, center=(int(y_centre), int(x_centre)), radius=radius, color=(0, 0, 255), thickness=thickness)

    return image


def draw_circles(image, masks):
    """Draw circles around found crowns

    Used only neural network prediction
    :param image: numpy array
        image on which need to draw circles around crowns
    :param masks: numpy array
        crowns for which need to draw circles around crowns
    :return:
        image with drew circles
    """

    for i in range(masks.shape[2]):
        contours = find_contours(masks[:, :, i], level=i)
        for contour in contours:
            x_centre = int(np.mean(contour[:, 1], axis=0))
            y_centre = int(np.mean(contour[:, 0], axis=0))
            cv2.circle(image, (x_centre, y_centre), 20, (255, 40, 47), 2)

    return image
