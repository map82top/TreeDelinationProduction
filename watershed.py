import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import morphology
from scipy import ndimage
from skimage.filters import rank
import cv2
from skimage.morphology import rectangle
from skimage.measure import find_contours


def count_trees_on_image(image, min_area=100):
    """Find crowns of trees with help algorithm watershed

    :param image: numpy array
        image on which need search crowns of trees
    :param min_area: int
        min area of crown
        crown not recognized as a tree if its crown is less then this value
    :return:
        count detected crowns of trees and shapes of crowns as list of numpy arrays
    """
    image = image.copy()

    filtering_image = cv2.bilateralFilter(image, 8, 40, 40)

    if filtering_image.ndim == 3:
        nir_image = filtering_image[:, :, 0]
    else:
        raise Exception("One dimension image")

    selem = rectangle(100, 100)
    local_otsu = rank.otsu(nir_image, selem)
    binary_image = nir_image > local_otsu

    distance = ndimage.distance_transform_edt(binary_image)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((43, 43)), labels=binary_image)
    markers = morphology.label(local_maxi)
    markers[~binary_image.astype(bool)] = -1

    shapes_trees = list()
    labels_ws = watershed(-distance, markers, mask=binary_image)

    for i in range(np.max(labels_ws) - 1, -1, -1):
        local_shapes = find_contours(labels_ws, i)
        area_shapes = dict()
        for index, local_shape in enumerate(local_shapes):
            area_of_shape = calculate_area_of_label(local_shape[:, 1], local_shape[:, 0])
            if area_of_shape >= min_area:
                area_shapes[area_of_shape] = local_shape

        if len(area_shapes.keys()) > 0:
            max_area = max(area_shapes.keys())
            shapes_trees.append(area_shapes[max_area])

        labels_ws = np.where(labels_ws > i, 0, labels_ws)

    return len(shapes_trees), shapes_trees


def calculate_area_of_label(x, y):
    """Find area of tree crown

    :param x: numpy 1D array
        x points of crown shape
    :param y: numpy 1D array
        y points of crown shape
    :return: float
        area of tree crown
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def reduce_shape(shape, indent):
    """Reduce shape using received indent

    :param shape: numpy array
        shape for decrease
    :param indent: int
        intent for decrease a received shape
    :return: numpy array
        new shape
    """

    x_centre = np.mean(shape[:, 1], axis=0)
    y_centre = np.mean(shape[:, 0], axis=0)

    new_shape = np.zeros(shape.shape, dtype=np.int32)

    for i in range(shape.shape[0]):
        point = shape[i, :]
        x_point = point[1]
        if x_point < x_centre:
            x_point = x_point + indent
        else:
            x_point = x_point - indent

        y_point = point[0]
        if y_point < y_centre:
            y_point = y_point + indent
        else:
            y_point = y_point - indent

        new_shape[i, :] = np.array([y_point, x_point])

    return new_shape

