from visualise import draw_circles
from utils import filter_prediction
from mrcnn.extends.predictor import Predictor
from skimage import io
from preproccesing import create_image_slices, create_union_image
import cv2


def predict_small_image(path_to_weights, path_to_small_image, visialise=True):
    predictor = Predictor(path_to_weights)
    image_for_prediction = io.imread(path_to_small_image)

    filtering_image = cv2.bilateralFilter(image_for_prediction, 4, 20, 20)
    prediction = predictor.predict(filtering_image)[0]

    trees_in_prediction, new_prediction = filter_prediction(prediction)

    if visialise:
        img_with_shapes = draw_circles(image_for_prediction, new_prediction['masks'])
        return trees_in_prediction, img_with_shapes
    else:
        return trees_in_prediction, None


def predict_big_image(path_to_weights, path_to_big_image, size_of_slice=300, visialise=True):
    big_image = io.imread(path_to_big_image)
    slices = create_image_slices(big_image, size_of_slice, size_of_slice)

    new_slices_list = list()
    total_count_trees = 0
    predictor = Predictor(path_to_weights)

    for index, slice in enumerate(slices):
        filter_slice = cv2.bilateralFilter(slice.copy(), 2, 50, 50)
        prediction = predictor.predict(filter_slice)[0]
        trees_in_prediction, new_prediction = filter_prediction(prediction)
        total_count_trees = total_count_trees + trees_in_prediction

        if visialise:
            img_with_shapes = draw_circles(slice, new_prediction['masks'])
            new_slices_list.append(img_with_shapes)

    union_image = None
    if len(new_slices_list) > 0:
        union_image = create_union_image(new_slices_list, big_image.shape)

    return total_count_trees, union_image