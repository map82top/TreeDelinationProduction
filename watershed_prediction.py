from watershed import count_trees_on_image
from visualise import draw_shapes, draw_dots
from skimage import io
from preproccesing import create_image_slices, create_union_image


def predict_small_image(image_path, visualise=True, min_area=100, d=8, sigmaColor=40, sigmaSpace=40, footprintSize=(43, 43)):

    """Predict and draw crown of trees on an image

        :param path_to_weights: str
            full path to a file with neural network weights (with extend .h5)
        :param path_to_small_image: str
            full path to an image for create a prediction
        :param visialise: bool
            if is true then function will been draw crown of trees on got image
        :param min_area: int
            min area of crown
            crown not recognized as a tree if its crown is less then this value
        :param d: int
            value of parameter d for bilateralFilter
        :param sigmaColor: int
            value of parameter sigmaColor for bilateralFilter
        :param sigmaSpace: int
            value of parameter sigmaSpace for bilateralFilter
        :return:
            if visialise parameter is true then function return count of prediction trees and an image with drew predicted crown of trees
            else function return only count of prediction tree
    """

    big_image = io.imread(image_path)
    count_trees, shapes = count_trees_on_image(big_image, min_area, d, sigmaColor, sigmaSpace, footprintSize)

    if visualise:
        img_with_shapes = draw_shapes(big_image, shapes, 1)
        img_with_shapes = draw_dots(img_with_shapes, shapes)
        return count_trees, img_with_shapes

    else:
        return count_trees, None


def predict_big_image(image_path, size_of_slice=512, visualise=True, min_area=100, d=8, sigmaColor=40, sigmaSpace=40, footprintSize=(43, 43)):

    """Predict and draw crown of trees on a big image

        :param path_to_weights: str
            full path to a file with neural network weights (with extend .h5)
        :param path_to_big_image: str
            full path to an image for create a prediction
        :param size_of_slice: int
           image size (width and height) into which a big image will be split
        :param visialise: bool
            if is true then function will been draw crown of trees on got image
        :param min_area: int
            min area of crown
            crown not recognized as a tree if its crown is less then this value
        :param d: int
            value of parameter d for bilateralFilter
        :param sigmaColor: int
            value of parameter sigmaColor for bilateralFilter
        :param sigmaSpace: int
            value of parameter sigmaSpace for bilateralFilter
        :param footprintSize: tuple of int
            size of space on which need to search local maximum
        :return:
            if visialise parameter is true then function return count of prediction trees and an image with drew predicted crown of trees
            else function return only count of prediction tree
    """

    big_image = io.imread(image_path)
    slices = create_image_slices(big_image, size_of_slice, size_of_slice)

    new_slices_list = list()
    total_count_trees = 0

    for number, slice_img in enumerate(slices):
        count_trees, shapes = count_trees_on_image(slice_img, 120, min_area, d, sigmaColor, sigmaSpace, footprintSize)
        total_count_trees = total_count_trees + count_trees

        if visualise:
            img_with_shapes = draw_shapes(slice_img, shapes, 1)
            img_with_shapes = draw_dots(img_with_shapes, shapes)

            new_slices_list.append(img_with_shapes)

    union_image = None
    if len(new_slices_list) > 0:
        union_image = create_union_image(new_slices_list, big_image.shape)

    return total_count_trees, union_image
