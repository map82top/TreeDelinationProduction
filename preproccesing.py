import math
import numpy as np


def create_image_slices(big_image, split_img_width, split_img_height):
    """Split a big image on many small image given size

    :param big_image: numpy array
        loaded into RAM a big image
    :param split_img_width: int
        width image on which a big image will be split
    :param split_img_height: int
        height image on which a big image will be split
    :return:
       list of slice from a big image
    """
    height, width, channels = big_image.shape
    img_slices = list()

    count_columns = int(math.ceil(width / split_img_width))
    count_rows = int(math.ceil(height / split_img_height))
    for row in range(0, count_rows):
        for column in range(0, count_columns):
            start_split_width = column * split_img_width
            start_split_height = row * split_img_height
            end_split_width = (column + 1) * split_img_width
            end_split_height = (row + 1) * split_img_height

            if end_split_width >= width:
                end_split_width = width

            if end_split_height >= height:
                end_split_height = height

            w_difference = end_split_width - start_split_width
            h_difference = end_split_height - start_split_height

            split_img = np.zeros((split_img_width, split_img_height, channels), dtype=np.uint8)
            split_img[:h_difference, :w_difference, :] = big_image[start_split_height:end_split_height, start_split_width:end_split_width, :]
            img_slices.append(split_img)

    return img_slices


def create_union_image(list_with_slices, union_image_size):
    """Union a list from small images into a big image

    :param list_with_slices: list of numpy array
        content images which need to union
    :param union_image_size: numpy 1D array
        array shape of big image (width, height, weight)
    :return:
        union image
    """
    if len(list_with_slices) > 0:
        slice = list_with_slices[0].shape

        width_slice = slice[1]
        height_slice = slice[0]
        union_image_width = union_image_size[1]
        union_image_height = union_image_size[0]

        count_columns = int(math.ceil(union_image_width / width_slice))
        count_rows = int(math.ceil(union_image_height / height_slice))

        union_image = np.zeros(union_image_size, dtype=np.uint8)

        for row in range(0, count_rows):
            for column in range(0, count_columns):
                slice_image = list_with_slices[row * count_columns + column]

                start_split_width = column * width_slice
                start_split_height = row * height_slice
                end_split_width = (column + 1) * width_slice
                end_split_height = (row + 1) * height_slice

                if end_split_width >= union_image_width:
                    end_split_width = union_image_width

                if end_split_height >= union_image_height:
                    end_split_height = union_image_height

                w_difference = end_split_width - start_split_width
                h_difference = end_split_height - start_split_height

                union_image[start_split_height:end_split_height, start_split_width: end_split_width, :] = slice_image[:h_difference, :w_difference, :]

        return union_image
    else:
        raise Exception("List of slices is empty")
    return
