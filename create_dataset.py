import os
from os import listdir
from os.path import isfile, join, exists
import pickle
from skimage import io
from preproccesing import create_image_slices
from watershed import count_trees_on_image, reduce_shape
import cv2
from visualise import draw_dots, draw_shapes
import matplotlib.pyplot as plt


def create(path_to_resources_folder, path_to_save_folder):
    """Generate training data set for Mask R-CNN

    :param path_to_resources_folder: str
        full path to a folder containing image files for creating training data set
    :param path_to_save_folder: str
       full path to a folder where should be save a created data set
    :return: nothing
    """

    base_img_name = "image_"
    base_img_ext = ".png"
    base_shape_ext = ".shp"
    base_img_shape_name = "image_with_shape_"

    if not exists(path_to_resources_folder):
        raise Exception("Primary dataset not found")

    if not exists(path_to_save_folder):
        os.mkdir(path_to_save_folder)

    path_to_images = [img for img in listdir(path_to_resources_folder) if isfile(join(path_to_resources_folder, img))]

    counter_images = 0
    for index, name_img in enumerate(path_to_images):
        print("Work with image ", name_img)
        resource_image = io.imread(join(path_to_resources_folder, name_img))
        big_image_slices = create_image_slices(resource_image, 400, 400)

        for number, slice_img in enumerate(big_image_slices):
            count_trees, shapes = count_trees_on_image(slice_img.copy(), 100)

            #resize
            scale_percent = slice_img.shape[0] / 128.0
            resize_img = cv2.resize(slice_img, (128, 128), interpolation=cv2.INTER_AREA)
            img_path = join(path_to_save_folder, base_img_name + str(counter_images) + base_img_ext)
            io.imsave(img_path, resize_img)

            processed_shapes = list()
            for i in range(len(shapes)):
                shapes[i] = shapes[i] / scale_percent
                simple_shape = reduce_shape(shapes[i], 5)
                processed_shapes.append(simple_shape)

            shape_path = join(path_to_save_folder, base_img_name + str(counter_images) + base_shape_ext)
            ser_pickle = pickle.dump(processed_shapes, open(shape_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            counter_images = counter_images + 1

        print("DataSet have ", counter_images, " images")