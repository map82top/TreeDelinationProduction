import numpy as np
import cv2
from skimage import io
from imgaug import augmenters as iaa
from mrcnn import utils
import mrcnn.model as modellib
from skimage import draw
from mrcnn.extends.config import TrainConfig
import mrcnn.visualize as visualize

from os.path import isfile, join, exists
import pickle


class ShapesDataSet(utils.Dataset):
    """Class for load data set"""
    def load_trees(self, dataset_dir, start_number, end_number, width, height):
        if not exists(dataset_dir):
            raise Exception("Dataset directory not exist")
        # Add classes
        self.add_class("trees", 1, "trees")

        # Add images
        image_name = 'image_'
        image_ext = '.png'
        shape_ext = '.shp'

        for i in range(start_number, end_number):
            path_to_img = join(dataset_dir, image_name + str(i) + image_ext)
            path_tp_shapes = join(dataset_dir, image_name + str(i) + shape_ext)
            shapes = pickle.load(open(path_tp_shapes, 'rb'))
            self.add_image("trees", image_id=i, path=path_to_img, width=width, height=height, shapes=shapes)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        path_to_img = info["path"]
        image = io.imread(path_to_img)
        return cv2.bilateralFilter(image, 2, 50, 50)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "trees":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], 1], dtype=np.uint8)
        for i, shape in enumerate(shapes):
            # rr, cc = draw.circle(shape[0], shape[1], shape[2])
            rr, cc = draw.polygon(shape[:, 0], shape[:, 1])
            rr = np.where(rr > 127, 127, rr)
            cc = np.where(cc > 127, 127, cc)

            mask[rr, cc, 0] = 1

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


def train_mrcnn(path_to_dataset,
                path_to_logs,
                path_to_base_weights,
                path_to_save_weights,
                start_of_train_dataset,
                end_index_of_train_dataset,
                start_of_validate_dataset,
                end_of_validate_dataset,
                train_head_epochs,
                train_all_epochs):

    """Train Mask R-CNN neural network

    :param path_to_dataset: str
        full path to folder when need to load train data set
    :param path_to_logs: str
        full path to directory when need to save train logs
    :param path_to_base_weights: str
        full path to file with neural network weights
    :param path_to_save_weights: str
        full path to folder when need to save weights trained network
    :param start_of_train_dataset: int
        start index of train sample from loaded data set
    :param end_index_of_train_dataset: int
        end index of train sample from loaded data set
    :param start_of_validate_dataset: int
        start index of validate sample from loaded data set
    :param end_of_validate_dataset: int
        end index of validate sample from loaded data set
    :param train_head_epochs: int
        count of epochs for train head of neural network
    :param train_all_epochs: int
        count of epochs for train all layers of neural network
    :return:
        nothing
    """

    config = TrainConfig()
    config.display()

    # Training dataset
    dataset_train = ShapesDataSet()
    dataset_train.load_trees(path_to_dataset, start_of_train_dataset, end_index_of_train_dataset, 128, 128)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ShapesDataSet()
    dataset_val.load_trees(path_to_dataset, start_of_validate_dataset, end_of_validate_dataset,  128, 128)
    dataset_val.prepare()

    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 2)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names, 1)

    augmentation = iaa.SomeOf((0, 2), [iaa.Fliplr(0.5), iaa.Flipud(0.5),
                                       iaa.OneOf([iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270)]), iaa.Multiply((0.5, 1.5))])

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=path_to_logs)

    model.load_weights(path_to_base_weights, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=train_head_epochs, augmentation=augmentation, layers='heads')
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=train_all_epochs, augmentation=augmentation, layers='all')
    model.keras_model.save_weights(join(path_to_save_weights, "tree_crown_prediction_mrcnn_weights.h5"))



