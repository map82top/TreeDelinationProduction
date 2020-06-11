from mrcnn.config import Config
import numpy as np


class TrainConfig(Config):
    NUM_CLASSES = 1 + 1
    NAME = 'trees'
    BATCH_SIZE = 10
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    STEPS_PER_EPOCH = 4000
    VALIDATION_STEPS = 500
    BACKBONE = "resnet50"
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    IMAGE_CHANNEL_COUNT = 3
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    MAX_GT_INSTANCES = 150
    DETECTION_MAX_INSTANCES = 150

    USE_MINI_MASK = False
    TRAIN_ROIS_PER_IMAGE = 150
    RPN_TRAIN_ANCHORS_PER_IMAGE = 150
    IMAGE_RESIZE_MODE = "square"

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }


class InferenceConfig(TrainConfig):
    NUM_CLASSES = 1 + 1
    NAME = 'trees'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BACKBONE = "resnet50"
    USE_MINI_MASK = False
    IMAGE_CHANNEL_COUNT = 3
    RPN_NMS_THRESHOLD = 0.9
    MEAN_PIXEL = np.array([105, 236, 189])
    DETECTION_NMS_THRESHOLD = 0.3
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    MAX_GT_INSTANCES = 220
    DETECTION_MAX_INSTANCES = 220
    TRAIN_ROIS_PER_IMAGE = 220
    RPN_TRAIN_ANCHORS_PER_IMAGE = 220
