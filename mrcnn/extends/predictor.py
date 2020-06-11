from mrcnn.extends.config import InferenceConfig
from mrcnn import model as modellib


class Predictor:
    def __init__(self, path_to_weights):
        self.config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode='inference', config=self.config, model_dir="../../logs")
        self.model.load_weights(path_to_weights, by_name=True)
        print("Model is loaded!")

    def predict(self, image):
        return self.model.detect([image], verbose=1)