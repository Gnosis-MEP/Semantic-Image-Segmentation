import os

import torch

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
import urllib
# import matplotlib.pyplot as plt

from semantic_image_segmentation.conf import MODELS_PATH


class BaseModelLoader(object):
    def __init__(self, base_configs, lazy_setup=False):
        super(BaseModelLoader, self).__init__()
        self.base_configs = base_configs
        if not lazy_setup:
            self.setup()

    def setup(self):
        # model_name = self.base_configs['model_name']
        # detection_threshold = self.base_configs['detection_threshold']
        # gpu_fraction = self.base_configs['gpu_fraction']
        # torch.cuda.set_per_process_memory_fraction(0.5, 0)
        pass