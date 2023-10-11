import os
import urllib

from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
from torchvision import transforms


from semantic_image_segmentation.models.model_loader import BaseModelLoader
from semantic_image_segmentation.conf import MODELS_PATH


class DeepLabv3ModelLoader(BaseModelLoader):
    def __init__(self, base_configs, lazy_setup=False):
        self.model = None
        self.class_labels = self._load_class_labels()
        super(DeepLabv3ModelLoader, self).__init__(base_configs, lazy_setup)

    def _load_class_labels(self):
        class_labels = []
        with open(os.path.join(MODELS_PATH, 'pascalvoc2012.txt'), 'r') as f:
            class_labels = [line.replace('\n', '').split(',')[-1] for line in f.readlines()]
        return class_labels

    def setup(self):
        # 'deeplabv3_resnet50'
        model_name = self.base_configs['model_name']
        # detection_threshold = self.base_configs['detection_threshold']
        if torch.cuda.is_available():
            gpu_fraction = self.base_configs['gpu_fraction']
            torch.cuda.set_per_process_memory_fraction(gpu_fraction, 0)

        # torch_home = self.base_configs['torch_home']
        self.model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
        # or any of these variants
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
        self.model.eval()

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            self.model.to('cuda')

        if self.base_configs.get('hot_start', False) is True:
            print('Running hot start...')
            self._hot_start(225, 225)
            print('Finished hot start...')

    def _hot_start(self, width, height, bgr_color=(0, 0, 0)):
        # Create black blank image
        image = np.zeros((height, width, 3), np.uint8)
        # Fill image with color
        image[:] = bgr_color
        return self.predict(image)


    def preprocess(self, input_image):
        rgb_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        # input_image = input_image.convert("RGB")
        preprocess_trfsm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess_trfsm(rgb_input_image)

    def post_processing(self, np_predicts):
        predicted_classes = {}
        for row in np_predicts:
            for val in row:
                class_idx = int(val)
                predicted_classes[class_idx] = self.class_labels[class_idx]

        return {
            'data': np_predicts,
            'labels': predicted_classes
        }


    def predict(self, input_image):
        input_tensor = self.preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        np_predicts = output_predictions.numpy().astype("uint8")
        return self.post_processing(np_predicts)

def run(model_loader, url, filename):

    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    input_image = np.array(Image.open(filename))
    bgr_input_image =  cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    output = model_loader.predict(bgr_input_image)


    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    print(output['labels'])

    # r = Image.fromarray(cv2.cvtColor(bgr_input_image, cv2.COLOR_BGR2RGB))
    # # plot the semantic segmentation predictions of 21 classes in each color
    # r = Image.fromarray(output['data']).resize(input_image.shape[:2][::-1])
    # r.putpalette(colors)
    # plt.imshow(r)
    # plt.show()

if __name__ == '__main__':
    model_loader = DeepLabv3ModelLoader(
        base_configs = {
            'model_name': 'deeplabv3_resnet50',
            'hot_start': True
        }
    )
    urls = [
        "https://github.com/pytorch/hub/raw/master/images/deeplab1.png",
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
    ]

    for url in urls:
        filename = url.split('/')[-1]
        run(model_loader, url, filename)

    # input_image = np.array(Image.open(filename))
    # bgr_input_image =  cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    # output = model_loader.predict(bgr_input_image)


    # # create a color pallette, selecting a color for each class
    # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    # colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    # colors = (colors % 255).numpy().astype("uint8")

    # print(output['labels'])

    # # plot the semantic segmentation predictions of 21 classes in each color
    # r = Image.fromarray(output['data']).resize(input_image.shape[:2][::-1])
    # # r = Image.fromarray(cv2.cvtColor(bgr_input_image, cv2.COLOR_BGR2RGB))
    # r.putpalette(colors)
    # plt.imshow(r)
    # plt.show()