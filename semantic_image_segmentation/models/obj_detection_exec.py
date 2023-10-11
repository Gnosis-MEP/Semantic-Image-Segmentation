import os
import urllib
import json

from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
from torchvision import transforms


from semantic_image_segmentation.models.model_loader import BaseModelLoader
from semantic_image_segmentation.conf import MODELS_PATH


class Yolov5ModelLoader(BaseModelLoader):
    def __init__(self, base_configs, lazy_setup=False):
        self.model = None
        self.class_labels = self._load_class_labels()
        super(Yolov5ModelLoader, self).__init__(base_configs, lazy_setup)

    def _load_class_labels(self):
        class_labels = []
        with open(os.path.join(MODELS_PATH, 'pascalvoc2012.txt'), 'r') as f:
            class_labels = [line.replace('\n', '').split(',')[-1] for line in f.readlines()]
        return class_labels

    def setup(self):
        model_name = self.base_configs['model_name']
        if torch.cuda.is_available():
            gpu_fraction = self.base_configs['gpu_fraction']
            torch.cuda.set_per_process_memory_fraction(gpu_fraction, 0)

        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.eval()
        self.class_labels = self.model.names

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
        # 1280
        return rgb_input_image

    # def normalized_bbox(self, bboxes, origin_height, origin_width, new_height, new_width):

    #     # rescale the coordinates to the original image
    #     bboxes[:, 0] *= (origin_width / float(new_width))
    #     bboxes[:, 2] *= (origin_width / float(new_width))
    #     bboxes[:, 1] *= (origin_height / float(new_height))
    #     bboxes[:, 3] *= (origin_height / float(new_height))

    #     return bboxes

    def post_processing(self, np_predicts):


    #         obj = {
    #             'label': label,
    #             'bounding_box': [int(i) for i in bbox],
    #             'confidence': float(score)
    #         }
    #         output.append(obj)
    # return {'data': output}
        # boxes = self.normalized_bbox(boxes, origin_height, origin_width, new_height, new_width)
        # output.ims.shape
        # (1026, 1282, 3) original image shape
        predicted_classes = {}
        for row in np_predicts:
            xmin, ymin, xmax, ymax, conf, class_idx = row
            label = self.class_labels[class_idx]
            obj = {
                'label': label,
                'bounding_box': [float(i) for i in [xmin, ymin, xmax, ymax]],
                'confidence': float(conf)
            }
            class_objs = predicted_classes.setdefault(label, [])
            class_objs.append(obj)
        return {
            'data': predicted_classes
        }


    def predict(self, input_image):
        preprocess_img = self.preprocess(input_image)

        with torch.no_grad():
            output = self.model([preprocess_img])
            output_predictions = output.xyxyn[0]

        np_predicts = output_predictions.numpy().astype("float32")
        # import ipdb;ipdb.set_trace()
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

    print(json.dumps(output, indent=4))

    # r = Image.fromarray(cv2.cvtColor(bgr_input_image, cv2.COLOR_BGR2RGB))
    # # plot the semantic segmentation predictions of 21 classes in each color
    # r = Image.fromarray(output['data']).resize(input_image.shape[:2][::-1])
    # r.putpalette(colors)
    # plt.imshow(r)
    # plt.show()

if __name__ == '__main__':
    model_loader = Yolov5ModelLoader(
        base_configs = {
            # 'model_name': 'YOLOv5l6',
            'model_name': 'yolov5s',
            'hot_start': False
        }
    )
    urls = [
        "https://github.com/pytorch/hub/raw/master/images/deeplab1.png",
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
    ]

    for url in urls:
        filename = url.split('/')[-1]
        run(model_loader, url, filename)

    input_image = np.array(Image.open(filename))
    bgr_input_image =  cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    output = model_loader.predict(bgr_input_image)


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