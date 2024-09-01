#!/usr/bin/env python
import json
import sys
import glob
import os

import torch

import cv2
import numpy as np

from PIL import Image
from torchvision import transforms

from semantic_image_segmentation.conf import (
    MODELS_PATH,
    MASK_OUTPUTS_PATH,
    SAMPLES_OUTPUTS_PATH,
    MORPH_KERNEL_SIZE,
    FRAME_SAMPLES_SIZE
)




class MaskCreator():
    def __init__(self, input_dir, video_sub_seg_id, masked_class_labels):
        self.input_dir = input_dir
        self.video_sub_seg_id = video_sub_seg_id
        self.masked_class_labels = masked_class_labels
        self.mask_output_dir =  os.path.join(MASK_OUTPUTS_PATH, self.video_sub_seg_id, '_'.join(masked_class_labels))
        self.samples_output_path = os.path.join(SAMPLES_OUTPUTS_PATH, self.video_sub_seg_id, '_'.join(masked_class_labels))

        self.morph_kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),np.uint8)

        self.class_labels = []
        with open(os.path.join(MODELS_PATH, 'pascalvoc2012.txt'), 'r') as f:
            self.class_labels = [line.replace('\n', '').split(',')[-1] for line in f.readlines()]
        self.masked_classes_ids = [
            cid
            for cid, class_label in enumerate(self.class_labels)
              if class_label.lower() in self.masked_class_labels
        ]

        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
        # or any of these variants
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

    def preprocess(self, input_image):
        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transf(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        return input_batch

    def predict(self, input_image):
        input_batch = self.preprocess(input_image)
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # classes_idx_set = set()
        # for row in output_predictions:
        #     for val in row:
        #         class_idx = int(val)
        #         classes_idx_set.add(class_idx)

        # predict_class_labels = set()
        # for class_idx in classes_idx_set:
        #     predict_class_labels.add(self.class_labels[class_idx])
        # print(list(predict_class_labels), classes_idx_set)
        return output_predictions.cpu().numpy().astype("uint8")

    def post_processing(self, input_image, output_predictions):

        # self.masked_classes_ids = [15, 17]
        for class_id in self.masked_classes_ids:
            output_predictions[output_predictions == class_id] = 255
        output_predictions[output_predictions != 255] = 0

        output_predictions = cv2.dilate(output_predictions, self.morph_kernel, iterations=1)
        # output_predictions = cv2.erode(output_predictions, np.ones((5 ,5),np.uint8), iterations=1)
        # output_predictions = cv2.morphologyEx(output_predictions, cv2.MORPH_CLOSE, self.morph_kernel)

        semantic_mask = Image.fromarray(output_predictions).resize(input_image.size)

        colors_total = semantic_mask.getcolors()

        has_oi = False
        for color_tuple in colors_total:
            if color_tuple[1] == 255:
                has_oi = True
                break

        # if len(colors_total) == 1:
        #     self.samples_output_path
        #     if colors_total[0][1] == 0:
            # return None
        return semantic_mask, has_oi

    def create_mask(self, image_path):
        image_name = os.path.basename(image_path)
        mask_image_name = f'mask_{image_name}'
        mask_image_path = os.path.join(self.mask_output_dir, mask_image_name)
        input_image = Image.open(image_path)
        output_predictions = self.predict(input_image)

        mask_image, has_oi = self.post_processing(input_image, output_predictions)

        if has_oi:
            original_oi_sample = os.path.join(self.samples_output_path, image_name)
            input_image.save(original_oi_sample)

        if mask_image is not None:
            mask_image.save(mask_image_path)
        return mask_image_path

    def run(self):
        input_images_path = os.path.join(self.input_dir, '*.png')
        if not os.path.exists(self.mask_output_dir):
            os.makedirs(self.mask_output_dir, exist_ok=True)
        if not os.path.exists(self.samples_output_path):
            os.makedirs(self.samples_output_path, exist_ok=True)

        mask_path_list = []
        sorted_images = sorted(glob.glob(input_images_path), key=lambda s: int(os.path.basename(s).split('frame_')[1].split('.png')[0]))
        for image_path in sorted_images[:FRAME_SAMPLES_SIZE]:
            mask_path = self.create_mask(image_path)
            if mask_path is not None:
                mask_path_list.append(mask_path)

        return mask_path_list



if __name__ == '__main__':
    input_dir = sys.argv[1]
    video_sub_seg_id = sys.argv[2]
    masked_class_labels = sys.argv[3].lower().split(',')
    mask_creator = MaskCreator(input_dir, video_sub_seg_id, masked_class_labels)
    res = mask_creator.run()
    print(json.dumps(res, indent=4))
