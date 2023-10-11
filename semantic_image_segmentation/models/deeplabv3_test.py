import os

import torch

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
import urllib
import matplotlib.pyplot as plt

from semantic_image_segmentation.conf import MODELS_PATH


class_labels = []

with open(os.path.join(MODELS_PATH, 'pascalvoc2012.txt'), 'r') as f:
    class_labels = [line.replace('\n', '').split(',')[-1] for line in f.readlines()]

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
model.eval()



url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)



input_image = Image.open(filename)
input_image = input_image.convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

classes_idx_set = set()
np_predicts = output_predictions.cpu().numpy().astype("uint8")
for row in np_predicts:
    for val in row:
        class_idx = int(val)
        classes_idx_set.add(class_idx)

predict_class_labels = set()
for class_idx in classes_idx_set:
    predict_class_labels.add(class_labels[class_idx])
print(list(predict_class_labels))


masked_classes_ids = [15, 17]
# np_predicts = output_predictions.byte().cpu().numpy()
# np_predicts[(np_predicts == 15) | (np_predicts==12)] = 255
for class_id in masked_classes_ids:
    np_predicts[np_predicts == class_id] = 255
# np_predicts[np_predicts == 12] = 255
np_predicts[np_predicts != 255] = 0


# plot the semantic segmentation predictions of 21 classes in each color
semantic_mask = Image.fromarray(np_predicts).resize(input_image.size)
import ipdb; ipdb.set_trace()
# semantic_mask.putpalette(colors)
semantic_mask.save('new.jpg')


plt.imshow(semantic_mask)
plt.show()
# import time
# time.sleep(10)