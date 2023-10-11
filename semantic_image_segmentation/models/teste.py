from PIL import Image

import numpy as np
import torch
import matplotlib.pyplot as plt


# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()
# Images
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images


image = np.zeros((250, 250, 3), np.uint8)
# Fill image with color
image[:] = (0, 0, 0)

# return self.predict(image)
# import ipdb; ipdb.set_trace()


with torch.no_grad():
    # output = self.model([preprocess_img])# Inference
    results = model(imgs)
    # results = model([image])


# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)


# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

import ipdb; ipdb.set_trace()
# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(results.xyxy[0].numpy()).resize((1280, 720))
r.putpalette(colors)

plt.imshow(r)
plt.show()
import time
time.sleep(10)