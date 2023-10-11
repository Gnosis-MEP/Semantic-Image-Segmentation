FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime


ARG SIT_PYPI_USER_VAR
ARG SIT_PYPI_PASS_VAR
ENV SIT_PYPI_USER $SIT_PYPI_USER_VAR
ENV SIT_PYPI_PASS $SIT_PYPI_PASS_VAR
ENV POCV2_VERSION 4.5.5.64

RUN /opt/conda/bin/conda install -y -c fastai opencv-python-headless==${POCV2_VERSION}

RUN mkdir -p /root/.cache/torch/hub/checkpoints/

ADD https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth /root/.cache/torch/hub/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth
ADD https://github.com/pytorch/vision/zipball/v0.10.0 /root/.cache/torch/hub/v0.10.0.zip

WORKDIR /service
## install only the service requirements
ADD ./requirements.txt /service/requirements.txt
ADD ./setup.py /service/setup.py
RUN mkdir -p /service/semantic_image_segmentation/ && \
    touch /service/semantic_image_segmentation/__init__.py

RUN pip install --extra-index-url https://${SIT_PYPI_USER}:${SIT_PYPI_PASS}@arruda.pythonanywhere.com/simple -r requirements.txt  && \
    pip install tornado==6 && \
    rm -rf /tmp/pip* /root/.cache/pip


## add all the rest of the code and install the actual package
## this should keep the cached layer above if no change to the pipfile or setup.py was done.
ADD . /service
RUN pip install -e . && \
    rm -rf /tmp/pip* /root/.cache/pip
