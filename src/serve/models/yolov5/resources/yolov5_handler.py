# This module is developed based on
# https://gist.github.com/joek13/b895db0cd50a7c71a123611885057c69#file-torchserve_handler-py

"""Custom TorchServe model handler for YOLOv5 models.
"""
import base64
import io
from typing import List

# import numpy as np
import torch

# import torchvision
import torchvision.transforms as tf
from PIL import Image
from torch import Tensor
from ts.torch_handler.base_handler import BaseHandler

# import yolov5 utils
from yolov5.utils.general import non_max_suppression


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    img_size = 640
    """Image size (px). Images will be resized to this resolution before inference.
    """

    def __init__(self):
        # call superclass initializer
        super().__init__()

    def preprocess(self, data: List[dict]):
        """Converts input images to float tensors.
        Args:
            data (list): List of the data from the request input.

        Returns:
            Tensor: single Tensor of shape [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE]
        """
        images = []

        transform = tf.Compose(
            [tf.ToTensor(), tf.Resize((self.img_size, self.img_size))]
        )

        # load images
        # taken from https://github.com/pytorch/serve/blob/master/ts/torch_handler/vision_handler.py # noqa: E501

        # handle if images are given in base64, etc.
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            # force convert to tensor
            # and resize to [img_size, img_size]
            image = transform(image)

            images.append(image)

        # convert list of equal-size tensors to single stacked tensor
        # has shape BATCH_SIZE x 3 x IMG_SIZE x IMG_SIZE
        images_tensor = torch.stack(images).to(self.device)

        return images_tensor

    def postprocess(self, data: Tensor):
        """The post process function makes use of the output from the
        inference and converts into a Torchserve supported response output.

        Args:
            data (Torch Tensor): The torch tensor received from the prediction
                output of the model.

        Returns:
            List: The post process function returns a list of the predicted
                output.
        """
        # perform NMS (nonmax suppression) on model outputs
        pred = non_max_suppression(data[0])

        # initialize empty list of detections for each image
        detections = [[] for _ in range(len(pred))]

        for i, image_detections in enumerate(pred):  # axis 0: for each image
            for det in image_detections:  # axis 1: for each detection
                # x1,y1,x2,y2 in normalized image coordinates (i.e. 0.0-1.0)
                xyxy = det[:4] / self.img_size
                # confidence value
                conf = det[4].item()
                # index of predicted class
                class_idx = int(det[5].item())
                # get label of predicted class
                # if missing, then just return class idx
                label = self.mapping.get(str(class_idx), class_idx)

                detections[i].append(
                    {
                        label: [
                            xyxy[0].item(),
                            xyxy[1].item(),
                            xyxy[2].item(),
                            xyxy[3].item(),
                        ],
                        "score": conf,
                    }
                )

        return detections
