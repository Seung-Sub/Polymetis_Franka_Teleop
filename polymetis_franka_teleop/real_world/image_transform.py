"""
Picklable image transform class for multiprocessing.

This module is separate to ensure proper pickling when using
spawn multiprocessing method.
"""

import cv2
import numpy as np


class ImageTransform:
    """Picklable image transform class for multiprocessing.

    Uses cv2.resize directly instead of get_image_transform closure
    to ensure picklability for multiprocessing.
    """
    def __init__(self, input_res, output_res, bgr_to_rgb=True, float32=False):
        self.input_res = input_res
        self.output_res = output_res
        self.bgr_to_rgb = bgr_to_rgb
        self.float32 = float32

    def __call__(self, data):
        img = data['color']

        # Resize if needed
        if img.shape[:2] != (self.output_res[1], self.output_res[0]):
            img = cv2.resize(img, self.output_res, interpolation=cv2.INTER_AREA)

        # Color conversion (BGR to RGB)
        if self.bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Float conversion
        if self.float32:
            img = img.astype(np.float32) / 255

        data['color'] = img
        return data

    def __repr__(self):
        return f'ImageTransform({self.input_res} -> {self.output_res}, bgr_to_rgb={self.bgr_to_rgb}, float32={self.float32})'
