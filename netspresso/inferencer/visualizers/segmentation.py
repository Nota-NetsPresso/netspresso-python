import cv2
import numpy as np
from matplotlib import pyplot as plt
from netspresso.inferencer.visualizers.utils import voc_color_map


class SegmentationVisualizer:
    def __init__(self, class_map, pallete=None, normalized=False, brightness_factor=1.5):
        self.cmap = voc_color_map(N=256, normalized=normalized, brightness_factor=brightness_factor)
        self.class_map = class_map

    def draw(self, image, pred, model_input_shape=None):
        resize_factor = max((image.shape[0] / model_input_shape[0]), (image.shape[1] / model_input_shape[1]))
        assert len(pred.shape) == 2
        size = pred.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == pred)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (pred == 255)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image
        # import ipdb; ipdb.set_trace()
        # visualize_image = image.copy()

        # return visualize_image

    def _convert(self, gray_image):
        assert len(gray_image.shape) == 2
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (gray_image == 255)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

    def __call__(self, results, images=None):
        if len(results.shape) == 3:
            result_images = []
            for _real_gray_image in results:
                result_images.append(self._convert(_real_gray_image)[np.newaxis, ...])

            return np.concatenate(result_images, axis=0)
        elif len(results.shape) == 2:
            return self._convert(results)
        else:
            raise IndexError(f"gray_image.shape should be either 2 or 3, but {results.shape} were indexed.")
