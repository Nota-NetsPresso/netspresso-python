import cv2
import numpy as np
from matplotlib import pyplot as plt


def _voc_color_map(N=256, normalized=False, brightness_factor=1.5):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << (7 - j))
            g = g | (bitget(c, 1) << (7 - j))
            b = b | (bitget(c, 2) << (7 - j))
            c = c >> 3

        # Adjust brightness
        r = min(255, max(0, r * brightness_factor))
        g = min(255, max(0, g * brightness_factor))
        b = min(255, max(0, b * brightness_factor))

        cmap[i] = np.array([r, g, b])

    if normalized:
        cmap = cmap / 255
    return cmap


class DetectionVisualizer:
    def __init__(self, class_map=None, normalized=False, brightness_factor=1.5):
        self.cmap = _voc_color_map(N=256, normalized=normalized, brightness_factor=brightness_factor)
        self.class_map = class_map

    def draw(self, image, detections, model_input_shape):
        resize_factor = max((image.shape[0] / model_input_shape[0]), (image.shape[1] / model_input_shape[1]))
        detections[0][:, 1::2] *= resize_factor
        detections[0][:, :4:2] *= resize_factor

        visualize_image = image.copy()
        for bbox_label, class_label in zip(detections[0], detections[1]):
            # Get bbox coordinates
            x1 = int(bbox_label[0])
            y1 = int(bbox_label[1])
            x2 = int(bbox_label[2])
            y2 = int(bbox_label[3])

            # Get bbox color
            color = self.cmap[class_label].tolist()

            # Draw bbox
            visualize_image = cv2.rectangle(visualize_image, (x1, y1), (x2, y2), color=color, thickness=2)

            # Get class name
            class_name = self.class_map[class_label] if self.class_map else str(class_label)

            # Draw class info
            text_size, _ = cv2.getTextSize(str(class_name), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_w, text_h = text_size
            visualize_image = cv2.rectangle(visualize_image, (x1, y1-5-text_h), (x1+text_w, y1), color=color, thickness=-1)
            visualize_image = cv2.putText(visualize_image, str(class_name), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return visualize_image

    def visualize_by_plt(self, image):
        plt.imshow(image)
        plt.show()
