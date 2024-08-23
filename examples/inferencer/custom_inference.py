import cv2
import numpy as np

from netspresso import NetsPresso


def preprocessor():
    "Use the preprocess you used for training"
    pass

def postprocessor():
    "Use the postprocess you used for training"
    pass


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

inferencer = netspresso.custom_inferencer()

image_path = "YOUR_IMAGE_PATH"
colored_img = cv2.imread(image_path)
img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)
img_draw = img.copy()

# Preprocess
img = preprocessor(img)
dataset_path = "YOUR_DATASET_PATH"
np.save(dataset_path, img)

# Inference
input_model_path = "YOUR_MODEL_PATH"
outputs = inferencer.inference(input_model_path, dataset_path)

# Postprocess
pred = postprocessor()
