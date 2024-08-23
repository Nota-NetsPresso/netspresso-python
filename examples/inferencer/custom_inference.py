from glob import glob

from netspresso import NetsPresso


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

inferencer = netspresso.custom_inferencer()

valid_imgs = glob("/root/datasets/traffic-sign/images/valid/*.jpg")
input_model_path = "./outputs/inference_test (2)/detection_yolox_s_best.onnx"
dataset_path = ""


outputs = inferencer.inference(input_model_path, dataset_path)
