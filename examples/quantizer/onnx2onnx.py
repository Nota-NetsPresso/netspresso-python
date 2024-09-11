import json
from pathlib import Path

from netspresso import NetsPresso
from netspresso.enums import QuantizationDataType


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

def save_json(file_path, results):
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)


# 1. Declare quantizer
quantizer = netspresso.quantizer()

# 2. Set variables for quantize
input_model = "./examples/sample_models/yolo-fastest.onnx"
OUTPUT_DIR = "./outputs/quantized/onnx2onnx_2"
CALIBRATION_DATASET_PATH = "./examples/sample_datasets/pickle_calibration_dataset_128x128.npy"
BITWIDTH = QuantizationDataType.INT8

# 3. Run quantize
metadata = quantizer.quantize_model(
    input_model_path=input_model,
    output_dir=f"{OUTPUT_DIR}/{Path(input_model).stem}_{BITWIDTH}",
    threshold=0,
    weight_quantization_bitwidth=BITWIDTH,
    activation_quantization_bitwidth=BITWIDTH,
    dataset_path=CALIBRATION_DATASET_PATH,
    sleep_interval=10,
)
