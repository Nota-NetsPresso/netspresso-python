from glob import glob
import json
from pathlib import Path
import time

from netspresso import NetsPresso
from netspresso.enums import QuantizationDataType, QuantizationMode, SimilarityMetric


EMAIL = "nppd_test_001@nota.ai"
PASSWORD = "Nota180928!"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

def save_json(file_path, results):
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)


# 1. Declare quantizer
quantizer = netspresso.quantizer()

# 2. Set variables for quantize
input_models = glob("../trainer_models/segmentation*")
OUTPUT_DIR = "./outputs/quantized/onnx2onnx_2"
CALIBRATION_DATASET_PATH = "./examples/sample_datasets/pickle_calibration_dataset_224x224.npy"

# 3. Run quantize
bitwidths = [
    "int4",
    "int5",
    "int6",
    "int7",
    "int8",
    "int9",
    "int10",
    "int11",
    "int12",
    "int13",
    "int14",
    "int15",
    "int16",
    "int17",
    "int18",
    "int19",
    "int20",
    "int21",
    "int22",
    "int23",
    "int24",
    "int25",
    "int26",
    "int27",
    "int28",
    "int29",
    "int30",
    "int31",
]

results = []

file_path = 'results_segmentation.json'

for input_model in input_models:
    for bitwidth in bitwidths:
        start = time.time()
        metadata = quantizer.quantize_model(
            input_model_path=input_model,
            output_dir=f"{OUTPUT_DIR}/{Path(input_model).stem}_{bitwidth}",
            threshold=0,
            weight_quantization_bitwidth=bitwidth,
            activation_quantization_bitwidth=bitwidth,
            dataset_path=CALIBRATION_DATASET_PATH,
            sleep_interval=10,
        )
        end = time.time()
        print(f"Done!!!!!! {Path(input_model).stem}, {bitwidth}")
        results.append(
            {
                "model_name": Path(input_model).stem,
                "bitwidth": bitwidth,
                "quantize_task_uuid": metadata.quantize_task_info.quantize_task_uuid,
                "status": metadata.status,
                "elapsed_time": f"{end - start:.5f} sec"
            }
        )
        save_json(file_path, results)

print(results)
