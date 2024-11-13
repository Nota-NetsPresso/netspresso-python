from pathlib import Path

from netspresso import NetsPresso
from netspresso.enums import QuantizationPrecision, OnnxOperator
from netspresso.quantizer.quantizer import PrecisionByOperator


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)


# 1. Declare quantizer
quantizer = netspresso.quantizer()

# 2. Set variables for quantize
input_model = "./examples/sample_models/test.onnx"
OUTPUT_DIR = "./outputs/quantized/onnx2onnx_2"
CALIBRATION_DATASET_PATH = "./examples/sample_datasets/pickle_calibration_dataset_128x128.npy"

precision_by_operator_type = [
    PrecisionByOperator(
        type=OnnxOperator.HardSigmoid,
        precision=QuantizationPrecision.FLOAT16,
    )
]

# 3. Custom Quantization
quantizer.custom_quantization_by_operator_type(
    input_model_path=input_model,
    output_dir=f"{OUTPUT_DIR}/{Path(input_model).stem}",
    precision_by_operator_type=precision_by_operator_type,
    dataset_path=CALIBRATION_DATASET_PATH,
)
