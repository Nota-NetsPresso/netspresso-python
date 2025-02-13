from pathlib import Path

from netspresso import NetsPresso
from netspresso.enums import QuantizationPrecision, SimilarityMetric


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

# 1. Declare quantizer
quantizer = netspresso.quantizer()

# 2. Set variables for quantize
input_model = "./examples/sample_models/test.onnx"
OUTPUT_DIR = "./outputs/quantized/automatic_quantization"
CALIBRATION_DATASET_PATH = "./examples/sample_datasets/pickle_calibration_dataset_128x128.npy"

# 3. Automatic Quantization
quantization_result = quantizer.automatic_quantization(
    input_model_path=input_model,
    output_dir=f"{OUTPUT_DIR}/{Path(input_model).stem}",
    dataset_path=CALIBRATION_DATASET_PATH,
    weight_precision=QuantizationPrecision.INT8,
    activation_precision=QuantizationPrecision.INT8,
    metric=SimilarityMetric.SNR,
    threshold=0,
)
