from netspresso import NetsPresso
from netspresso.enums import QuantizationDataType, QuantizationMode, SimilarityMetric


EMAIL = "nppd_test_001@nota.ai"
PASSWORD = "Nota180928!"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

# 1. Declare quantizer
quantizer = netspresso.quantizer()

# 2. Set variables for quantize
INPUT_MODEL_PATH = "./examples/sample_models/yolo-fastest.onnx"
OUTPUT_DIR = "./outputs/quantized/onnx2onnx"
CALIBRATION_DATASET_PATH = "./examples/sample_datasets/pickle_calibration_dataset_128x128.npy"

# 3. Run quantize
quantization_task = quantizer.quantize_model(
    input_model_path=INPUT_MODEL_PATH,
    output_dir=OUTPUT_DIR,
    threshold=0,
    weight_quantization_bandwidth=QuantizationDataType.INT8,
    activation_quantization_bandwidth=QuantizationDataType.INT8,
    dataset_path=CALIBRATION_DATASET_PATH,
)
print(quantization_task)
