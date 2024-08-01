from netspresso import NetsPresso
from netspresso.enums import DeviceName, Framework, DataType, SoftwareVersion

###
# Available target frameworks for conversion with onnx models
#
# Framework.TENSORRT <-- For NVIDIA Devices
# Framework.OPENVINO <-- For Intel CPUs
# Framework.TENSORFLOW_LITE <-- For the devices like Raspberry Pi devices
# Framework.DRPAI <-- For Renesas Devices like RZ/V2M, RZ/V2L
# Framework.DLC <-- For GALAXY_S24_ULTRA
#

###
# Available devices for Framework.DLC (target_framework)
#
# DeviceName.GALAXY_S24_ULTRA
#

EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

# 1. Declare converter
converter = netspresso.converter_v2()

# 2. Set variables for convert
INPUT_MODEL_PATH = "./examples/sample_models/yolo-fastest.onnx"
OUTPUT_DIR = "./outputs/converted/DLC_GALAXY_S24_ULTRA"
TARGET_FRAMEWORK = Framework.DLC
TARGET_DEVICE_NAME = DeviceName.GALAXY_S24_ULTRA
TARGET_DATA_TYPE = DataType.FP32
TARGET_SOFTWARE_VERSION = SoftwareVersion.SNPE_2_20

# 3. Run convert
conversion_task = converter.convert_model(
    input_model_path=INPUT_MODEL_PATH,
    output_dir=OUTPUT_DIR,
    target_framework=TARGET_FRAMEWORK,
    target_device_name=TARGET_DEVICE_NAME,
    target_data_type=TARGET_DATA_TYPE,
    target_software_version=TARGET_SOFTWARE_VERSION,
)
print(conversion_task)

# 4. Declare benchmarker
benchmarker = netspresso.benchmarker_v2()

# 5. Run benchmark
benchmark_task = benchmarker.benchmark_model(
    input_model_path=conversion_task.converted_model_path,
    target_device_name=TARGET_DEVICE_NAME,
    target_software_version=TARGET_SOFTWARE_VERSION,
)
print(f"model inference latency: {benchmark_task.benchmark_result.latency} ms")
