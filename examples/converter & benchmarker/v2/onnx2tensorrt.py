from netspresso import NetsPresso
from netspresso.enums import DeviceName, Framework, SoftwareVersion

###
# Available target frameworks for conversion with onnx models
#
# Framework.TENSORRT <-- For NVIDIA Devices
# Framework.OPENVINO <-- For Intel CPUs
# Framework.TENSORFLOW_LITE <-- For the devices like Raspberry Pi devices
# Framework.DRPAI <-- For Renesas Devices like RZ/V2M, RZ/V2L
#

###
# Available devices for Framework.TENSORRT (target_framework)
#
# DeviceName.JETSON_NANO
# DeviceName.JETSON_TX2
# DeviceName.JETSON_XAVIER
# DeviceName.JETSON_NX
# DeviceName.JETSON_AGX_ORIN
# DeviceName.AWS_T4
#

###
# Available software versions for jetson devices
#
# DeviceName.JETSON_NANO : SoftwareVersion.JETPACK_4_6, SoftwareVersion.JETPACK_4_4_1
# DeviceName.JETSON_TX2 : SoftwareVersion.JETPACK_4_6
# DeviceName.JETSON_XAVIER : SoftwareVersion.JETPACK_4_6
# DeviceName.JETSON_NX : SoftwareVersion.JETPACK_5_0_2, SoftwareVersion.JETPACK_4_6,
# DeviceName.JETSON_AGX_ORIN : SoftwareVersion.JETPACK_5_0_1
#

EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

# 1. Declare converter
converter = netspresso.converter_v2()

# 2. Set variables for convert
INPUT_MODEL_PATH = "./examples/sample_models/test.onnx"
OUTPUT_DIR = "./outputs/converted/JETSON_NANO"
TARGET_DEVICE_NAME = DeviceName.JETSON_NANO
TARGET_SOFTWARE_VERSION = SoftwareVersion.JETPACK_4_6

# 3. Run convert
conversion_task = converter.convert_tensorrt(
    input_model_path=INPUT_MODEL_PATH,
    output_dir=OUTPUT_DIR,
    target_device_name=TARGET_DEVICE_NAME,
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
