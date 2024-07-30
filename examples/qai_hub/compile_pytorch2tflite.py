from netspresso import QAIHub
from netspresso.qai_hub import Device
from netspresso.qai_hub.options import CompileOptions, Runtime, ComputeUnit, ProfileOptions, TfliteOptions


API_TOKEN = "YOUR API TOKEN"
qai_hub = QAIHub(api_token=API_TOKEN)

converter = qai_hub.converter()

convert_options = CompileOptions(
    target_runtime=Runtime.TFLITE,
    compute_unit=[ComputeUnit.CPU, ComputeUnit.GPU],
)
conversion_task = converter.convert_model(
    input_model_path="./examples/sample_models/pytorch_model_automatic_0.9.onnx",
    output_dir="./outputs/qai_hub/qnn_context_binary",
    target_device_name=Device(name="QCS6490 (Proxy)"),
    options=convert_options,
    input_shapes=dict(image=(1, 3, 64, 64)),
)

benchmarker = qai_hub.benchmarker()

benchmark_option = ProfileOptions(
    compute_unit=[ComputeUnit.GPU],
    tflite_options=TfliteOptions(number_of_threads=4)
)
profile_job = benchmarker.benchmark_model(
    input_model_path=conversion_task.converted_model_path,
    target_device_name=Device("QCS6490 (Proxy)"),
    options=benchmark_option,
)
