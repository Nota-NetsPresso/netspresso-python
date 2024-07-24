from typing import Tuple

import torch
import torchvision

from netspresso import QAIHub
from netspresso.qai_hub import Device
from netspresso.qai_hub.options import CompileOptions, Runtime, ComputeUnit, ProfileOptions, TfliteOptions

# Using pre-trained MobileNet
torch_model = torchvision.models.mobilenet_v2(pretrained=True)
torch_model.eval()

# Trace model
qai_hub = QAIHub(api_token="660f4ed09d67e9931837db9f52ab48308415d360")
converter = qai_hub.converter()

input_shape: Tuple[int, ...] = (1, 3, 64, 64)
# INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [224, 224]}]
options = CompileOptions(
    target_runtime=Runtime.TFLITE,
    compute_unit=[ComputeUnit.CPU, ComputeUnit.GPU]
)

conversion_task = converter.convert_model(
    input_model_path="./examples/sample_models/pytorch_model_automatic_0.9.onnx",
    output_dir="./outputs/qai_hub/qnn_context_binary",
    target_device_name=Device(name="QCS6490 (Proxy)"),
    options=options,
    input_shape=input_shape,
)
print(conversion_task)

# import ipdb; ipdb.set_trace()
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

# input_tensor = np.random.random(input_shape).astype(np.float32)

# inference_job = benchmarker.inference_model(
#     model=compile_job.get_target_model(),
#     device=hub.Device("Samsung Galaxy S23 (Family)"),
#     inputs=input_tensor,
# )

# print(compile_job)
# print(profile_job)
# print(inference_job)
