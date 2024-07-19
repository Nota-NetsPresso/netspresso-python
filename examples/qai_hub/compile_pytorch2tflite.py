from typing import Tuple

import torch
import torchvision

from netspresso import QAIHub
from netspresso.qai_hub import Device
from netspresso.qai_hub.options import CompileOptions, Runtime

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
)

conversion_task = converter.convert_model(
    input_model_path="./examples/sample_models/pytorch_model_automatic_0.9.onnx",
    output_dir="./outputs/qai_hub/qnn_context_binary",
    target_device_name=Device(name="Samsung Galaxy S23 (Family)"),
    options=options,
    input_shape=input_shape,
)

import ipdb; ipdb.set_trace()
print(conversion_task)

# benchmarker = qai_hub.benchmarker()

# profile_job = benchmarker.benchmark_model(
#     model=conversion_task.converted_model_path,
#     output_dir="./outputs/qai_hub/qnn_context_binary",
#     device=Device("Samsung Galaxy S23 (Family)"),
# )

# input_tensor = np.random.random(input_shape).astype(np.float32)

# inference_job = benchmarker.inference_model(
#     model=compile_job.get_target_model(),
#     device=hub.Device("Samsung Galaxy S23 (Family)"),
#     inputs=input_tensor,
# )

# print(compile_job)
# print(profile_job)
# print(inference_job)
