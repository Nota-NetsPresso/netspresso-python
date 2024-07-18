from typing import Tuple

import torch
import torchvision

import qai_hub as hub
from netspresso import QAIHub
from netspresso.qai_hub.options import CompileOptions, Runtime

# Using pre-trained MobileNet
torch_model = torchvision.models.mobilenet_v2(pretrained=True)
torch_model.eval()

# Trace model
input_shape: Tuple[int, ...] = (1, 3, 224, 224)
example_input = torch.rand(input_shape)
pt_model = torch.jit.trace(torch_model, example_input)

qai_hub = QAIHub(api_token="660f4ed09d67e9931837db9f52ab48308415d360")
converter = qai_hub.converter()
benchmarker = qai_hub.benchmarker()

# import ipdb; ipdb.set_trace()

options = CompileOptions(
    target_runtime=Runtime.TFLITE,
)

compile_job = converter.convert_model(
    model="./pytorch_automatic_compression_1.onnx",
    output_dir="./outputs/qai_hub/qnn_context_binary",
    name="MyMobileNet",
    device=hub.Device("Samsung Galaxy S23 (Family)"),
    options=options,
    input_shape=input_shape,
)
print(compile_job)

# profile_job = benchmarker.benchmark_model(
#     model=compile_job.get_target_model(),
#     device=hub.Device("Samsung Galaxy S23 (Family)"),
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
