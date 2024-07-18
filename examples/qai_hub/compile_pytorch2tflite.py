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
input_shape: Tuple[int, ...] = (1, 3, 64, 64)
example_input = torch.rand(input_shape)
pt_model = torch.jit.trace(torch_model, example_input)

qai_hub = QAIHub(api_token="660f4ed09d67e9931837db9f52ab48308415d360")
converter = qai_hub.converter()

options = CompileOptions(
    target_runtime=Runtime.TFLITE,
)

# import ipdb; ipdb.set_trace()
compile_job = converter.convert_model(
    model="./examples/sample_models/pytorch_model_automatic_0.9.onnx",
    output_dir="./outputs/qai_hub/qnn_context_binary",
    name="MyMobileNet",
    device=Device(name="Samsung Galaxy S23 (Family)"),
    options=options,
    input_shape=input_shape,
)
print(compile_job)

benchmarker = qai_hub.benchmarker()

profile_job = benchmarker.benchmark_model(
    model=compile_job.get_target_model(),
    output_dir="./outputs/qai_hub/qnn_context_binary",
    device=Device("Samsung Galaxy S23 (Family)"),
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
