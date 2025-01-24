import numpy as np
from netspresso import QAIHub
from netspresso.qai_hub.options.quantize import QuantizeOptions, RangeScheme, QuantizePrecision


API_TOKEN = "YOUR API TOKEN"
qai_hub = QAIHub(api_token=API_TOKEN)

quantizer = qai_hub.quantizer()

quantize_options = QuantizeOptions(
    range_scheme=RangeScheme.MSE_MINIMIZER,
)

quantize_tasks = {}
completed_quantize_tasks = set()
calibration_data = {"images": [np.random.randn(1, 3, 128, 128).astype(np.float32)]}

for i in range(1):
    quantize_task = quantizer.quantize_model(
        input_model_path="./examples/sample_models/yolo-fastest.onnx",
        output_dir="./outputs/qai_hub/quantized_model",
        calibration_data=calibration_data,
        weights_dtype=QuantizePrecision.INT8,
        activations_dtype=QuantizePrecision.INT8,
    )
    quantize_tasks[i] = quantize_task
    print(f"Task {i} started")

while len(completed_quantize_tasks) < 1:
    for i in range(1):
        if i not in completed_quantize_tasks:
            status = quantizer.get_quantize_task_status(quantize_tasks[i].quantize_info.quantize_task_uuid)
            if status.finished:
                quantize_tasks[i] = quantizer.update_quantize_task(quantize_tasks[i])
                completed_quantize_tasks.add(i)
                print(f"Task {i} completed")
            else:
                print(f"Task {i} is still running")

print("All tasks completed!")

