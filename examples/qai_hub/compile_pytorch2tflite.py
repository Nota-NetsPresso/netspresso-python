from netspresso import QAIHub
from netspresso.np_qai import Device
from netspresso.np_qai.options import CompileOptions, ComputeUnit, ProfileOptions, Runtime, TfliteOptions

API_TOKEN = "YOUR API TOKEN"
qai_hub = QAIHub(api_token=API_TOKEN)

converter = qai_hub.converter()

convert_options = CompileOptions(
    target_runtime=Runtime.TFLITE,
    compute_unit=[ComputeUnit.CPU, ComputeUnit.GPU],
)

convert_tasks = {}
completed_convert_tasks = set()

for i in range(3):
    conversion_task = converter.convert_model(
        input_model_path="./examples/sample_models/pytorch_model_automatic_0.9.onnx",
        output_dir="./outputs/qai_hub/qnn_context_binary",
        target_device_name=Device(name="QCS6490 (Proxy)"),
        options=convert_options,
        input_shapes=dict(image=(1, 3, 64, 64)),
    )
    convert_tasks[i] = conversion_task
    print(f"Task {i} started")

while len(completed_convert_tasks) < 3:
    for i in range(3):
        if i not in completed_convert_tasks:
            status = converter.get_convert_task_status(convert_tasks[i].convert_task_info.convert_task_uuid)
            if status.finished:
                convert_tasks[i] = converter.update_convert_task(convert_tasks[i])
                completed_convert_tasks.add(i)
                print(f"Task {i} completed")

print("All tasks completed!")


benchmarker = qai_hub.benchmarker()

benchmark_option = ProfileOptions(
    compute_unit=[ComputeUnit.GPU],
    tflite_options=TfliteOptions(number_of_threads=4)
)

benchmark_tasks = {}
completed_benchmark_tasks = set()

for i in range(3):
    profile_job = benchmarker.benchmark_model(
        input_model_path=convert_tasks[0].converted_model_path,
        target_device_name=Device("QCS6490 (Proxy)"),
        options=benchmark_option,
    )
    benchmark_tasks[i] = profile_job
    print(f"Task {i} started")

while len(completed_benchmark_tasks) < 3:
    for i in range(3):
        if i not in completed_benchmark_tasks:
            status = benchmarker.get_benchmark_task_status(benchmark_tasks[i].benchmark_task_info.benchmark_task_uuid)
            if status.finished:
                benchmark_tasks[i] = benchmarker.update_benchmark_task(benchmark_tasks[i])
                completed_benchmark_tasks.add(i)
                print(f"Task {i} completed")

print("All tasks completed!")
