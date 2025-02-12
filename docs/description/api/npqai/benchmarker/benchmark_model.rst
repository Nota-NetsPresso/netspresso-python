Benchmark Model
===============

.. autofunction:: netspresso.np_qai.benchmarker.NPQAIBenchmarker.benchmark_model


Example
-------

Visit your job on `Qualcomm AI Hub <https://app.aihub.qualcomm.com/jobs/>`_ to see other inference metrics like memory usage, load time, layer by layer analysis and model visualization.

.. code-block:: python

    from netspresso import NPQAI
    from netspresso.np_qai import Device
    from netspresso.np_qai.options import ProfileOptions, TfliteOptions, ComputeUnit

    QAI_HUB_API_TOKEN = "YOUR_QAI_HUB_API_TOKEN"
    np_qai = NPQAI(api_token=QAI_HUB_API_TOKEN)
    
    benchmarker = np_qai.benchmarker()

    INPUT_MODEL_PATH = "YOUR_INPUT_MODEL_PATH"
    OUTPUT_DIR = "YOUR_OUTPUT_DIR"
    JOB_NAME = "YOUR_JOB_NAME"
    DEVICE_NAME = "QCS6490 (Proxy)"

    benchmark_options = ProfileOptions(
        compute_unit=[ComputeUnit.NPU],
        tflite_options=TfliteOptions(number_of_threads=4),
    )

    benchmark_result = benchmarker.benchmark_model(
        input_model_path=INPUT_MODEL_PATH,
        target_device_name=Device(DEVICE_NAME),
        options=benchmark_options,
        job_name=JOB_NAME,
    )

    # Monitor task status
    while True:
        status = benchmarker.get_benchmark_task_status(benchmark_result.benchmark_task_info.benchmark_task_uuid)
        if status.finished:
            benchmark_result = benchmarker.update_benchmark_task(benchmark_result)
            print("Benchmark task completed")
            break
        else:
            print("Benchmark task is still running")
