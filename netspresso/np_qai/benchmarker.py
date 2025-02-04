from pathlib import Path
from typing import List, Optional, Union

import qai_hub as hub
from loguru import logger
from qai_hub.client import Dataset, Device, InferenceJob, ProfileJob
from qai_hub.public_rest_api import DatasetEntries

from netspresso.enums import Status
from netspresso.metadata.benchmarker import BenchmarkerMetadata
from netspresso.np_qai.base import NPQAIBase
from netspresso.np_qai.options import InferenceOptions, ProfileOptions
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class NPQAIBenchmarker(NPQAIBase):
    def download_benchmark_results(self, job: ProfileJob, artifacts_dir: str):
        results = job.download_results(artifacts_dir=artifacts_dir)

        return results

    def download_profile(self, job: ProfileJob):
        profile = job.download_profile()

        return profile

    def get_benchmark_task_status(self, benchmark_task_uuid):
        job: ProfileJob = hub.get_job(benchmark_task_uuid)
        status = job.get_status()

        return status

    def update_benchmark_task(self, metadata: BenchmarkerMetadata):
        job: ProfileJob = hub.get_job(metadata.benchmark_task_info.benchmark_task_uuid)
        status = job.wait()

        if status.success:
            logger.info(f"{status.symbol} {status.state.name}")
            profile = self.download_profile(job=job)
            metadata.benchmark_result.latency = profile["execution_summary"]["estimated_inference_time"] / 1000
            metadata.benchmark_result.memory_footprint = profile["execution_summary"][
                "estimated_inference_peak_memory"
            ]
            metadata.status = Status.COMPLETED
        elif status.failure:
            logger.info(f"{status.symbol} {status.state}: {status.message}")
            metadata.status = Status.ERROR

        folder_path = Path(metadata.input_model_path).parent
        file_path = folder_path / "benchmark.json"
        metadatas = []
        if FileHandler.check_exists(file_path):
            metadatas = MetadataHandler.load_json(file_path)

        for i, stored_metadata in enumerate(metadatas):
            if stored_metadata.get("benchmark_task_info", {}).get("benchmark_task_uuid") == metadata.benchmark_task_info.benchmark_task_uuid:
                metadatas[i] = metadata.asdict()
                break

        MetadataHandler.save_json(data=metadatas, folder_path=folder_path, file_name="benchmark")

        return metadata

    def benchmark_model(
        self,
        input_model_path: Union[str, Path],
        target_device_name: Union[Device, List[Device]],
        options: Union[ProfileOptions, str] = ProfileOptions(),
        job_name: Optional[str] = None,
        retry: bool = True,
    ) -> Union[BenchmarkerMetadata, List[BenchmarkerMetadata]]:
        FileHandler.check_input_model_path(input_model_path)

        folder_path = Path(input_model_path).parent
        file_name = "benchmark"

        metadatas = []
        file_path = folder_path / f"{file_name}.json"
        if FileHandler.check_exists(file_path):
            metadatas = MetadataHandler.load_json(file_path)

        metadata = BenchmarkerMetadata()
        metadata.input_model_path = Path(input_model_path).resolve().as_posix()
        metadata.benchmark_task_info.device_name = target_device_name.name
        metadata.benchmark_task_info.display_device_name = target_device_name.name
        metadatas.append(metadata.asdict())
        MetadataHandler.save_json(data=metadatas, folder_path=folder_path, file_name=file_name)

        try:
            model_type = hub.client._determine_model_type(model=input_model_path)
            framework = self.get_framework_by_model_type(model_type=model_type)
            display_framework = self.get_display_framework(framework)
            metadata.benchmark_task_info.framework = framework
            metadata.benchmark_task_info.display_framework = display_framework

            cli_string = options.to_cli_string() if isinstance(options, ProfileOptions) else options

            job: ProfileJob = hub.submit_profile_job(
                model=input_model_path,
                device=target_device_name,
                name=job_name,
                options=cli_string,
                retry=retry,
            )

            metadata.benchmark_task_info.benchmark_task_uuid = job.job_id

        except KeyboardInterrupt:
            metadata.status = Status.STOPPED

        metadatas[-1] = metadata.asdict()
        MetadataHandler.save_json(data=metadatas, folder_path=folder_path, file_name=file_name)

        return metadata

    def get_inference_task_status(self, inference_task_uuid):
        job: InferenceJob = hub.get_job(inference_task_uuid)
        status = job.get_status()

        return status

    def inference_model(
        self,
        input_model_path: Union[str, Path],
        target_device_name: Union[Device, List[Device]],
        inputs: Union[Dataset, DatasetEntries, str],
        job_name: Optional[str] = None,
        options: Union[InferenceOptions, str] = InferenceOptions(),
        retry: bool = True,
    ) -> Union[InferenceJob, List[InferenceJob]]:

        cli_string = options.to_cli_string() if isinstance(options, InferenceOptions) else options

        job: InferenceJob = hub.submit_inference_job(
            model=input_model_path,
            device=target_device_name,
            inputs=inputs,
            name=job_name,
            options=cli_string,
            retry=retry,
        )

        return job
