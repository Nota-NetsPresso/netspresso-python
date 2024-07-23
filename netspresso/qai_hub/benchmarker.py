from pathlib import Path
from typing import List, Optional, Union

import qai_hub as hub
from loguru import logger
from qai_hub.client import Dataset, Device, InferenceJob, Model, ProfileJob, TargetModel
from qai_hub.public_rest_api import DatasetEntries

from netspresso.enums import Status
from netspresso.metadata.benchmarker import BenchmarkerMetadata
from netspresso.qai_hub.base import QAIHubBase
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class QAIHubBenchmarker(QAIHubBase):
    def download_benchmark_results(self, job: ProfileJob, artifacts_dir: str):
        results = job.download_results(artifacts_dir=artifacts_dir)

        return results

    def download_profile(self, job: ProfileJob):
        profile = job.download_profile()

        return profile

    def benchmark_model(
        self,
        input_model_path: Union[str, Path],
        target_device_name: Union[Device, List[Device]],
        options: str = "",
        job_name: Optional[str] = None,
        retry: bool = True,
        wait_until_done: bool = True,
    ) -> Union[ProfileJob, List[ProfileJob]]:
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

            job: ProfileJob = hub.submit_profile_job(
                model=input_model_path,
                device=target_device_name,
                name=job_name,
                options=options,
                retry=retry,
            )

            metadata.benchmark_task_info.benchmark_task_uuid = job.job_id
            metadatas[-1] = metadata.asdict()
            MetadataHandler.save_json(data=metadatas, folder_path=folder_path, file_name=file_name)

            if wait_until_done:
                job = hub.get_job(job.job_id)
                status = job.wait()

                if status.success:
                    logger.info(f"{status.symbol} {status.state.name}")
                    profile = self.download_profile(job=job)
                    metadata.benchmark_result.latency = profile["execution_summary"]["estimated_inference_time"] / 1000
                    metadata.benchmark_result.memory_footprint = profile["execution_summary"]["estimated_inference_peak_memory"]
                    metadata.status = Status.COMPLETED
                else:
                    logger.info(f"{status.symbol} {status.state}: {status.message}")
                    metadata.status = Status.ERROR

            metadatas[-1] = metadata.asdict()
            MetadataHandler.save_json(data=metadatas, folder_path=folder_path, file_name=file_name)

            return metadata

        except KeyboardInterrupt:
            metadata.status = Status.STOPPED
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=folder_path, file_name=file_name)

    def inference_model(
        self,
        model: Union[str, Path],
        device: Union[Device, List[Device]],
        inputs: Union[Dataset, DatasetEntries, str],
        name: Optional[str] = None,
        options: str = "",
        retry: bool = True,
    ) -> Union[InferenceJob, List[InferenceJob]]:
        inference_job = hub.submit_inference_job(
            model=model,
            device=device,
            inputs={"image": [inputs]},
            name=name,
            options=options,
            retry=retry,
        )

        return inference_job
