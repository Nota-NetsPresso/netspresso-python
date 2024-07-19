from pathlib import Path
from typing import List, Optional, Union

from loguru import logger
import qai_hub as hub
from qai_hub.client import Dataset, Device, InferenceJob, Model, ProfileJob, TargetModel
from qai_hub.public_rest_api import DatasetEntries

from netspresso.enums import Status
from netspresso.qai_hub.base import QAIHubBase
from netspresso.metadata.benchmarker import BenchmarkerMetadata
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class QAIHubBenchmarker(QAIHubBase):
    def download_benchmark_results(self, job: ProfileJob, artifacts_dir: str):
        results = job.download_results(artifacts_dir=artifacts_dir)

        return results

    def benchmark_model(
        self,
        model: Union[str, Path],
        output_dir: str,
        device: Union[Device, List[Device]],
        name: Optional[str] = None,
        options: str = "",
        retry: bool = True,
        wait_until_done: bool = True,
    ) -> Union[ProfileJob, List[ProfileJob]]:
        job = hub.submit_profile_job(
            model=model,
            device=device,
            name=name,
            options=options,
            retry=retry,
        )

        folder_path = Path(model).parent
        
        metadata = BenchmarkerMetadata()
        metadata.input_model_path = Path(model).resolve().as_posix()
        file_path = folder_path / "benchmark.json"
        metadatas = []

        if FileHandler.check_exists(file_path):
            metadatas = MetadataHandler.load_json(file_path)

        if wait_until_done:
            job = hub.get_job(job.job_id)
            status = job.wait()

            if status.success:
                logger.info(f"{status.symbol} {status.state.name}")
                results = self.download_benchmark_results(job=job, artifacts_dir=output_dir)
                import ipdb; ipdb.set_trace()
                metadata.status = Status.COMPLETED
            else:
                logger.info(f"{status.symbol} {status.state}: {status.message}")
                metadata.status = Status.ERROR

        metadatas[-1] = metadata.asdict()
        MetadataHandler.save_json(data=metadatas, folder_path=folder_path, file_name="benchmark")

        return job

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
