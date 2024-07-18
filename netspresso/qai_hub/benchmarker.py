from pathlib import Path
from typing import List, Optional, Union

from loguru import logger
import qai_hub as hub
from qai_hub.client import Dataset, Device, InferenceJob, Model, ProfileJob, TargetModel
from qai_hub.public_rest_api import DatasetEntries

from netspresso.qai_hub.base import QAIHubBase
from netspresso.utils import FileHandler


class QAIHubBenchmarker(QAIHubBase):
    def download_benchmark_results(self, job: ProfileJob, artifacts_dir: str):
        job.download_results(artifacts_dir=artifacts_dir)

    def benchmark_model(
        self,
        model: Union[Model, TargetModel, str, Path],
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

        if wait_until_done:
            job = hub.get_job(job.job_id)
            status = job.wait()

            if status.success:
                logger.info(f"{status.symbol} {status.state.name}")
            else:
                logger.info(f"{status.symbol} {status.state}: {status.message}")

        self.download_benchmark_results(job=job, artifacts_dir=output_dir)

        return job

    def inference_model(
        self,
        model: Union[Model, TargetModel, str, Path],
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
