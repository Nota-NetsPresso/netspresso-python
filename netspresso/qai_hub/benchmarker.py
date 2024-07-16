from pathlib import Path
from typing import List, Optional, Union

import qai_hub as hub
from qai_hub.client import Dataset, Device, InferenceJob, Model, ProfileJob, TargetModel
from qai_hub.public_rest_api import DatasetEntries

from netspresso.qai_hub.base import QAIHubBase


class QAIHubBenchmarker(QAIHubBase):
    def benchmark_model(
        self,
        model: Union[Model, TargetModel, str, Path],
        device: Union[Device, List[Device]],
        name: Optional[str] = None,
        options: str = "",
        retry: bool = True,
    ) -> Union[ProfileJob, List[ProfileJob]]:
        profile_job = hub.submit_profile_job(
            model=model,
            device=device,
            name=name,
            options=options,
            retry=retry,
        )

        return profile_job

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
