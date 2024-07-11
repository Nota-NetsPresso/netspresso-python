from pathlib import Path
from typing import List, Optional, Tuple, Union

import qai_hub as hub
from qai_hub.client import CompileJob, Dataset, Device, Model, SourceModel
from qai_hub.public_rest_api import DatasetEntries, InputSpecs

from netspresso.qai_hub.base import QAIHubBase


class QAIHubConverter(QAIHubBase):
    def convert_model(
        self,
        model: Union[Model, SourceModel, str, Path],
        device: Union[Device, List[Device]],
        name: Optional[str] = None,
        input_shape: Union[List, Tuple, None] = None,
        options: str = "",
        single_compile: bool = True,
        calibration_data: Union[Dataset, DatasetEntries, str, None] = None,
        retry: bool = True,
    ) -> Union[CompileJob, List[CompileJob]]:

        if isinstance(input_shape, List):
            input_shape = tuple(input_shape)

        compile_job = hub.submit_compile_job(
            model=model,
            device=device,
            name=name,
            input_specs={'image': input_shape},
            options=options,
            single_compile=single_compile,
            calibration_data=calibration_data,
            retry=retry,
        )

        return compile_job
