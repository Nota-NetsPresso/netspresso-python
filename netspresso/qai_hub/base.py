from pathlib import Path
from typing import List, Optional, Union

import qai_hub as hub
from qai_hub.client import Device, Job


class QAIHubBase:
    def get_devices(self, name: str = "", os: str = "", attributes: Union[str, List[str]] = None) -> List[Device]:
        if attributes is None:
            attributes = []
        devices = hub.get_devices(name=name, os=os, attributes=attributes)

        return devices

    def get_device_attributes(self) -> List[str]:
        device_attributes = hub.get_device_attributes()

        return device_attributes

    def get_jobs(self, offset: int = 0, limit: int = 50, creator: Optional[str] = None) -> List[Job]:
        jobs = hub.get_jobs(offset=offset, limit=limit, creator=creator)

        return jobs

    def get_job(self, job_id: str) -> Job:
        job = hub.get_job(job_id=job_id)

        return job
