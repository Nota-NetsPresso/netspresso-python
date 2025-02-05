from celery import Celery

from app.services.user import user_service
from netspresso.utils.db.session import get_db

REDIS_URL = "localhost:6379"
REDIS_PASSWORD = ""

connection_url = f"redis://:{REDIS_PASSWORD}@{REDIS_URL}" if REDIS_PASSWORD else f"redis://{REDIS_URL}"


app = Celery('netspresso_converter',
             broker=f"{connection_url}/0",
             backend=f"{connection_url}/0")


@app.task
def convert_model_task(
    api_key: str,
    input_model_path: str,
    output_dir: str,
    target_framework: str,
    target_device_name: str,
    target_data_type: str,
    target_software_version: str = None,
    input_layer = None,
    dataset_path: str = None,
    input_model_id: str = None,
):
    db = next(get_db())
    netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
    converter = netspresso.converter_v2()
    task_id = converter.convert_model(
        input_model_path=input_model_path,
        output_dir=output_dir,
        target_framework=target_framework,
        target_device_name=target_device_name,
        target_data_type=target_data_type,
        target_software_version=target_software_version,
        input_layer=input_layer,
        dataset_path=dataset_path,
        wait_until_done=False,
        input_model_id=input_model_id
    )
    return task_id
