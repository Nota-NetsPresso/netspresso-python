import logging
import os

import boto3
from botocore.exceptions import ClientError

from .utils.singleton import SingletonInstance


class S3Provider(SingletonInstance):
    def __init__(self) -> None:
        minio_host = os.getenv("STORAGE_URL")
        self.client = self._create_client(minio_host)
        if self.client:
            logging.info("S3 client created successfully")

    def _create_client(self, url):
        aws_access_key_id = os.getenv("STORAGE_ACCESS_KEY")
        aws_secret_access_key = os.getenv("STORAGE_SECRET_KEY")
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=None,
            region_name="ap-northeast-2",
        )
        s3client = session.client("s3", endpoint_url=f"http://{url}")
        return s3client

    def upload_file(self, file_name, bucket, object_name=None):
        if object_name is None:
            object_name = os.path.basename(file_name)
        try:
            self.client.upload_file(file_name, bucket, object_name)
        except Exception as e:
            logging.error(e)
            return False
        return True

    def upload_fileobj(self, fileobj, bucket, object_name):
        try:
            self.client.upload_fileobj(fileobj, bucket, object_name)
        except Exception as e:
            logging.error(e)
            return False
        return True

    def download_file(self, bucket, object_name, dest_path=None):
        if dest_path is None:
            dest_path = f"./{object_name}"
        try:
            self.client.download_file(bucket, object_name, dest_path)
        except Exception as e:
            logging.error(e)
            return False
        return True

    def prefix_exists(self, bucket, prefix):
        dir_prefix = prefix.strip("/")
        if dir_prefix:
            dir_prefix = dir_prefix + "/"
        response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return "Contents" in response

    def get_object(self, bucket, key):
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            return response
        except Exception as e:
            logging.error(e)
            return None

    def delete_file(self, bucket, key):
        try:
            self.client.delete_object(Bucket=bucket, Key=key)
        except Exception as e:
            logging.error(e)
            return False
        return True

    def move_file(self, src_bucket, dest_bucket, old_key, new_key):
        try:
            self.client.copy_object(Bucket=dest_bucket, CopySource={"Bucket": src_bucket, "Key": old_key}, Key=new_key)
            self.client.delete_object(Bucket=src_bucket, Key=old_key)
        except Exception as e:
            logging.error(e)
            return False
        return True

    def download_s3_folder(self, bucket, object_dir, local_dir=None):
        # If local_dir is not specified, create folder with same name as S3 folder
        if local_dir is None:
            local_dir = object_dir

        # Create local directory if it doesn't exist
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # List all objects in the S3 folder
        paginator = self.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=object_dir)

        try:
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        # Get the file path
                        s3_path = obj["Key"]
                        # Skip if it's the folder itself
                        if s3_path.endswith("/"):
                            continue
                        # Create local file path
                        local_path = os.path.join(local_dir, os.path.relpath(s3_path, object_dir))
                        # Create directory if it doesn't exist
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        # Download file
                        print(f"Downloading {s3_path} to {local_path}")
                        self.client.download_file(bucket, s3_path, local_path)
        except Exception as e:
            logging.error(e)
            return False
        return True

    def delete_folder(self, bucket: str, folder_path: str) -> bool:
        """
        Delete a folder and all its contents from an S3 bucket, including the empty folder marker

        Args:
            bucket (str): Name of the S3 bucket
            folder_path (str): Path of the folder to delete (e.g., "path/to/folder")

        Returns:
            bool: True if deletion was successful
        """
        # Ensure the folder path ends with a forward slash
        if not folder_path.endswith("/"):
            folder_path += "/"

        try:
            # List all objects in the folder
            paginator = self.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=folder_path)

            folder_exists = False

            # Delete objects in batches of 1000 (S3 delete_objects limit)
            for page in pages:
                if "Contents" not in page:
                    continue

                folder_exists = True

                # Prepare objects for deletion
                objects_to_delete = [{"Key": obj["Key"]} for obj in page["Contents"]]

                # If there are objects to delete
                if objects_to_delete:
                    # Delete the batch of objects
                    self.client.delete_objects(
                        Bucket=bucket,
                        Delete={
                            "Objects": objects_to_delete,
                            "Quiet": True,  # Don't return the deleted objects in the response
                        },
                    )

                    logging.info(f"Deleted {len(objects_to_delete)} objects from {bucket}/{folder_path}")

            # Delete the empty folder marker itself
            try:
                self.client.delete_object(Bucket=bucket, Key=folder_path)
                logging.info(f"Deleted empty folder marker {bucket}/{folder_path}")
            except ClientError as e:
                # Ignore error if the folder marker doesn't exist
                if e.response["Error"]["Code"] != "NoSuchKey":
                    logging.error(f"Error deleting empty folder marker {bucket}/{folder_path}: {str(e)}")
                    return False

            if not folder_exists:
                logging.info(f"Folder {bucket}/{folder_path} was already empty or didn't exist")

            return True

        except Exception as e:
            logging.error(f"Error deleting folder {folder_path} from bucket {bucket}: {str(e)}")
            return False

    def rename_folder(self, bucket: str, old_folder_path: str, new_folder_path: str) -> bool:
        """
        Rename (move) a folder and all its contents in an S3 bucket

        Args:
            bucket (str): Name of the S3 bucket
            old_folder_path (str): Current path of the folder (e.g., "path/to/old_folder")
            new_folder_path (str): New path for the folder (e.g., "path/to/new_folder")

        Returns:
            bool: True if rename was successful
        """
        # Ensure folder paths end with a forward slash
        if not old_folder_path.endswith("/"):
            old_folder_path += "/"
        if not new_folder_path.endswith("/"):
            new_folder_path += "/"

        try:
            # List all objects in the old folder
            paginator = self.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=old_folder_path)

            folder_exists = False

            for page in pages:
                if "Contents" not in page:
                    continue

                folder_exists = True

                for obj in page["Contents"]:
                    old_key = obj["Key"]

                    # Skip if this is the empty folder placeholder
                    if old_key == old_folder_path:
                        continue

                    # Create new key by replacing old path prefix with new path
                    new_key = new_folder_path + old_key[len(old_folder_path) :]
                    self.move_file(bucket, bucket, old_key, new_key)

            if not folder_exists:
                logging.warning(f"Folder {bucket}/{old_folder_path} was empty or didn't exist")
                return False

            # Create new empty folder marker
            try:
                self.client.put_object(Bucket=bucket, Key=new_folder_path, Body="")
            except Exception as e:
                logging.error(f"Error creating new folder marker: {str(e)}")

            # Delete old empty folder marker
            try:
                self.client.delete_object(Bucket=bucket, Key=old_folder_path)
            except ClientError as e:
                # Ignore error if the folder marker doesn't exist
                if e.response["Error"]["Code"] != "NoSuchKey":
                    logging.error(f"Error deleting old folder marker {bucket}/{old_folder_path}: {str(e)}")
                    return False

            logging.info(f"Successfully renamed folder {old_folder_path} to {new_folder_path}")
            return True

        except Exception as e:
            error_msg = f"Error renaming folder {old_folder_path} to {new_folder_path} in bucket {bucket}: {str(e)}"
            logging.error(error_msg)
            return False
