from abc import ABC, abstractmethod
import pickle
from pathlib import Path
from typing import List
from google.cloud import storage


class StorageBackend(ABC):
    """Abstract base class for chunked storage operations"""

    @abstractmethod
    def exists(self, file_key: str) -> bool:
        pass

    @abstractmethod
    def load_pickle(self, file_key: str) -> object:
        pass

    @abstractmethod
    def save_pickle(self, data, file_key: str):
        pass

    @abstractmethod
    def list_checkpoints(self, prefix: str) -> list[str]:
        """Returns a list of all checkpoints matching the given prefix"""
        pass

    @abstractmethod
    def delete(self, file_key: str):
        pass


class LocalStorage(StorageBackend):

    def exists(self, file_key: str) -> bool:
        return Path(file_key).exists()

    def load_pickle(self, file_key: str) -> object:
        with open(file_key, "rb") as f:
            return pickle.load(f)

    def save_pickle(self, data, file_key: str):
        Path(file_key).parent.mkdir(parents=True, exist_ok=True)
        with open(file_key, "wb") as f:
            pickle.dump(data, f)

    def list_checkpoints(self, prefix: str) -> list[str]:
        """Emulates prefix matching for local storage"""
        dir_path = Path(prefix).parent
        file_prefix = Path(prefix).name
        if not dir_path.exists():
            return []
        return [str(p) for p in dir_path.glob(f"{file_prefix}*.pkl")]

    def delete(self, file_key: str):
        if Path(file_key).exists():
            Path(file_key).unlink()


class GCSStorage(StorageBackend):
    """Google Cloud Storage (GCS)  implementation of StorageBackend."""
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.client = storage.Client()                   # 1. Local: Initializes auth & HTTP session in RAM
        self.bucket = self.client.bucket(bucket_name)    # 2. Local: Creates a local Bucket reference object

    def exists(self, file_key: str) -> bool:
        blob = self.bucket.blob(file_key)                 # 3. Local: Creates a local Blob reference object
        return blob.exists()                              # HTTP HEAD request to GCS

    def load_pickle(self, file_key: str) -> object:
        blob = self.bucket.blob(file_key)
        data_bytes = blob.download_as_bytes()             # HTTP GET request to download
        return pickle.loads(data_bytes)

    def save_pickle(self, data, file_key: str):
        blob = self.bucket.blob(file_key)
        data_bytes = pickle.dumps(data)
        blob.upload_from_string(data_bytes, content_type="application/octet-stream")  # HTTP POST/PUT request to upload

    def list_checkpoints(self, prefix: str) -> list[str]:
        # list_blobs filters by object prefix natively
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
        return [blob.name for blob in blobs if blob.name.endswith(".pkl")]

    def delete(self, file_key: str):
        blob = self.bucket.blob(file_key)
        if blob.exists():
            blob.delete()               # HTTP DELETE request to remove



