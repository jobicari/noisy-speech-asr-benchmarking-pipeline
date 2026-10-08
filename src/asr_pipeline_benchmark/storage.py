from abc import ABC, abstractmethod
import pickle
from pathlib import Path
from typing import List


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
        return [str(p) for p in dir_path.glob("*.pkl")]

    def delete(self, file_key: str):
        if Path(file_key).exists():
            Path(file_key).unlink()


