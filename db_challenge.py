from dataclasses import dataclass
from threading import Lock
from uuid import uuid4


@dataclass
class VirtualMachine:
    cpu_count: int
    mem_size_gb: int
    image: str


_lock = Lock()
_records = {}


def insert(vm: VirtualMachine):
    """Insert a VM into the in-memory database."""
    key = uuid4().hex
    with _lock:
        _records[key] = vm
    return key


def get(key) -> VirtualMachine:
    """Get a VM by its ID."""
    with _lock:
        return _records.get(key)
