from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from uuid import uuid4


@dataclass
class Sale:
    time: datetime
    customer_id: str
    sku: str
    amount: int
    price: int  # ¢


_lock = Lock()
_records = {}


def insert(sale: Sale):
    """Insert a sale into the database."""
    key = uuid4().hex
    with _lock:
        _records[key] = sale
    return key


def get(key) -> Sale:
    """Get a sale from the database."""
    with _lock:
        return _records.get(key)
