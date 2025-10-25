from threading import Lock
from typing import TypedDict

_lock = Lock()
_db = {}  # login -> user


class User(TypedDict):
    login: str
    uid: int
    name: str
    is_admin: bool
    icon: bytes


def get(login: str):
    """Get user from database"""
    with _lock:
        return _db.get(login)


def set(login: str, user: User):
    """Add a user to database"""
    with _lock:
        _db[login] = user
