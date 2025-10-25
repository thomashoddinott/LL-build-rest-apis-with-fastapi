# region all imports
import sqlite3
import math
from contextlib import asynccontextmanager
from enum import Enum
from functools import wraps
import asyncio
from time import sleep, perf_counter
from datetime import datetime, timedelta
from os import environ
from fastapi import FastAPI, HTTPException, Form, Request, Response, Query, Depends
from fastapi.responses import (
    RedirectResponse,
    StreamingResponse,
    JSONResponse,
    PlainTextResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_serializer, Field
from http import HTTPStatus
from typing import Annotated
from PIL import Image
from io import BytesIO, StringIO
import logging, csv, logs, base64

from db.db_challenge import VirtualMachine, insert, get
import db.db
import db.db_challenge_ch04
import db.db_ch05_02


# endregion all imports

# region config_logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
# endregion config_logging


# region setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db
    _db = sqlite3.connect("other_imports_ch05_04/trades.db", check_same_thread=False)
    try:
        yield
    finally:
        _db.close()


app = FastAPI(lifespan=lifespan)

if __name__ == "__main__":  # production `python server.py`
    from argparse import ArgumentParser

    import uvicorn
    from config import settings

    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=settings.port)
    args = parser.parse_args()

    settings.update(vars(args))

    if settings.port < 0 or settings.port > 65_535:
        raise SystemExit(f"error: invalid port - {settings.port}")

    uvicorn.run(app, port=settings.port)
# endregion setup

# === SERVER CODE BELOW ===

# region ch05_challenge
from fastapi import FastAPI, HTTPException, Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


app = FastAPI()

lat_km = 92
lng_km = 111


def distance(lat1, lng1, lat2, lng2):
    """Return euclidean distance (in kilometers) between two coordinates.

    >>> distance(0, 0, 1, 1)
    144.1700384962146
    """
    delta_lat = (lat1 - lat2) * lat_km
    delta_lng = (lng1 - lng2) * lng_km
    return math.hypot(delta_lat, delta_lng)


def parse_csv(fp):
    """Parse CSV, returns tuple of:
    - number of samples
    - total distance
    - average speed in km/h
    """
    reader = csv.DictReader(fp)
    start_time = end_time = None
    prev_lat = prev_lng = None
    count = total_distance = 0

    for row in reader:
        count += 1
        time = datetime.fromisoformat(row["time"])
        # Rows are in chronological order
        if start_time is None:
            start_time = time
        else:
            end_time = time

        lat, lng = float(row["lat"]), float(row["lng"])
        if prev_lat:
            total_distance += distance(lat, lng, prev_lat, prev_lng)
        prev_lat, prev_lng = lat, lng

    duration_hours = (end_time - start_time).total_seconds() / (60 * 60)
    speed = total_distance / duration_hours

    return count, total_distance, speed


MAX_CSV_SIZE = 5 * (1 << 20)  # 5 MB


def timed(fn):
    """A decorator that logs function run time."""
    fn_name = fn.__name__

    @wraps(fn)
    async def wrapper(*args, **kw):
        start = perf_counter()
        try:
            return await fn(*args, **kw)
        finally:
            duration = perf_counter() - start
            logging.info(f"[metric:{fn_name}.time] %.3f", duration)

    return wrapper


@app.post("/run")
@timed
async def run_stats(request: Request):
    if (mime_type := request.headers["content-type"]) != "text/csv":
        logging.error(f"bad format: {mime_type}")
        raise HTTPException(
            HTTPStatus.NOT_ACCEPTABLE,
            detail="not a CSV",
        )

    if (size := int(request.headers["Content-Length"])) > MAX_CSV_SIZE:
        logging.error(f"[run_stats] file too large: {size}")
        raise HTTPException(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            detail="file too large",
        )

    data = await request.body()
    fp = StringIO(data.decode())
    count, distance, speed = parse_csv(fp)
    out = {
        "count": count,
        "distance": distance,
        "speed": speed,
    }
    logging.info("run_stats - %s", out)
    return out


# endregion ch05_challenge

# region Databases
# $ `cd other_imports_ch05_04 && bash create-db.sh`
_db = None


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global _db

#     _db = sqlite3.connect("other_imports_ch05_04/trades.db", check_same_thread=False)
#     try:
#         yield
#     finally:
#         _db.close()


# app = FastAPI(lifespan=lifespan)


def get_cursor():
    with _db as cursor:
        yield cursor


class Side(Enum):
    buy = "buy"
    sell = "sell"


class Trade(BaseModel):
    user: str = Field(min_length=3)
    time: datetime
    symbol: str = Field(min_length=3)
    price: int = Field(gt=0)  # In ¢
    volume: int = Field(gt=0)
    side: Side


@app.post("/trades")
def new_trade(trade: Trade, cursor=Depends(get_cursor)):
    # TODO: Validate trade
    params = {
        "user": trade.user,
        "time": trade.time,
        "symbol": trade.symbol,
        "price": trade.price,
        "volume": trade.volume,
        "side": trade.side.value,
    }

    cursor.execute(insert_sql, params)
    return {"error": None}


insert_sql = """
INSERT INTO trades 
    (user, time, symbol, price, volume, side) 
VALUES 
    (:user, :time, :symbol, :price, :volume, :side) 
"""


# endregion Databases


# region Security
import other_imports_ch05_03.users as users


@app.get("/users/{login}")
def get_user(login: str):
    user = users.get(login)
    if user is None:
        return JSONResponse(
            status_code=HTTPStatus.NOT_FOUND,
            content={"error": f"{login!r} not found"},
        )

    return user


class User(BaseModel):
    login: str
    uid: int
    name: str
    is_admin: bool


@app.post("/users/{login}")
def set_user(login, user: User):
    users.set(login, user.model_dump())
    return {
        "error": None,
        "login": user.login,
    }


@app.post("/users/{login}/icon")
async def set_icon(login: str, request: Request):
    user = users.get(login)
    if user is None:
        return JSONResponse(
            status_code=HTTPStatus.NOT_FOUND,
            content={"error": f"{login!r} not found"},
        )

    data = await request.body()
    user["icon"] = data
    # user["icon"] = base64.b64encode(data).decode("ascii")  # store as text
    # TODO: fix
    users.set(login, user)

    return {
        "error": None,
        "login": login,
    }


# endregion Security


# region do_some_logging
@app.get("/posts/{login}")
def get_posts(login: str, since: str = None):
    if since:
        since = datetime.strptime(since, "%Y%m%d")
    else:
        since = datetime.now() - timedelta(days=7)
        # Round to day
        since = datetime(since.year, since.month, since.day)

    logging.info("get posts for %s since %s", login, since)
    posts = db_ch05_02.query_posts(login, since)
    return posts


@app.middleware("http")
async def timing(request: Request, call_next):
    start = perf_counter()
    response = await call_next(request)
    duration = perf_counter() - start
    logging.info(
        "[metric:call.duration] %s %s %d - %.2fs",
        request.method,
        request.url,
        response.status_code,
        duration,
    )
    return response


# endregion do_some_logging


# region CH4_challenge
class Log(BaseModel):
    level: str
    time: datetime
    message: str


class LogsResponse(BaseModel):
    count: int
    offset: int
    logs: list[Log]


@app.get("/logs")
def query_logs(req: Request, count: int = 100, offset: int = 0):
    if count < 1 or offset < 0:
        return Response(
            status_code=HTTPStatus.BAD_REQUEST,
            content="bad count or offset",
        )
    mime_type = req.headers.get("Accept", "application/json")
    if mime_type == "*/*":
        mime_type = "application/json"
    if mime_type not in {"application/json", "text/csv"}:
        return Response(
            status_code=HTTPStatus.BAD_REQUEST,
            content="bad Accept",
        )

    records = list(db_challenge_ch04.query_logs(offset, count))
    if not records:
        return Response(status_code=HTTPStatus.NOT_FOUND)

    fn = json_response if mime_type == "application/json" else csv_response
    return fn(records, offset)


def json_response(records: list[dict], offset: int) -> LogsResponse:
    logs = [Log(**r) for r in records]

    return LogsResponse(count=len(records), offset=offset, logs=logs)


def csv_response(logs: list[dict], _: int) -> Response:
    io = StringIO()
    writer = csv.DictWriter(io, fieldnames=["time", "level", "message"])
    writer.writeheader()
    writer.writerows(
        {
            "time": log["time"].isoformat(),
            "level": log["level"],
            "message": log["message"],
        }
        for log in logs
    )

    return Response(content=io.getvalue(), media_type="text/csv")


# endregion CH4_challenge


# region CH4_04
class FreqError(Exception):
    pass


@app.exception_handler(FreqError)
async def freq_error_handler(request: Request, exc: FreqError):
    return JSONResponse(
        status_code=HTTPStatus.BAD_REQUEST,
        content={"error": str(exc)},  # Security risk (could expose sensitive info)
        headers={
            "X-Freq-Text": request.query_params.get("text"),
        },
    )


def char_freq(text: str):
    if not text:
        raise FreqError("empty text")

    freqs = {}
    for c in text.lower():
        freqs[c] = freqs.get(c, 0) + 1
    return freqs


@app.get("/freq")
def freq(text: str):
    return char_freq(text)


# endregion CH4_04

# region CH4_03
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB


@app.post("/resize")
async def resize(width: int, height: int, request: Request):
    size = int(request.headers.get("Content-Length", 0))
    if not size:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="missing content-length header",
        )
    if size > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="image too large (max is 5MB)",
        )

    if width <= 0 or height <= 0:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="width and height must be positive",
        )

    data = await request.body()
    io = BytesIO(data)
    img = Image.open(io)
    img = img.resize((width, height))
    out = BytesIO()
    img.save(out, format="PNG")
    return Response(
        content=out.getvalue(),
        status_code=HTTPStatus.OK,
        media_type="image/png",
    )


# endregion CH4_03


# region CH4_02
class Event(BaseModel):
    time: datetime
    user: str
    action: str
    uri: str


def query_events(start_time: datetime):
    """Dummy query for events."""
    time = start_time
    for _ in range(10):
        time += timedelta(seconds=19)
        event = Event(
            time=time,
            user="elliot",
            action="read",
            uri="file:///etc/passwd",
        )
        yield event.model_dump_json() + "\n"


@app.get("/events")
async def get_gen(start: datetime):
    events = query_events(start)
    return StreamingResponse(events)


# endregion CH4_02


# region CH4_01
class TimeResponse(BaseModel):
    delta: timedelta

    @field_serializer("delta")
    def serialize_delta(self, v: timedelta) -> float:
        return int(v.total_seconds())


@app.get("/time_delta")
def time_diff(start: datetime, end: datetime) -> TimeResponse:
    delta = end - start
    return TimeResponse(delta=delta)


# endregion CH4_01

# region challenge
VALID_IMAGES = {"ubuntu-24.04", "debian:bookworm", "alpine:3.20"}


@app.post("/vm/start")
def start_VM(vm: VirtualMachine):
    # validation
    if not (0 < vm.cpu_count < 65):
        raise HTTPException(
            status_code=400, detail="cpu_count must be between 1 and 64"
        )
    if not (8 < vm.mem_size_gb < 1025):
        raise HTTPException(
            status_code=400, detail="mem_size_gb must be between 9 and 1024"
        )
    if vm.image not in VALID_IMAGES:
        raise HTTPException(
            status_code=400, detail=f"image must be one of {VALID_IMAGES}"
        )

    vm_id = insert(vm)
    return {"id": vm_id}


@app.get("/vm/{key}")
def get_vm(key: str) -> VirtualMachine:
    record = get(key)
    if record is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="vm not found")

    vm = VirtualMachine(
        cpu_count=record.cpu_count,
        mem_size_gb=record.mem_size_gb,
        image=record.image,
    )
    return vm


# endregion

# region CH3_04
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB


@app.post("/size")
async def size(request: Request):
    size = int(request.headers.get("Content-Length", 0))
    if not size:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="missing content-length header",
        )
    if size > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="image too large (max is 5MB)",
        )

    data = await request.body()
    io = BytesIO(data)
    img = Image.open(io)
    return {"width": img.width, "height": img.height}


# endregion

# region CH3_03
app.mount("/static", StaticFiles(directory="static"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


@app.post("/survey")
# http://localhost:8000/static/survey.html
def survey(
    name: Annotated[str, Form()],
    happy: Annotated[str, Form()],
    talk: Annotated[str, Form()],
):
    logging.info("[survey] name: %r, happy: %r, talk: %r", name, happy, talk)
    return RedirectResponse(
        url="/static/thanks.html",
        status_code=HTTPStatus.FOUND,
    )


# endregion


# region everything else
class Sale(BaseModel):
    time: datetime
    customer_id: str = Field(min_length=2)
    sku: str = Field(min_length=2)
    amount: int = Field(gt=0)
    price: float = Field(gt=0)  # $


@app.post("/sales/")
def new_sale(sale: Sale):
    record = db.Sale(
        time=sale.time,
        sku=sale.sku,
        customer_id=sale.customer_id,
        amount=sale.amount,
        price=int(sale.price * 100),
    )
    key = db.insert(record)
    return {
        "key": key,
    }


@app.get("/sales/{key}")
def get_sale(key: str) -> Sale:
    record = db.get(key)
    if record is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="sale not found")

    s = Sale(
        time=record.time,
        sku=record.sku,
        customer_id=record.customer_id,
        amount=record.amount,
        price=record.price / 100,
    )
    return s


@app.get("/logs")
def logs_query(start: datetime, end: datetime, level: str = None):
    if start >= end:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="start must be before end"
        )
    if not level or not logs.is_valid_level(level):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="invalid log level"
        )

    records = logs.query(start, end, level)
    if not records:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="no logs found")

    return {
        "count": len(records),
        "records": records,
    }


@app.get("/info")
def info():
    return {
        "version": "0.1.0",
        "time": datetime.utcnow().isoformat(),
        "env": environ["USER"],
    }


@app.get("/sleep/sys")
def nsys_sleep():
    sleep(1)
    return {"error": None}


@app.get("/sleep/async-sys")
# Don't do this
async def sys_sleep():
    await asyncio.sleep(1)
    return {"error": None}


@app.get("/sleep/async-aio")
# test with `hey -c 10 -n 10 http://localhost:8000/sleep/sys`
async def aio_sleep():
    await asyncio.sleep(1)
    return {"error": None}


# endregion everything else
