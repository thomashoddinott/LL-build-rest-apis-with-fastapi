# region all imports
import asyncio
from time import sleep
from datetime import datetime, timedelta
from os import environ
from fastapi import FastAPI, HTTPException, Form, Request, Response, Query
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
from io import BytesIO

import logging
import yaml
import logs  # dummy db
import db
from db_challenge import VirtualMachine, insert, get
from db_challenge_ch04 import query_logs

# endregion all imports


app = FastAPI()


# region CH4_challenge
import yaml
from fastapi.responses import PlainTextResponse


@app.get("/logs")
def get_logs(
    offset: int = Query(0, ge=0),
    count: int = Query(100, ge=1, le=1000),
    request: Request = None,
):
    logs_list = list(query_logs(offset, count))

    if not logs_list:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="no logs found")

    accept = request.headers.get("accept", "").lower()

    if "text/csv" in accept:
        lines = ["level,time,message"]
        for log in logs_list:
            lines.append(f"{log['level']},{log['time'].isoformat()},{log['message']}")
        return PlainTextResponse("\n".join(lines), media_type="text/csv")

    elif "application/yaml" in accept:
        yaml_data = yaml.dump(
            {
                "count": len(logs_list),
                "offset": offset,
                "logs": logs_list,
            },
            sort_keys=False,
            default_flow_style=False,
        )
        return PlainTextResponse(yaml_data, media_type="application/yaml")

    # Default JSON
    return {
        "count": len(logs_list),
        "offset": offset,
        "logs": logs_list,
    }


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


if __name__ == "__main__":  # just run with `python server.py`
    import uvicorn

    uvicorn.run(app)
