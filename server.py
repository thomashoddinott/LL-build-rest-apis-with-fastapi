import asyncio
from time import sleep
from datetime import datetime
from os import environ
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from http import HTTPStatus
from typing import Annotated
from PIL import Image
from io import BytesIO

import logging
import logs  # dummy db
import db
from db_challenge import VirtualMachine, insert, get


app = FastAPI()

# region challenge
VALID_IMAGES = {"ubuntu-24.04", "debian:bookworm", "alpine:3.20"}
@app.post("/vm/start")
def start_VM(vm: VirtualMachine):
    # validation
    if not (0 < vm.cpu_count < 65):
        raise HTTPException(status_code=400, detail="cpu_count must be between 1 and 64")
    if not (8 < vm.mem_size_gb < 1025):
        raise HTTPException(status_code=400, detail="mem_size_gb must be between 9 and 1024")
    if vm.image not in VALID_IMAGES:
        raise HTTPException(status_code=400, detail=f"image must be one of {VALID_IMAGES}")

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


if __name__ == "__main__":  # just run with `python server.py`
    import uvicorn

    uvicorn.run(app)
