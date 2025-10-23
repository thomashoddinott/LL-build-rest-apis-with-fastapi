import asyncio
from time import sleep
from datetime import datetime
from os import environ
from fastapi import FastAPI, HTTPException

import logs  # dummy db

app = FastAPI()


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
