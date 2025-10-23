import asyncio
from time import sleep
from fastapi import FastAPI

app = FastAPI()

@app.get('/sleep/sys')
def nsys_sleep():
    sleep(1)
    # return {'error': None}
    return {'obj': None}

@app.get('/sleep/async-sys')
# Don't do this
async def sys_sleep():
    await asyncio.sleep(1)
    return {'error': None}

@app.get('/sleep/async-aio')
# test with `hey -c 10 -n 10 http://localhost:8000/sleep/sys`
async def aio_sleep():
    await asyncio.sleep(1)
    return {'error': None}

if __name__ == "__main__": #just run with `python server.py`
    import uvicorn
    uvicorn.run(app)
