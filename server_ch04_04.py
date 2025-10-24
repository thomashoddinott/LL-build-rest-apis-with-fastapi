from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from http import HTTPStatus

app = FastAPI()


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
