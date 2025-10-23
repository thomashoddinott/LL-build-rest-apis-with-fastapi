from fastapi import FastAPI 

app = FastAPI()

@app.get("/health")
def health():
    return {"errors": None}

if __name__ == "__main__": #just run with `python server.py`
    import uvicorn
    uvicorn.run(app)