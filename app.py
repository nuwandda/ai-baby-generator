from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import fastapi as _fapi

import schemas as _schemas
import services as _services
import traceback


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to AI Baby Generator API"}


# Endpoint to test the backend
@app.get("/api")
async def root():
    return {"message": "Welcome to the AI Baby Generator with FastAPI"}


@app.post("/api/generate/")
async def generate_image(babyCreate: _schemas.BabyCreate = _fapi.Depends()):
    
    try:
        images = await _services.generate_image(babyCreate=babyCreate)
    except Exception as e:
        print(traceback.format_exc())
        return {"message": f"{e.args}"}
    
    payload = {
        "mime" : "image/jpg",
        "images": images
        }
    
    return payload
