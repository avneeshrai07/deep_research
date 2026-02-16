# app.py
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import json
import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from datetime import datetime
from contextlib import asynccontextmanager



from main import main_function
# ----------------------------
# FastAPI App
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        pass

    except Exception as e:
        raise

    

    yield 


    try:
        print("Completed")
    except Exception as e:
        raise



# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "https://orbitaim-lime.vercel.app",
    "http://127.0.0.1:5050",
    "https://orbit.orbitaim.io"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/deep-research")
async def cold_email(request: Request):
    try:
        data = await request.json()
        # Shallow, Intermediate, Deep
        research_type = data.get('research_type', "Shallow")
        query = data.get('query', None)
        if not query:
            return HTTPException(status_code=500, detail="No User Query")
        return await main_function(research_type=research_type, query=query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ----------------------------
# Local Run
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8050,
        reload=True
    )
