
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import httpx
import asyncio

from emote import uploaded_frame

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MIDI_SERVER_URL = "http://127.0.0.1:8002"


async def send_batch_to_midi(batch):
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{MIDI_SERVER_URL}/generate",
                json=batch
            )
            print("MIDI server response:", response.status_code, response.text)

    except Exception as e:
        print("Failed to send batch to MIDI server:", e)


@app.post("/Emote")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    result = uploaded_frame(frame)

    if result.get("batch") is not None:
        print("new batch:", result["batch"])
        asyncio.create_task(send_batch_to_midi(result["batch"]))

    return result