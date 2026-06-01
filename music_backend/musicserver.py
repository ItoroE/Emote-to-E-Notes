from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pygame
import threading
import time
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

_playback_lock = threading.Lock()


class PlayRequest(BaseModel):
    path: str


def _play(path: str):
    with _playback_lock:
        print("Play request path:", path)

        if not os.path.exists(path):
            print("File does not exist:", path)
            return

        if os.path.getsize(path) == 0:
            print("File is empty:", path)
            return

        print("Loading MIDI file...")
        pygame.mixer.music.stop()
        pygame.mixer.music.load(path)

        print("Playing MIDI file...")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        print("Finished playing.")


@app.post("/play")
async def play(req: PlayRequest):
    t = threading.Thread(target=_play, args=(req.path,), daemon=True)
    t.start()

    return {
        "status": "playing",
        "file": req.path
    }