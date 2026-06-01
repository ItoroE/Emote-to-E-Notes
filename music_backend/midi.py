from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import pretty_midi
import csv
import os
import time
import random

from musicgenconversion import (
    add_emotion,
    get_emotions,
    clear_emotions,
    generate_song,
    model,
    DEVICE
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MUSIC_SERVER_URL = "http://127.0.0.1:8001"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MIDI_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output_full.mid")
LOG_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "emotion_midi_log.csv")

full_midi = pretty_midi.PrettyMIDI()
full_instrument = pretty_midi.Instrument(program=0)
current_midi_time = 0.0
batch_id = 0

SEGMENT_SECONDS = 5.0
OVERLAP_SECONDS = 1.5


class EmotionBatch(BaseModel):
    start_batch: float
    end_batch: float
    arousal: float
    valence: float
    emotion_name: str


def write_log_row(row):
    file_exists = os.path.exists(LOG_OUTPUT_PATH)

    with open(LOG_OUTPUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch_id",
                "emotion_name",
                "arousal",
                "valence",
                "emotion_start_time",
                "emotion_end_time",
                "midi_note_start",
                "midi_note_end"
            ]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


@app.post("/reset_midi")
async def reset_midi():
    global full_midi, full_instrument, current_midi_time, batch_id

    full_midi = pretty_midi.PrettyMIDI()
    full_instrument = pretty_midi.Instrument(program=0)
    current_midi_time = 0.0
    batch_id = 0

    clear_emotions()

    if os.path.exists(MIDI_OUTPUT_PATH):
        os.remove(MIDI_OUTPUT_PATH)

    if os.path.exists(LOG_OUTPUT_PATH):
        os.remove(LOG_OUTPUT_PATH)

    return {
        "status": "reset",
        "message": "MIDI history and emotion log cleared"
    }

# used Chat to implement the servers discussing part.

@app.post("/generate")
async def generate(batch: EmotionBatch):
    global full_midi, full_instrument, current_midi_time, batch_id

    batch_id += 1

    add_emotion(batch.arousal, batch.valence)

    start_time = time.time()
    all_notes = generate_song(model, get_emotions(), DEVICE, segment_ms=int(SEGMENT_SECONDS * 1000))
    print(f"Generation took {time.time() - start_time:.2f}s")

    print("Total segments:", len(all_notes))
    for i, seg in enumerate(all_notes):
        print(f"Segment {i} note count:", len(seg))

    notes_to_use = []

    for seg_index, seg in enumerate(all_notes):
        segment_offset_ms = seg_index * SEGMENT_SECONDS * 1000.0

        for note in seg:
            new_note = note.clone()
            new_note[1] = float(new_note[1]) + segment_offset_ms
            notes_to_use.append(new_note)

    if len(notes_to_use) == 0:
        print(f"Batch {batch_id} produced no notes, skipping")
        return {
            "status": "skipped",
            "batch_id": batch_id,
            "reason": "no notes generated"
        }

    batch_note_start = current_midi_time
    latest_note_end = current_midi_time

    for note in notes_to_use:
        pitch = int(note[0])
        note_start = current_midi_time + float(note[1]) / 1000.0
        note_start += random.uniform(-0.05, 0.05)
        note_start = max(0.0, note_start)

        duration = float(note[2]) / 1000.0
        velocity = int(note[3])
        note_end = note_start + duration

        if duration <= 0:
            continue

        midi_note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=note_start,
            end=note_end
        )

        full_instrument.notes.append(midi_note)

        if note_end > latest_note_end:
            latest_note_end = note_end

    batch_note_end = latest_note_end
    current_midi_time += max(0.5, len(all_notes) * (SEGMENT_SECONDS - OVERLAP_SECONDS))

    full_midi.instruments = [full_instrument]
    full_midi.write(MIDI_OUTPUT_PATH)

    write_log_row({
        "batch_id": batch_id,
        "emotion_name": batch.emotion_name,
        "arousal": batch.arousal,
        "valence": batch.valence,
        "emotion_start_time": batch.start_batch,
        "emotion_end_time": batch.end_batch,
        "midi_note_start": batch_note_start,
        "midi_note_end": batch_note_end
    })

    print(f"Updated {MIDI_OUTPUT_PATH} with batch {batch_id}")

    batch_path = os.path.join(OUTPUT_DIR, f"batch_{batch_id}.mid")
    batch_midi = pretty_midi.PrettyMIDI()
    batch_instrument = pretty_midi.Instrument(program=0)

    min_start = min(float(note[1]) / 1000.0 for note in notes_to_use)

    for note in notes_to_use:
        pitch = int(note[0])
        note_start = float(note[1]) / 1000.0 - min_start
        note_start += random.uniform(-0.05, 0.05)
        note_start = max(0.0, note_start)

        duration = float(note[2]) / 1000.0
        velocity = int(note[3])
        note_end = note_start + duration

        if duration <= 0:
            continue

        batch_instrument.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=note_start,
                end=note_end
            )
        )

    batch_midi.instruments.append(batch_instrument)
    batch_midi.write(batch_path)

    print(f"Wrote batch file {batch_path}")
    print("Batch file size:", os.path.getsize(batch_path))

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(
                f"{MUSIC_SERVER_URL}/play",
                json={"path": batch_path}
            )
            print("music_server response:", r.status_code, r.text)

    except Exception as e:
        print("Failed to reach music_server:", e)

    return {
        "status": "generated",
        "file": MIDI_OUTPUT_PATH,
        "log": LOG_OUTPUT_PATH,
        "batch_file": batch_path,
        "batch_id": batch_id,
        "emotion_name": batch.emotion_name,
        "arousal": batch.arousal,
        "valence": batch.valence,
        "midi_note_start": batch_note_start,
        "midi_note_end": batch_note_end
    }