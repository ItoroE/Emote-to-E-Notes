# Emote to E-Notes

Emotion-conditioned MIDI generation using facial-expression recognition and a GRU-based conditional variational autoencoder (CVAE).

## Overview

The system converts webcam frames into valence and arousal signals, then uses those signals to condition MIDI note generation. Three FastAPI services handle emotion inference, music generation, and playback.

## Processing pipeline

1. A browser client submits camera frames to the FastAPI emotion service (`POST /Emote`).
2. OpenCV and dlib detect a face and 68 facial landmarks. An `EmotionsDlib` estimator produces arousal, valence, intensity, and an emotion label.
3. Exponential smoothing stabilizes the frame-level values. The service aggregates roughly 10-second windows and forwards completed emotion batches to the MIDI generation service.
4. A PyTorch GRU-based conditional variational autoencoder (CVAE) generates MIDI note events conditioned on arousal and valence.
5. The MIDI service builds batch and cumulative MIDI files, logs the emotion-to-music timeline, and requests playback from a separate Pygame service.

### Services

| Component | Location | Description |
| --- | --- | --- |
| Browser interface | `musicUI/` | Captures webcam images and sends them to the emotion API. |
| Emotion service | `emotion_backend/server.py` | Runs on port `8000`, processes uploaded frames, and forwards completed emotion batches. |
| MIDI generation service | `music_backend/midi.py` | Runs on port `8002`, generates MIDI sequences, and maintains session output and logs. |
| Playback service | `music_backend/musicserver.py` | Runs on port `8001` and handles playback requests for generated MIDI files. |

The browser sends frames to the emotion service. When an emotion batch is ready, the emotion service sends its arousal and valence values to the MIDI generation service. The generated MIDI file is then passed to the playback service.


## Tech stack

Python, PyTorch, FastAPI, OpenCV, dlib, NumPy, PrettyMIDI, Pygame, HTTPX, HTML, and JavaScript.

## Project structure

| Path | Responsibility |
| --- | --- |
| `emotion_backend/emote.py` | Face detection, emotion inference, smoothing, batching, JSON logging |
| `emotion_backend/server.py` | `/Emote` API and asynchronous forwarding of emotion batches |
| `music_backend/train.py` | GRU-based `EmoCVAE` architecture |
| `music_backend/musicgenconversion.py` | Checkpoint loading and emotion-conditioned note generation |
| `music_backend/midi.py` | `/generate`, `/reset_midi`, MIDI assembly and CSV logs |
| `music_backend/musicserver.py` | `/play` endpoint and Pygame playback |
| `musicUI/` | Browser-side interface |
| `emopia_dataset.csv` | Dataset exported for music-model development |

## Model architecture

The CVAE uses a GRU encoder, latent mean and log-variance layers, a reparameterization step, and a GRU decoder. A separate emotion-prediction head encourages the latent representation to retain conditioning information. Each MIDI note is represented by **pitch, start time, duration, and velocity**; the conditioning vector contains **arousal and valence**.

## Setup

### Prerequisites

- Python environment with the dependencies imported by the source files
- dlib facial-landmark predictor and emotion-estimation model assets
- Compatible `bcvae_emopia_gru.pt` model checkpoint
- MIDI-capable playback environment

### Configuration

1. Install the required Python packages, including `python-multipart` for file uploads.
2. Place the dlib 68-point landmark predictor and the emotion-model files at the paths referenced in `emotion_backend/emote.py`.
3. Supply the trained checkpoint expected by `music_backend/musicgenconversion.py` (`bcvae_emopia_gru.pt`) and verify that its architecture matches the model definition.
4. Start each service in a separate terminal:

   - **Emotion backend:** Open a terminal in `emotion_backend/` and run Bash: `uvicorn server:app --port 8000`
   - **Playback backend:** Open another terminal in `music_backend/` and run Bash: `uvicorn musicserver:app --port 8001`
   - **MIDI generation backend:** Open a third terminal in `music_backend/` and run Bash: `uvicorn midi:app --port 8002`

5. Serve `musicUI/` locally and enable webcam access.



## API endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /Emote` (emotion service) | Analyze an uploaded image and return smoothed emotion values |
| `POST /generate` (MIDI service) | Generate and append MIDI from an emotion batch |
| `POST /reset_midi` (MIDI service) | Clear the cumulative MIDI session and logs |
| `POST /play` (playback service) | Start playback of a MIDI file path |

## Known limitations

- Required trained model weights and facial-analysis assets are not included.
- `EmoCVAE.forward()` calls `decode(z, c)` without the required `numnotes` argument; this needs correction for that execution path.
- Some generation batches may produce no MIDI notes.
- Playback depends on the local MIDI/audio configuration.
- Localhost URLs and permissive CORS settings are intended for development.

## Related repository

[Updating Emote to E-Notes](https://github.com/ItoroE/Updating-Emote-to-ENotes) — transformer-based waveform generation and stateful audio continuation.
