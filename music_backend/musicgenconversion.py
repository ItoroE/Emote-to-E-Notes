import torch
from train import EmoCVAE


# converting to script for easier pipeline
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NOTE_DIM = 4
NUM_NOTES = 1000
H_DIM = 1000
Z_DIM = 125
CONDITION_DIM = 2

NUM_EPOCHS = 100
BATCH_SIZE = 32
Learning_Rate = 3e-4
BETA = 0.001
LAMBDA_EMO = 0.1

CHECKPOINT_PATH = "bcvae_emopia_gru.pt"

model = EmoCVAE(
    #updated values when GRU was updated
    hidden_dim=H_DIM,
    z_dim=Z_DIM,
    condition_dim=CONDITION_DIM,
    num_notes=NUM_NOTES,
    note_dim=NOTE_DIM,
    beta=BETA,
    lambda_emo=LAMBDA_EMO
).to(DEVICE)

ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

print("Checkpoint type:", type(ckpt))
if isinstance(ckpt, dict):
    print("Checkpoint keys:", ckpt.keys())

model.load_state_dict(ckpt["model_state_dict"])

GLOBAL_MAX_START = ckpt["global_max_start"]
GLOBAL_MAX_DURATION = ckpt["global_max_duration"]

print("GLOBAL_MAX_START:", GLOBAL_MAX_START)
print("GLOBAL_MAX_DURATION:", GLOBAL_MAX_DURATION)

print("saved value with space:", ckpt.get("global_max_start "))
print("saved value no space:", ckpt.get("global_max_start"))

model.eval()

emo_history = []


def generation_condition(batch, keyname="C"):
    c = torch.tensor(
        [float(batch["arousal"]), float(batch["valence"])],
        dtype=torch.float32
    )
    return c


def generate_batch(model, batch, z, device, numnotes=NUM_NOTES, max_notes=NUM_NOTES):
    with torch.no_grad():
        c = generation_condition(batch).unsqueeze(0).to(device)

        generated = model.decode(z, c, numnotes)
        generated = generated.view(max_notes, 4)

        generated[:, 0] = generated[:, 0].clamp(0, 1)
        generated[:, 1] = generated[:, 1].clamp(0, 1)
        generated[:, 2] = generated[:, 2].clamp(0, 1)
        generated[:, 3] = generated[:, 3].clamp(0, 1)

        return generated.cpu()


def generate_song(model, emo_history, device, max_notes=NUM_NOTES, segment_ms=5000):
    all_notes = []

    if emo_history is None or len(emo_history) == 0:
        print("No emotion history found. Nothing to generate.")
        return []

    print(f"Generating song from {len(emo_history)} emotion segments.")

    for batch_index, batch in enumerate(emo_history):
        print(f"\nGenerating segment {batch_index}")
        print("Emotion batch:", batch)

        z = torch.randn(1, Z_DIM).to(device)

        notes = generate_batch(
            model=model,
            batch=batch,
            z=z,
            device=device,
            numnotes=max_notes,
            max_notes=max_notes
        )

        #unormalize values
        notes[:, 0] = notes[:, 0] * 127
        notes[:, 1] = notes[:, 1] * GLOBAL_MAX_START
        notes[:, 2] = torch.clamp(notes[:, 2] * GLOBAL_MAX_DURATION, min=20, max=2000)
        notes[:, 3] = notes[:, 3] * 127

        #debugging
        print("Raw generated note count:", len(notes))
        print("First 10 raw notes:", notes[:10])

        if len(notes) == 0:
            print("No raw notes generated.")
            all_notes.append(torch.zeros((0, 4)))
            continue

        min_start = torch.min(notes[:, 1])
        notes[:, 1] = notes[:, 1] - min_start

        notes = notes[notes[:, 1] < segment_ms]

        if len(notes) == 0:
            print("No notes inside local segment window.")
            all_notes.append(torch.zeros((0, 4)))
            continue

        end_times = notes[:, 1] + notes[:, 2]
        clipped_end_times = torch.clamp(end_times, max=segment_ms)
        notes[:, 2] = clipped_end_times - notes[:, 1]

        notes = notes[notes[:, 2] > 0]

        if len(notes) == 0:
            print("All notes had invalid durations after clipping.")
            all_notes.append(torch.zeros((0, 4)))
            continue

        notes_sorted = notes[notes[:, 1].argsort()]

        durations = [float(note[2]) for note in notes_sorted]

        if len(durations) > 0:
            print(
                f"Duration range: {min(durations):.2f} - {max(durations):.2f}, "
                f"count: {len(durations)}"
            )
        else:
            print("No durations generated for this segment.")
            # help to check if notes lenghts are coming out to short
            all_notes.append(torch.zeros((0, 4)))
            continue

        filtered = []
        repeat_count = 0
        last_pitch = -1

        for i in range(len(notes_sorted)):
            note = notes_sorted[i]

            duration = float(note[2])
            pitch = int(note[0])
            velocity = int(note[3])

            if duration < 10:
                continue

            if velocity < 10:
                velocity = 60
                note[3] = velocity

            if pitch < 21:
                note[0] = 21

            if pitch > 108:
                note[0] = 108

            if pitch == last_pitch:
                repeat_count += 1
            else:
                repeat_count = 0
                last_pitch = pitch

            filtered.append(note)

        if len(filtered) > 0:
            notes = torch.stack(filtered)
        else:
            notes = torch.zeros((0, 4))

        print(f"Notes before filter: {len(notes_sorted)}, after filter: {len(filtered)}")

        all_notes.append(notes)

    return all_notes


def add_emotion(arousal, valence, max_len=4):
    emo_history.append({
        "arousal": float(arousal),
        "valence": float(valence)
    })

    while len(emo_history) > max_len:
        emo_history.pop(0)

    print("Added emotion:", emo_history[-1])
    print("Emotion history length:", len(emo_history))


def clear_emotions():
    emo_history.clear()
    print("Emotion history cleared.")


def get_emotions():
    return emo_history.copy()


if __name__ == "__main__":
    test_emo_history = [
        {"arousal": 0.8, "valence": 0.6},
        {"arousal": 0.3, "valence": 0.2},
        {"arousal": 0.7, "valence": 0.9},
    ]

    notes_list = generate_song(model, test_emo_history, DEVICE)

    print(f"\nGenerated {len(notes_list)} segments.")
    for i, notes in enumerate(notes_list):
        print(f"Segment {i}: shape = {notes.shape}")