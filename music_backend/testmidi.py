import json
import pretty_midi

from musicgenconversion import add_emotion, get_emotions, clear_emotions, generate_song, model, DEVICE

data = [  {"start_batch": 1775973921.1915574, "end_batch": 1775973925.1786008, "arousal": 0.555, "valence": 0.54, "emotion_name": "Extremely happy"},
  {"start_batch": 1775973925.1786008, "end_batch": 1775973929.1847227, "arousal": 0.61, "valence": 0.81, "emotion_name": "Extremely happy"},
  {"start_batch": 1775973845.1936576, "end_batch": 1775973849.180462, "arousal": 0.31, "valence": 0.705, "emotion_name": "Extremely happy"},
  {"start_batch": 1775973849.180462, "end_batch": 1775973853.1700745, "arousal": 0.475, "valence": 0.405, "emotion_name": "Very delighted"},
  {"start_batch": 1775973853.1700745, "end_batch": 1775973855.173602, "arousal": 0.48, "valence": 0.43, "emotion_name": "Very delighted"},
  {"start_batch": 1775973855.173602, "end_batch": 1775973857.2001357, "arousal": 0.4, "valence": 0.45, "emotion_name": "Very delighted"},
  {"start_batch": 1775973857.2001357, "end_batch": 1775973859.2227175, "arousal": 0.48, "valence": 0.36, "emotion_name": "Very delighted"},
  {"start_batch": 1775973859.2227175, "end_batch": 1775973863.186081, "arousal": 0.44, "valence": 0.46, "emotion_name": "Very delighted"},
  {"start_batch": 1775973911.169873, "end_batch": 1775973913.1800628, "arousal": 0.34, "valence": 0.46, "emotion_name": "Very happy"},
  {"start_batch": 1775973871.1854255, "end_batch": 1775973873.1864476, "arousal": 0.36, "valence": 0.43, "emotion_name": "Moderately delighted"}]

print(data)

for i, batch in enumerate(data):
    add_emotion(batch["arousal"], batch["valence"])

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    all_notes = generate_song(model, get_emotions(), DEVICE)

    clear_emotions()

    for segment in all_notes:
        for note in segment:
            pitch = int(note[0])
            start = float(note[1]) / 1000.0
            duration = float(note[2]) / 1000.0
            velocity = int(note[3])
            end = start + duration

            midi_note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start,
                end=end
            )
            instrument.notes.append(midi_note)

    midi.instruments.append(instrument)

    output_path = f"emotion_batch_{i}_{batch['emotion_name'].replace(' ', '_')}.mid"
    midi.write(output_path)
    print(f"Wrote {output_path}")