const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const resetBtn = document.getElementById("resetBtn");

const emotionOutput = document.getElementById("emotionOutput");
const midiStatus = document.getElementById("midiStatus");


#connectionofallservers
const EMOTION_SERVER = "http://127.0.0.1:8000";
const MUSIC_SERVER = "http://127.0.0.1:8001";
const MIDI_SERVER = "http://127.0.0.1:8002"

let autoInterval = null;

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
}

async function scanOnce() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
      const res = await fetch(`${EMOTION_SERVER}/Emote`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (data.detected) {
        emotionOutput.textContent = JSON.stringify({
          emotion: data.emotion_name,
          arousal: data.arousal,
          valence: data.valence,
          frame: data.frame,
        }, null, 2);
      } else {
        emotionOutput.textContent = "No face detected";
      }

      if (data.batch) {
        midiStatus.textContent =
          `Batch ${data.batch.batch_id} generated — ` +
          `emotion: ${data.batch.emotion_name}, ` +
          `arousal: ${data.batch.arousal.toFixed(2)}, ` +
          `valence: ${data.batch.valence.toFixed(2)}`;
      }

    } catch (err) {
      emotionOutput.textContent = `Error: ${err.message}`;
    }
  }, "image/jpeg");
}


async function startSystem() {
  await resetSession();

  await startCamera();

  if (!autoInterval) {
    autoInterval = setInterval(scanOnce, 2000);
  }
}
async function resetSession() {
  try {
    await fetch(`${MIDI_SERVER}/reset_midi`, {
      method: "POST",
    });

    midiStatus.textContent = "Session reset. Ready to record.";
    emotionOutput.textContent = "No data yet";

  } catch (err) {
    midiStatus.textContent = `Reset failed: ${err.message}`;
  }
}


function stopSystem() {
  if (autoInterval) {
    clearInterval(autoInterval);
    autoInterval = null;
  }

  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
    video.srcObject = null;
  }
}

startBtn.addEventListener("click", startSystem);
stopBtn.addEventListener("click", stopSystem);
resetBtn.addEventListener("click", resetSession); 