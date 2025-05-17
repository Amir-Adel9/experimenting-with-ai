import time
import json
import whisperx

# === START TIMER ===
start_time = time.time()

# === CONFIG ===
audio_file = "audio/zakir_1.wav"
device = "cuda"  # or "cpu"
compute_type = "float32"

# === LOAD MODEL ===
print("🔄 Loading model...")
model = whisperx.load_model("small", device, compute_type=compute_type)

# === TRANSCRIBE ===
print("📝 Transcribing audio...")
transcription_result = model.transcribe(audio_file)

# === LOAD ALIGNMENT MODEL ===
print("🎯 Loading alignment model...")
alignment_model, metadata = whisperx.load_align_model(
    language_code=transcription_result["language"], device=device
)

# === ALIGN ===
print("📌 Aligning words...")
aligned_result = whisperx.align(
    transcription_result["segments"], alignment_model, metadata, audio_file, device
)

# === SAVE JSON ===
with open("aligned.json", "w", encoding="utf-8") as f_json:
    json.dump(aligned_result, f_json, ensure_ascii=False, indent=2)

# === SAVE PLAIN TEXT TRANSCRIPT ===
with open("transcript.txt", "w", encoding="utf-8") as f_txt:
    for segment in aligned_result["segments"]:
        f_txt.write(segment.get("text", "") + "\n")

# === SAVE SRT FILE ===
def to_srt(segments):
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")

    srt_lines = []
    for i, seg in enumerate(segments, 1):
        srt_lines.append(f"{i}")
        srt_lines.append(f"{format_time(seg['start'])} --> {format_time(seg['end'])}")
        srt_lines.append(seg['text'])
        srt_lines.append("")  # Blank line
    return "\n".join(srt_lines)

with open("aligned.srt", "w", encoding="utf-8") as f_srt:
    f_srt.write(to_srt(aligned_result["segments"]))

# === FINAL TRANSCRIPT WITH TIMESTAMPS (TO CONSOLE + FILE) ===
def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.2f}"

print("\n📝 Final transcript with timestamps:\n")

with open("transcript_with_timestamps.txt", "w", encoding="utf-8") as f_timed:
    for seg in aligned_result["segments"]:
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"]
        line = f"[{start} - {end}] {text}"
        print(line)
        f_timed.write(line + "\n")

# === DONE ===
elapsed = time.time() - start_time
print(f"\n✅ Done in {elapsed:.2f} seconds.")
print("📁 Output files saved:")
print(" - aligned.json")
print(" - transcript.txt")
print(" - aligned.srt")
print(" - transcript_with_timestamps.txt")
