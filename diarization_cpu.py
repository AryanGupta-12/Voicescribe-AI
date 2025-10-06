import whisperx
import torch
import warnings
import dotenv
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

dotenv.load_dotenv()

audio_file = r"recordings\recording_20251005_122705.wav"
file_name = audio_file.split("\\")[1].split(".")[0]

model = whisperx.load_model("small", device="cpu", compute_type="int8")
result = model.transcribe(audio_file)

align_model, align_metadata = whisperx.load_align_model(
    language_code=result["language"], device="cpu"
)

result_aligned = whisperx.align(
    result["segments"],      
    align_model,             
    align_metadata,          
    audio_file,              
    device="cpu"
)

hf_token = os.getenv("HF_TOKEN")

diarize_model = whisperx.diarize.DiarizationPipeline(
    use_auth_token=hf_token,
    device=torch.device("cpu")
)

diarize_segments = diarize_model(audio_file)

result_diarized = whisperx.assign_word_speakers(
    diarize_segments,
    result_aligned
)

os.makedirs("diarized_transcripts", exist_ok = True)

with open(f"diarized_transcripts/diarized_output_{file_name}.txt", "w", encoding="utf-8") as f:
    for seg in result_diarized["segments"]:
        speaker = seg.get('speaker', 'UNKNOWN')
        f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {speaker}: {seg['text']}\n")

print("Transcription saved to transcription_output.txt")