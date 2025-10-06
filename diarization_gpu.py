import whisperx
import torch
import warnings
import dotenv
import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

dotenv.load_dotenv()

audio_file = r"recordings\recording_20251003_195741.wav"

model = whisperx.load_model("small", device="cuda")
result = model.transcribe(audio_file)

align_model, align_metadata = whisperx.load_align_model(
    language_code=result["language"], device="cuda"
)

result_aligned = whisperx.align(
    result["segments"],      
    align_model,             
    align_metadata,          
    audio_file,              
    device="cuda"
)

hf_token = os.getenv("HF_TOKEN")

diarize_model = whisperx.diarize.DiarizationPipeline(
    use_auth_token=hf_token,
    device=torch.device("cuda")
)

diarize_segments = diarize_model(audio_file)

result_diarized = whisperx.assign_word_speakers(
    diarize_segments,
    result_aligned
)


with open("diarized_transcripts/transcription_output.txt", "w", encoding="utf-8") as f:
    for seg in result_diarized["segments"]:
        speaker = seg.get('speaker', 'UNKNOWN')
        f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {speaker}: {seg['text']}\n")

print("Transcription saved to transcription_output.txt")