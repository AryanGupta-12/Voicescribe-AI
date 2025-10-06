from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import torch
import asyncio
import os
import uuid
from datetime import datetime
import soundfile as sf
import numpy as np
import io
from pydub import AudioSegment
import whisperx
from dotenv import load_dotenv
import warnings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_realtime = whisper.load_model("tiny")
RECORDINGS_DIR = "recordings"
DIARIZED_DIR = "diarized_transcripts"
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(DIARIZED_DIR, exist_ok=True)

audio_buffers = {}
full_transcript = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    audio_buffers[session_id] = []
    full_transcript[session_id] = ""
    
    try:
        while True:
            data = await websocket.receive()
            
            if "text" in data:
                break
            
            if "bytes" in data:
                audio_buffers[session_id].append(data["bytes"])
                
                if len(audio_buffers[session_id]) >=5 :
                    try:

                        audio_data = b''.join(audio_buffers[session_id])
                        
                        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        if len(audio_array) >= 12000:
                            result = model_realtime.transcribe(
                                audio_array, 
                                fp16=False, 
                                language="en",
                                beam_size=1,
                                best_of=1,
                                temperature=0.0,
                                condition_on_previous_text=False  
                            )
                            text = result["text"].strip()
                            
                            if text:
                                await websocket.send_json({
                                    "transcript": text,
                                    "type": "current"
                                })
                            
                            audio_buffers[session_id] = []
                    except Exception as e:
                        print(f"Transcription error: {e}")
                        import traceback
                        traceback.print_exc()
                        
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if session_id in audio_buffers:
            del audio_buffers[session_id]
        if session_id in full_transcript:
            del full_transcript[session_id]

@app.post("/save")
async def save_recording(audio: UploadFile = File(...)):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    webm_filename = f"recording_{timestamp}.webm"
    wav_filename = f"recording_{timestamp}.wav"
    transcript_filename = f"transcript_{timestamp}.txt"
    
    webm_path = os.path.join(RECORDINGS_DIR, webm_filename)
    wav_path = os.path.join(RECORDINGS_DIR, wav_filename)
    transcript_path = os.path.join(RECORDINGS_DIR, transcript_filename)
    
    audio_data = await audio.read()
    with open(webm_path, "wb") as f:
        f.write(audio_data)
    
    try:
        
        audio_segment = AudioSegment.from_file(webm_path, format="webm")
        audio_segment.export(wav_path, format="wav")

        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        load_dotenv()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        whisperx_model = whisperx.load_model("small", device=device, compute_type=compute_type)
        whisperx_result = whisperx_model.transcribe(wav_path)

        basic_transcript = " ".join([seg["text"] for seg in whisperx_result["segments"]])
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(basic_transcript)
        
        align_model, align_metadata = whisperx.load_align_model(
            language_code=whisperx_result["language"], device=device
        )

        result_aligned = whisperx.align(
            whisperx_result["segments"],
            align_model,
            align_metadata,
            wav_path,
            device=device
        )

        hf_token = os.getenv("HF_TOKEN")
        
        if hf_token:
            diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=hf_token,
                device=torch.device(device)
            )
            
            diarize_segments = diarize_model(wav_path)
            
            result_diarized = whisperx.assign_word_speakers(
                diarize_segments,
                result_aligned
            )

            diarized_filename = "diarized_"+transcript_filename
            diarized_path = os.path.join(DIARIZED_DIR, diarized_filename)

            with open(diarized_path, "w", encoding="utf-8") as f:
                for seg in result_diarized["segments"]:
                    speaker = seg.get('speaker', 'UNKNOWN')
                    f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {speaker}: {seg['text']}\n")
        else:
            print("HF_TOKEN not found, skipping diarization")
        
            
        
    except Exception as e:
        transcript = f"Error during transcription: {e}"
        print(transcript)
        import traceback
        traceback.print_exc()
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
    
    return {
        "audio_file": wav_filename,
        "diarized_file": diarized_filename
    }

@app.get("/download/audio/{filename}")
async def download_audio(filename: str):
    file_path = os.path.join(RECORDINGS_DIR, filename)
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

@app.get("/download/diarized/{filename}")
async def download_diarized(filename: str):
    file_path = os.path.join(DIARIZED_DIR, filename)
    return FileResponse(file_path, media_type="text/plain", filename=filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)