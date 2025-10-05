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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("base")
model_realtime = whisper.load_model("tiny")
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

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
        
        result = model.transcribe(wav_path, fp16=False)
        transcript = result["text"]
        
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
            
        
    except Exception as e:
        transcript = f"Error during transcription: {e}"
        print(transcript)
        import traceback
        traceback.print_exc()
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
    
    return {
        "audio_file": wav_filename,
        "transcript_file": transcript_filename
    }

@app.get("/download/audio/{filename}")
async def download_audio(filename: str):
    file_path = os.path.join(RECORDINGS_DIR, filename)
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

@app.get("/download/transcript/{filename}")
async def download_transcript(filename: str):
    file_path = os.path.join(RECORDINGS_DIR, filename)
    return FileResponse(file_path, media_type="text/plain", filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)