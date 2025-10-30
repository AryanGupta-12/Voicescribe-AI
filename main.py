from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import torch
import os
import uuid
import numpy as np
import whisperx
from dotenv import load_dotenv
import warnings
from dual_audio_recorder import DualAudioRecorder
import os
import asyncio
import httpx
from fastapi import WebSocket, WebSocketDisconnect, Request
from fastapi.responses import RedirectResponse
from msal import PublicClientApplication
from typing import List
import msal
import atexit


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

active_recorders = {}
audio_buffers = {}
full_transcript = {}

# Control websocket connections will be stored here
control_connections: List[WebSocket] = []

# Presence monitoring config
load_dotenv()
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
print("client: ",CLIENT_ID)

MONITOR_USER_UPN = os.getenv("MONITOR_USER_UPN")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))
AUTHORITY = os.getenv("AUTHORITY", "https://login.microsoftonline.com/common")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8000/auth/callback")
SCOPES = [os.getenv("MONITOR_SCOPE", "https://graph.microsoft.com/Presence.Read")]

# Path to store token cache
cache_file = "msal_cache.bin"

token_cache = msal.SerializableTokenCache()
if os.path.exists(cache_file):
    token_cache.deserialize(open(cache_file, "r").read())

msal_app = msal.PublicClientApplication(
    CLIENT_ID,
    authority=AUTHORITY,
    token_cache=token_cache
)

atexit.register(lambda: open(cache_file, "w").write(token_cache.serialize()) if token_cache.has_state_changed else None)
print("🧩 MSAL cache loaded:", os.path.exists(cache_file))
user_accounts = []     # Will hold signed-in user accounts

@app.websocket("/ws/control")
async def control_ws(websocket: WebSocket):
    """
    Persistent control websocket for renderer to receive start/stop commands.
    Renderer should connect on load.
    """
    await websocket.accept()
    control_connections.append(websocket)
    try:
        while True:
            # Keep connection alive by reading; renderer won't send anything normally.
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        control_connections.remove(websocket)
    except Exception:
        if websocket in control_connections:
            control_connections.remove(websocket)

async def broadcast_control_command(cmd: str):
    """
    cmd: 'start' or 'stop'
    Sends a JSON message to all connected control websocket clients.
    """
    dead = []
    for ws in list(control_connections):
        try:
            await ws.send_json({"command": cmd})
        except Exception:
            # mark for removal if connection dead
            dead.append(ws)
    for ws in dead:
        if ws in control_connections:
            control_connections.remove(ws)

@app.get("/login")
async def login(request: Request):

    accounts = msal_app.get_accounts()
    if accounts:
        result = msal_app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            global token_cache
            open(cache_file, "w").write(token_cache.serialize())
            print("🔁 Silent login successful for", result.get("id_token_claims", {}).get("name"))
            asyncio.create_task(poll_presence_loop())
            return {"message": "Silent login successful! Presence poller started."}

    auth_url = msal_app.get_authorization_request_url(
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )
    return RedirectResponse(auth_url)


@app.get("/auth/callback")
async def auth_callback(request: Request):
    # Step 2: exchange auth code for token
    code = request.query_params.get("code")
    if not code:
        return {"error": "Missing authorization code"}

    result = msal_app.acquire_token_by_authorization_code(
        code,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

    if "access_token" in result:
        global token_cache
        open(cache_file, "w").write(token_cache.serialize())
        print("✅ Login successful for", result.get("id_token_claims", {}).get("name"))
        asyncio.create_task(poll_presence_loop())
        return {"message": "Login successful! Presence poller started."}
    else:
        print("Login failed:", result)
        return {"error": "Login failed", "details": result}

async def poll_presence_loop():
    """
    Poll Microsoft Graph using delegated (user) token for /me/presence.
    """
    accounts = msal_app.get_accounts()
    if not accounts:
        print("⚠️ No account found in MSAL cache — please sign in via /login endpoint.")
        return

    result = msal_app.acquire_token_silent(SCOPES, account=accounts[0])
    if not result or "access_token" not in result:
        print("⚠️ No valid token found — please sign in via /login endpoint.")
        return

    headers = {"Authorization": f"Bearer {result['access_token']}"}
    meeting_active = False

    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            try:
                resp = await client.get("https://graph.microsoft.com/v1.0/me/presence", headers=headers)
                if resp.status_code == 401:
                    print("Token expired. Please re-login via /login.")
                    break  # Stop polling until user logs in again
                resp.raise_for_status()
                data = resp.json()
                activity = data.get("activity", "")
                availability = data.get("availability", "")
                print(f"Presence: {activity} ({availability})")

                is_meeting = any(word in activity.lower() for word in ["meeting", "call", "conference", "present"])
                if is_meeting and not meeting_active:
                    print("🎙️ Detected meeting start.")
                    await broadcast_control_command("start")
                    meeting_active = True
                elif not is_meeting and meeting_active:
                    print("🛑 Detected meeting end.")
                    await broadcast_control_command("stop")
                    meeting_active = False
            except Exception as e:
                print("Presence poller error:", e)

            await asyncio.sleep(POLL_INTERVAL_SECONDS)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    audio_buffers[session_id] = []
    full_transcript[session_id] = ""

    recorder = DualAudioRecorder()
    active_recorders[session_id] = recorder

    await websocket.send_json({
        "type": "session_id",
        "session_id": session_id
    })
    
    try:
        recorder.start_recording()
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
        if session_id in active_recorders:
            recorder = active_recorders[session_id]
            recorder.stop_recording()
            merged_path = recorder.merge_audio_tracks(
                system_gain_db=-2,  
                mic_gain_db=2
            )
            recorder.cleanup()

        if session_id in audio_buffers:
            del audio_buffers[session_id]

        if session_id in full_transcript:
            del full_transcript[session_id]

@app.post("/save")
async def save_recording(session_id: str):
    if session_id not in active_recorders:
        return {"error": "Session not found"}
    
    recorder = active_recorders[session_id]
    
    import time
    time.sleep(0.5)
    
    wav_path = recorder.merged_path
    timestamp = recorder.timestamp
    
    if not wav_path or not os.path.exists(wav_path):
        return {"error": "Merged audio file not found. Recording may not have completed properly."}
    
    transcript_filename = f"transcript_{timestamp}.txt"
    diarized_filename = f"diarized_{timestamp}.txt"
    
    transcript_path = os.path.join(RECORDINGS_DIR, transcript_filename)
    diarized_path = os.path.join(DIARIZED_DIR, diarized_filename)
    
    try:
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
            try:
                print("🗣️  Starting PyAnnote diarization...")
                
                from pyannote.audio import Pipeline
                
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                diarization_pipeline.to(torch.device(device))
                
                # Run diarization
                diarization = diarization_pipeline(wav_path)
                
                # Collect speaker segments with overlap detection
                speaker_segments = {}
                overlaps = []
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if speaker not in speaker_segments:
                        speaker_segments[speaker] = []
                    speaker_segments[speaker].append({
                        'start': turn.start,
                        'end': turn.end
                    })
                
                # Detect overlaps
                for segment in diarization.get_overlap():
                    overlapping_speakers = set()
                    for turn, _, speaker in diarization.crop(segment).itertracks(yield_label=True):
                        overlapping_speakers.add(speaker)
                    if len(overlapping_speakers) >= 2:
                        overlaps.append({
                            'start': segment.start,
                            'end': segment.end,
                            'speakers': list(overlapping_speakers)
                        })
                
                # Assign speakers to Whisper segments
                diarized_segments = []
                
                for seg in result_aligned["segments"]:
                    seg_start = seg['start']
                    seg_end = seg['end']
                    seg_mid = (seg_start + seg_end) / 2
                    
                    # Find which speaker was talking at segment midpoint
                    assigned_speaker = "UNKNOWN"
                    for speaker, turns in speaker_segments.items():
                        for turn in turns:
                            if turn['start'] <= seg_mid <= turn['end']:
                                assigned_speaker = speaker
                                break
                        if assigned_speaker != "UNKNOWN":
                            break
                    
                    diarized_segments.append({
                        'start': seg_start,
                        'end': seg_end,
                        'text': seg['text'],
                        'speaker': assigned_speaker
                    })
                
                # Write diarized transcript
                with open(diarized_path, "w", encoding="utf-8") as f:
                    f.write("=== SPEAKER DIARIZATION TRANSCRIPT ===\n")
                    f.write(f"Total speakers detected: {len(speaker_segments)}\n")
                    f.write(f"Overlapping speech segments: {len(overlaps)}\n\n")
                    
                    # Write overlaps if any
                    if overlaps:
                        f.write("⚠️  OVERLAPPING SPEECH DETECTED:\n")
                        for overlap in overlaps:
                            speakers_str = " + ".join(sorted(overlap['speakers']))
                            f.write(f"[{overlap['start']:.1f}s - {overlap['end']:.1f}s] → {speakers_str}\n")
                        f.write("\n")
                    
                    # Write transcript with speakers
                    f.write("=== TRANSCRIPT ===\n\n")
                    for seg in diarized_segments:
                        f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']}: {seg['text']}\n")
                
                print(f"✅ Enhanced diarized transcript saved to: {diarized_path}")
                
            except Exception as diarize_error:
                print(f"❌ Diarization error: {diarize_error}")
                import traceback
                traceback.print_exc()
                
                # Fallback: basic transcript without speakers
                with open(diarized_path, "w", encoding="utf-8") as f:
                    f.write("⚠️ Diarization failed. Basic transcript without speaker labels:\n\n")
                    for seg in result_aligned["segments"]:
                        f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}\n")
        else:
            print("❌ HF_TOKEN not found, skipping diarization")
            
            with open(diarized_path, "w", encoding="utf-8") as f:
                f.write("⚠️ No HF_TOKEN found. Basic transcript without speaker labels:\n\n")
                for seg in result_aligned["segments"]:
                    f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}\n")
            
    except Exception as e:
        transcript = f"Error during transcription: {e}"
        print(transcript)
        import traceback
        traceback.print_exc()
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
    
    if session_id in active_recorders:
        del active_recorders[session_id]
        
    return {
        "audio_file": f"merged_{timestamp}.wav",
        "transcript_file": transcript_filename,
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

async def try_silent_login():
    accounts = msal_app.get_accounts()
    if accounts:
        result = msal_app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            print("🔁 Silent login successful for", result.get("id_token_claims", {}).get("name"))
            asyncio.create_task(poll_presence_loop())
            return True
    print("⚠️ Silent login failed or no cached account found.")
    return False

@app.on_event("startup")
async def startup_event():
    print("🚀 App started. Please open http://localhost:8000/login to sign in.")
    success = await try_silent_login()
    if not success:
        print("🔒 Please open http://localhost:8000/login to sign in.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)