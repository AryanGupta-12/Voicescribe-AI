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
import json
import re
from fastapi import Body
from typing import Dict
from insights_generator import generate_insights_from_file
import shutil
from datetime import datetime


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
user_accounts = []  

meeting_metadata = []

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
                    print("🔁 Access token expired — refreshing silently...")
                    result = msal_app.acquire_token_silent(SCOPES, account=accounts[0])
                    if not result or "access_token" not in result:
                        print("⚠️ Silent refresh failed — please log in via /login.")
                        break
                    headers["Authorization"] = f"Bearer {result['access_token']}"
                    continue
                    
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

def extract_project_state(markdown_path: str, json_path: str):
    """Parse the latest insight markdown into structured JSON fields."""
    try:
        with open(markdown_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Simple section extraction using regex headings
        def section(*names):
            """
            Match headings like ## or ### (case-insensitive), optionally prefixed
            with numbering (e.g. "3. Action Items"), and support alternate variations
            like 'New Action Items', 'Updated Action Items', etc.
            Returns the combined text for the first match.
            """
            # combine all heading names into one regex group
            joined = "|".join([re.escape(n) for n in names])
            pattern = rf"(?mi)^#+\s*(\d+\.\s*)?(?:{joined})(?:\s*[:\-–]?\s*)?(.*?)$(.*?)^(?=#+\s|\Z)"
            match = re.search(pattern, content, re.S | re.M)
            return match.group(3).strip() if match else ""


        data = {
            "project_name": os.path.basename(os.path.dirname(os.path.dirname(markdown_path))),
            "project_objective": section("Project Objective"),
            "discussion_summary": section("Discussion Summary", "Key Updates Since Last Meeting"),
            "roadmap_timeline": section("Roadmap", "Roadmap & Timeline"),
            "current_status": section("Current Status", "Status"),
            "requirements": section("Requirements", "Requirements Discussed"),
            "gaps": section("Gaps", "Risks", "New Risks or Blockers"),
            "action_items": section("Action Items", "New Action Items"),
            "key_decisions": section("Key Decisions", "New Decisions"),
            "questions": section("Questions", "Questions & Concerns Raised"),
            "next_steps": section("Next Steps"),
            "changes_since_last_meeting": section("CHANGES SINCE LAST MEETING", "Changes Since Last Meeting"),
            "last_updated": datetime.now().isoformat()
        }

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(data, jf, indent=2)
        print(f"🗂️ project_state.json updated at {json_path}")

    except Exception as e:
        print(f"⚠️ Failed to extract project state: {e}")



@app.post("/save")
async def save_recording(session_id: str, meeting_name: str = None):
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
    
    # === NEW CODE: Group transcript by meeting name (from renderer) ===
    

    if meeting_name:
        # Remove trailing " | Microsoft Teams" or "– Microsoft Teams"
        meeting_name = re.sub(r"\s*[\|\-–]\s*Microsoft Teams", "", meeting_name, flags=re.IGNORECASE)

        # Replace invalid filename characters with underscores
        meeting_name = re.sub(r'[<>:"/\\|?*]', "_", meeting_name)

        # Trim and normalize spaces
        meeting_name = meeting_name.strip().replace("  ", " ")

        participants = []
    elif meeting_metadata and len(meeting_metadata) == 2:
        meeting_name = meeting_metadata[0].replace("/", "_").replace("\\", "_").strip()
        participants = meeting_metadata[1]
    else:
        meeting_name = "Unknown_Meeting"
        participants = []

    print("Meeting Name:", meeting_name)
    grouped_dir = os.path.join(DIARIZED_DIR, meeting_name)
    os.makedirs(grouped_dir, exist_ok=True)

    # Copy and prepend meeting info
    grouped_path = os.path.join(grouped_dir, diarized_filename)
    with open(diarized_path, "r", encoding="utf-8") as src, open(grouped_path, "w", encoding="utf-8") as dest:
        dest.write("=== MEETING INFORMATION ===\n")
        dest.write(f"Meeting Name: {meeting_name}\n")
        if participants:
            dest.write(f"Participants: {', '.join(participants)}\n\n")
        else:
            dest.write("Participants: Unknown\n\n")
        dest.write(src.read())

    print(f"📁 Transcript grouped under: {grouped_dir}")

    # === Generate or update project insights ===
    try:
        project_dir = os.path.join(DIARIZED_DIR, meeting_name)
        insights_dir = os.path.join(project_dir, "insights")
        os.makedirs(insights_dir, exist_ok=True)

        current_transcript = diarized_path
        previous_insight = get_last_insight_path(project_dir)

        if previous_insight and os.path.exists(previous_insight):
            print(f"🧠 Updating insights for existing project: {meeting_name}")
            # Merge old insight and new transcript
            merged_input_path = os.path.join(insights_dir, "_tmp_merged.txt")
            with open(previous_insight, "r", encoding="utf-8") as oldf, \
                open(current_transcript, "r", encoding="utf-8") as newf, \
                open(merged_input_path, "w", encoding="utf-8") as outf:
                outf.write("=== PREVIOUS INSIGHT ===\n")
                outf.write(oldf.read())
                outf.write("\n\n=== NEW TRANSCRIPT ===\n")
                outf.write(newf.read())

            output_name = f"insight_{timestamp}.md"
            output_path = os.path.join(insights_dir, output_name)
            insights = generate_insights_from_file(
                transcript_path=merged_input_path,
                output_path=output_path
            )
            # update latest.md link
            shutil.copy(output_path, os.path.join(insights_dir, "latest.md"))
            os.remove(merged_input_path)
        else:
            print(f"🧠 Generating first insight for project: {meeting_name}")
            output_name = f"insight_{timestamp}.md"
            output_path = os.path.join(insights_dir, output_name)
            insights = generate_insights_from_file(
                transcript_path=current_transcript,
                output_path=output_path
            )
            shutil.copy(output_path, os.path.join(insights_dir, "latest.md"))

        # Convert insight markdown to a summarized project_state.json
        latest_md = os.path.join(insights_dir, "latest.md")
        json_state_path = os.path.join(project_dir, "project_state.json")
        extract_project_state(latest_md, json_state_path)
        print(f"✅ Insight generation complete for project: {meeting_name}")

    except Exception as insight_err:
        print(f"⚠️ Insight generation failed: {insight_err}")

        
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

@app.get("/projects")
async def list_projects():
    projects = []
    try:
        # list directories only
        for name in sorted(os.listdir(DIARIZED_DIR)):
            path = os.path.join(DIARIZED_DIR, name)
            if os.path.isdir(path):
                
                state_file = os.path.join(path, "project_state.json")
                has_state = os.path.exists(state_file)
                projects.append({
                    "name": name,
                    "has_state": has_state
                })
    except Exception as e:
        print("Error listing projects:", e)
        return {"error": str(e), "projects": []}
    return {"projects": projects}

@app.post("/projects/create")
async def create_project(payload: Dict = Body(...)):
    name = payload.get("name")
    if not name:
        return {"error": "Missing project name"}
    # sanitize name similarly to how you sanitize meeting_name
    safe_name = re.sub(r'\s*[\|\-–]\s*Microsoft Teams', "", name, flags=re.IGNORECASE)
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", safe_name).strip()
    if not safe_name:
        return {"error": "Invalid project name after sanitization"}
    path = os.path.join(DIARIZED_DIR, safe_name)
    try:
        os.makedirs(path, exist_ok=True)
        # also create meetings and insights subfolders for better organization
        os.makedirs(os.path.join(path, "meetings"), exist_ok=True)
        os.makedirs(os.path.join(path, "insights"), exist_ok=True)
        return {"status": "created", "project": {"name": safe_name}}
    except Exception as e:
        print("Error creating project:", e)
        return {"error": str(e)}

def get_last_insight_path(project_dir: str) -> str:
    insights_dir = os.path.join(project_dir, "insights")
    if not os.path.exists(insights_dir):
        return None
    md_files = sorted(
        [f for f in os.listdir(insights_dir) if f.endswith(".md")],
        reverse=True
    )
    return os.path.join(insights_dir, md_files[0]) if md_files else None

@app.get("/project/overview")
async def get_project_overview(name: str):
    """
    Returns the latest project_state.json contents for the given project name.
    """
    try:
        project_dir = os.path.join(DIARIZED_DIR, name)
        json_path = os.path.join(project_dir, "project_state.json")

        if not os.path.exists(json_path):
            return {"error": f"No overview found for {name}"}

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["project_name"]=name
        return data

    except Exception as e:
        return {"error": str(e)}


@app.post("/test_insight")
async def test_insight(project_name: str, transcript_path: str):
    """
    Simulate Phase 2 without a real meeting or audio.
    Example POST body:
    {
      "project_name": "Project Orion",
      "transcript_path": "sample_transcript.txt"
    }
    """
    try:
        project_dir = os.path.join(DIARIZED_DIR, project_name)
        os.makedirs(os.path.join(project_dir, "insights"), exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        current_transcript = transcript_path
        previous_insight = get_last_insight_path(project_dir)

        if previous_insight and os.path.exists(previous_insight):
            merged_input_path = os.path.join(project_dir, "insights", "_tmp_merged.txt")
            with open(previous_insight, "r", encoding="utf-8") as oldf, \
                 open(current_transcript, "r", encoding="utf-8") as newf, \
                 open(merged_input_path, "w", encoding="utf-8") as outf:
                outf.write("=== PREVIOUS INSIGHT ===\n")
                outf.write(oldf.read())
                outf.write("\n\n=== NEW TRANSCRIPT ===\n")
                outf.write(newf.read())

            output_path = os.path.join(project_dir, "insights", f"insight_{timestamp}.md")
            generate_insights_from_file(merged_input_path, output_path)
            os.remove(merged_input_path)
        else:
            output_path = os.path.join(project_dir, "insights", f"insight_{timestamp}.md")
            generate_insights_from_file(current_transcript, output_path)

        # Update latest + JSON
        shutil.copy(output_path, os.path.join(project_dir, "insights", "latest.md"))
        extract_project_state(os.path.join(project_dir, "insights", "latest.md"),
                              os.path.join(project_dir, "project_state.json"))
        return {"status": "success", "message": "Insights generated", "output": output_path}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

@app.get("/project/full_report")
async def get_full_report(name: str):
    """
    Returns the raw markdown of the latest insight report (latest.md)
    for the selected project.
    """
    try:
        project_dir = os.path.join(DIARIZED_DIR, name)
        latest_md = os.path.join(project_dir, "insights", "latest.md")

        if not os.path.exists(latest_md):
            return {"error": f"No report found for {name}"}

        with open(latest_md, "r", encoding="utf-8") as f:
            content = f.read()
        return {"name": name, "content": content}

    except Exception as e:
        return {"error": str(e)}


@app.on_event("startup")
async def startup_event():
    print("🚀 App started. Please open http://localhost:8000/login to sign in.")
    success = await try_silent_login()
    if not success:
        print("🔒 Please open http://localhost:8000/login to sign in.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)