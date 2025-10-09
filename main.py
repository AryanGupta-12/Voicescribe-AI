# from fastapi import FastAPI, WebSocket, UploadFile, File
# from fastapi.responses import FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# import whisper
# import torch
# import os
# import uuid
# from datetime import datetime
# import numpy as np
# from pydub import AudioSegment
# import whisperx
# from dotenv import load_dotenv
# import warnings
# from dual_audio_recorder import DualAudioRecorder

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# model_realtime = whisper.load_model("tiny")
# RECORDINGS_DIR = "recordings"
# DIARIZED_DIR = "diarized_transcripts"
# os.makedirs(RECORDINGS_DIR, exist_ok=True)
# os.makedirs(DIARIZED_DIR, exist_ok=True)

# active_recorders = {}
# audio_buffers = {}
# full_transcript = {}

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     session_id = str(uuid.uuid4())
#     audio_buffers[session_id] = []
#     full_transcript[session_id] = ""

#     recorder = DualAudioRecorder()
#     active_recorders[session_id] = recorder
    
#     # Send session_id to frontend immediately after connection
#     await websocket.send_json({
#         "type": "session_id",
#         "session_id": session_id
#     })
    
#     try:
#         recorder.start_recording()
#         while True:
#             data = await websocket.receive()
            
#             if "text" in data:
#                 break
            
#             if "bytes" in data:
#                 audio_buffers[session_id].append(data["bytes"])
                
#                 if len(audio_buffers[session_id]) >=5 :
#                     try:

#                         audio_data = b''.join(audio_buffers[session_id])
                        
#                         audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        
#                         if len(audio_array) >= 12000:
#                             result = model_realtime.transcribe(
#                                 audio_array, 
#                                 fp16=False, 
#                                 language="en",
#                                 beam_size=1,
#                                 best_of=1,
#                                 temperature=0.0,
#                                 condition_on_previous_text=False  
#                             )
#                             text = result["text"].strip()
                            
#                             if text:
#                                 await websocket.send_json({
#                                     "transcript": text,
#                                     "type": "current"
#                                 })
                            
#                             audio_buffers[session_id] = []
#                     except Exception as e:
#                         print(f"Transcription error: {e}")
#                         import traceback
#                         traceback.print_exc()
                        
#     except Exception as e:
#         print(f"WebSocket error: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         if session_id in active_recorders:
#             recorder = active_recorders[session_id]
#             recorder.stop_recording()
#             # Merge the tracks
#             merged_path = recorder.merge_audio_tracks(
#                 system_gain_db=-2,  # Adjust as needed
#                 mic_gain_db=2
#             )
#             recorder.cleanup()
#             # del active_recorders[session_id]

#         if session_id in audio_buffers:
#             del audio_buffers[session_id]

#         if session_id in full_transcript:
#             del full_transcript[session_id]

# @app.post("/save")
# async def save_recording(session_id: str):
#     if session_id not in active_recorders:
#         return {"error": "Session not found"}
    
#     recorder = active_recorders[session_id]
    
#     import time
#     time.sleep(0.5)
    
#     wav_path = recorder.merged_path
#     timestamp = recorder.timestamp
    
#     if not wav_path or not os.path.exists(wav_path):
#         return {"error": "Merged audio file not found. Recording may not have completed properly."}
    
#     transcript_filename = f"transcript_{timestamp}.txt"
#     diarized_filename = f"diarized_{timestamp}.txt"
    
#     transcript_path = os.path.join(RECORDINGS_DIR, transcript_filename)
#     diarized_path = os.path.join(DIARIZED_DIR, diarized_filename)
    
#     try:
#         warnings.filterwarnings("ignore", category=UserWarning)
#         warnings.filterwarnings("ignore", category=FutureWarning)
        
#         load_dotenv()
        
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         compute_type = "float16" if device == "cuda" else "int8"
        
#         whisperx_model = whisperx.load_model("small", device=device, compute_type=compute_type)
#         whisperx_result = whisperx_model.transcribe(wav_path)

#         basic_transcript = " ".join([seg["text"] for seg in whisperx_result["segments"]])
#         with open(transcript_path, "w", encoding="utf-8") as f:
#             f.write(basic_transcript)
        
#         align_model, align_metadata = whisperx.load_align_model(
#             language_code=whisperx_result["language"], device=device
#         )

#         result_aligned = whisperx.align(
#             whisperx_result["segments"],
#             align_model,
#             align_metadata,
#             wav_path,
#             device=device
#         )

#         hf_token = os.getenv("HF_TOKEN")
        
#         if hf_token:
#             diarize_model = whisperx.diarize.DiarizationPipeline(
#                 use_auth_token=hf_token,
#                 device=torch.device(device)
#             )
            
#             diarize_segments = diarize_model(wav_path)
            
#             result_diarized = whisperx.assign_word_speakers(
#                 diarize_segments,
#                 result_aligned
#             )

#             with open(diarized_path, "w", encoding="utf-8") as f:
#                 for seg in result_diarized["segments"]:
#                     speaker = seg.get('speaker', 'UNKNOWN')
#                     f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {speaker}: {seg['text']}\n")
#         else:
#             print("HF_TOKEN not found, skipping diarization")
            
#     except Exception as e:
#         transcript = f"Error during transcription: {e}"
#         print(transcript)
#         import traceback
#         traceback.print_exc()
#         with open(transcript_path, "w", encoding="utf-8") as f:
#             f.write(transcript)
    
#     if session_id in active_recorders:
#         del active_recorders[session_id]

#     return {
#         "audio_file": f"merged_{timestamp}.wav",
#         "transcript_file": transcript_filename,
#         "diarized_file": diarized_filename
#     }

# @app.get("/download/audio/{filename}")
# async def download_audio(filename: str):
#     file_path = os.path.join(RECORDINGS_DIR, filename)
#     return FileResponse(file_path, media_type="audio/wav", filename=filename)

# @app.get("/download/diarized/{filename}")
# async def download_diarized(filename: str):
#     file_path = os.path.join(DIARIZED_DIR, filename)
#     return FileResponse(file_path, media_type="text/plain", filename=filename)


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import torch
import os
import uuid
from datetime import datetime
import numpy as np
from pydub import AudioSegment
import whisperx
from dotenv import load_dotenv
import warnings
from dual_audio_recorder import DualAudioRecorder

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    audio_buffers[session_id] = []
    full_transcript[session_id] = ""

    recorder = DualAudioRecorder()
    active_recorders[session_id] = recorder
    
    # Send session_id to frontend immediately after connection
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
                
                # Use pure PyAnnote pipeline (better than WhisperX diarization)
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
            
            # Write basic transcript without speakers
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)