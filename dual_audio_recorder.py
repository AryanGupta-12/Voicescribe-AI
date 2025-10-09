import pyaudio
import wave
import threading
import numpy as np
from datetime import datetime
import os
from pydub import AudioSegment

class DualAudioRecorder:
    def __init__(self, output_dir="recordings"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        
        # Control flags
        self.is_recording = False
        self.system_thread = None
        self.mic_thread = None
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
        # File paths
        self.timestamp = None
        self.system_path = None
        self.mic_path = None
        self.merged_path = None
        
    def find_audio_devices(self):
        """Find system audio (loopback) and microphone devices"""
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        system_device = None
        mic_device = None
        
        print("\n=== Available Audio Devices ===")
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            device_name = device_info.get('name').lower()
            max_input = device_info.get('maxInputChannels')
            
            print(f"{i}: {device_info.get('name')} (Inputs: {max_input})")
            
            # Detect system audio (loopback/stereo mix)
            if max_input > 0 and any(keyword in device_name for keyword in 
                ['stereo mix', 'loopback', 'wave out', 'what u hear']):
                system_device = i
                print(f"   → Found System Audio Device!")
            
            # Detect microphone
            if max_input > 0 and any(keyword in device_name for keyword in 
                ['microphone', 'mic', 'input']):
                if mic_device is None:  # Get first mic
                    mic_device = i
                    print(f"   → Found Microphone Device!")
        
        print("=" * 35)
        return system_device, mic_device
    
    def record_audio_stream(self, device_index, output_path, stream_name):
        """Record audio from a specific device"""
        frames = []
        
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK
            )
            
            print(f"[{stream_name}] Recording started on device {device_index}")
            
            while self.is_recording:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    print(f"[{stream_name}] Read error: {e}")
                    continue
            
            stream.stop_stream()
            stream.close()
            
            # Save to WAV file
            wf = wave.open(output_path, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            print(f"[{stream_name}] Recording saved to {output_path}")
            
        except Exception as e:
            print(f"[{stream_name}] Error: {e}")
    
    def start_recording(self):
        """Start recording both system and mic audio in parallel"""
        if self.is_recording:
            print("Already recording!")
            return False
        
        # Find devices
        system_device, mic_device = self.find_audio_devices()
        
        if system_device is None:
            print("⚠️  System audio device not found!")
            print("Enable 'Stereo Mix' in Windows Sound Settings:")
            print("   Sound Settings → Sound Control Panel → Recording")
            print("   Right-click → Show Disabled Devices → Enable Stereo Mix")
            return False
        
        if mic_device is None:
            print("⚠️  Microphone not found!")
            return False
        
        # Generate filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.system_path = os.path.join(self.output_dir, f"system_{self.timestamp}.wav")
        self.mic_path = os.path.join(self.output_dir, f"mic_{self.timestamp}.wav")
        self.merged_path = os.path.join(self.output_dir, f"merged_{self.timestamp}.wav")
        
        # Start recording flag
        self.is_recording = True
        
        # Start recording threads
        self.system_thread = threading.Thread(
            target=self.record_audio_stream,
            args=(system_device, self.system_path, "SYSTEM"),
            daemon=True
        )
        
        self.mic_thread = threading.Thread(
            target=self.record_audio_stream,
            args=(mic_device, self.mic_path, "MIC"),
            daemon=True
        )
        
        self.system_thread.start()
        self.mic_thread.start()
        
        print(f"\n🎙️  Recording started!")
        print(f"System: {self.system_path}")
        print(f"Mic: {self.mic_path}")
        return True
    
    def stop_recording(self):
        """Stop recording and wait for threads to finish"""
        if not self.is_recording:
            print("Not recording!")
            return None
        
        print("\n⏹️  Stopping recording...")
        self.is_recording = False
        
        # Wait for threads to finish
        if self.system_thread:
            self.system_thread.join(timeout=2)
        if self.mic_thread:
            self.mic_thread.join(timeout=2)
        
        print("✅ Recording stopped!")
        return self.system_path, self.mic_path
    
    def merge_audio_tracks(self, system_gain_db=0, mic_gain_db=0):
        """
        Merge system and mic audio with volume adjustments
        
        Args:
            system_gain_db: Volume adjustment for system audio (in dB)
            mic_gain_db: Volume adjustment for mic audio (in dB)
        """
        if not os.path.exists(self.system_path) or not os.path.exists(self.mic_path):
            print("❌ Audio files not found!")
            return None
        
        print("\n🔄 Merging audio tracks...")
        
        try:
            # Load audio files
            system_audio = AudioSegment.from_wav(self.system_path)
            mic_audio = AudioSegment.from_wav(self.mic_path)
            
            # Apply gain adjustments
            system_audio = system_audio + system_gain_db
            mic_audio = mic_audio + mic_gain_db
            
            # Ensure same length (pad shorter one with silence)
            max_length = max(len(system_audio), len(mic_audio))
            
            if len(system_audio) < max_length:
                system_audio = system_audio + AudioSegment.silent(
                    duration=max_length - len(system_audio)
                )
            
            if len(mic_audio) < max_length:
                mic_audio = mic_audio + AudioSegment.silent(
                    duration=max_length - len(mic_audio)
                )
            
            # Mix (overlay) the tracks
            merged = system_audio.overlay(mic_audio)
            
            # Export merged audio
            merged.export(self.merged_path, format="wav")
            
            print(f"✅ Merged audio saved to: {self.merged_path}")
            return self.merged_path
            
        except Exception as e:
            print(f"❌ Merge error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup(self):
        """Clean up PyAudio resources"""
        self.audio.terminate()


# Example usage
if __name__ == "__main__":
    recorder = DualAudioRecorder()
    
    try:
        # Start recording
        recorder.start_recording()
        
        # Record for 10 seconds (or until user stops)
        import time
        input("Press Enter to stop recording...")
        
        # Stop recording
        recorder.stop_recording()
        
        # Merge tracks (adjust volumes if needed)
        # Positive values increase volume, negative decrease
        merged_file = recorder.merge_audio_tracks(
            system_gain_db=-3,  # Reduce system audio by 3dB
            mic_gain_db=3       # Boost mic by 3dB
        )
        
        if merged_file:
            print(f"\n🎉 Final merged file: {merged_file}")
        
    finally:
        recorder.cleanup()