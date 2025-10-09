const { ipcRenderer } = require('electron');

let mediaRecorder;
let audioChunks = [];
let socket;
let isRecording = false;
let audioContext;
let processor;
let sessionId = null;

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const transcriptBox = document.getElementById('transcript');
const downloadSection = document.getElementById('downloadSection');
const audioLink = document.getElementById('audioLink');
const diarizedLink = document.getElementById('diarizedLink');
const statusText = document.getElementById('status');

startBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

async function startRecording() {
  try {
    audioChunks = [];
    transcriptBox.value = '';
    downloadSection.style.display = 'none';
    
    const sources = await ipcRenderer.invoke('get-sources');
    const systemSource = sources[0];
    
    const systemStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: 'desktop',
          chromeMediaSourceId: systemSource.id
        }
      },
      video: {
        mandatory: {
          chromeMediaSource: 'desktop',
          chromeMediaSourceId: systemSource.id
        }
      }
    });

    audioContext = new AudioContext({ sampleRate: 16000 });
    const systemAudioSource = audioContext.createMediaStreamSource(systemStream);
    
    const destination = audioContext.createMediaStreamDestination();
    systemAudioSource.connect(destination);

    mediaRecorder = new MediaRecorder(destination.stream, {
      mimeType: 'audio/webm'
    });

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    socket = new WebSocket('ws://localhost:8000/ws');
    
    socket.onopen = () => {
      statusText.textContent = 'Recording...';
      isRecording = true;
      startBtn.disabled = true;
      stopBtn.disabled = false;
    
      processor = audioContext.createScriptProcessor(4096, 1, 1);
      
      processor.onaudioprocess = (e) => {
        if (socket.readyState === WebSocket.OPEN) {
          const inputData = e.inputBuffer.getChannelData(0);
          const int16Data = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            int16Data[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
          }
          socket.send(int16Data.buffer);
        }
      };
    
      systemAudioSource.connect(processor);
      processor.connect(audioContext.destination);
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      // Capture session_id
      if (data.type === 'session_id') {
        sessionId = data.session_id;
        console.log('Session ID:', sessionId);
        return;
      }
      
      // Handle transcript
      if (data.transcript) {
        transcriptBox.value = data.transcript;
        transcriptBox.scrollTop = transcriptBox.scrollHeight;
      }
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    mediaRecorder.start(100);
  } catch (err) {
    statusText.textContent = 'Error: ' + err.message;
    console.error(err);
  }
}

async function stopRecording() {
  if (!isRecording) return;

  isRecording = false;
  statusText.textContent = 'Processing...';
  startBtn.disabled = false;
  stopBtn.disabled = true;

  if (processor) {
    processor.disconnect();
    processor = null;
  }

  mediaRecorder.stop();

  setTimeout(async () => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ action: 'stop' }));
      socket.close();
    }
    
    if (audioContext) {
      await audioContext.close();
    }

    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    try {
      const response = await fetch(`http://localhost:8000/save?session_id=${sessionId}`, {
        method: 'POST'
      });
    
      const result = await response.json();
      
      if (result.error) {
        statusText.textContent = 'Error: ' + result.error;
        return;
      }
      
      audioLink.href = `http://localhost:8000/download/audio/${result.audio_file}`;
      diarizedLink.href = `http://localhost:8000/download/diarized/${result.diarized_file}`;
      
      downloadSection.style.display = 'block';
      statusText.textContent = 'Recording Complete';
    } catch (err) {
      statusText.textContent = 'Error saving: ' + err.message;
      console.error(err);
    }
  }, 500);
}