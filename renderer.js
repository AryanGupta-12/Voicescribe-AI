const { ipcRenderer } = require('electron');
const activeWin = require('active-win');
const overviewContent = document.getElementById('overviewContent');

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
const viewFullSummaryBtn = document.getElementById('viewFullSummaryBtn');
const projectSelect = document.getElementById('projectSelect');
const meetingTitleDiv = document.getElementById('meetingTitle');

startBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

// CONTROL SOCKET: receives start/stop from backend presence poller
let controlSocket = null;

async function getMeetingTitle() {
  try {
    const win = await activeWin();
    if (win && win.title && win.title.includes("Microsoft Teams")) {
      // Example title: "Project Orion – Microsoft Teams"
      return win.title.replace(" - Microsoft Teams", "").replace("– Microsoft Teams", "").trim();
    }
  } catch (err) {
    console.error("Failed to get meeting title:", err);
  }
  return "Unknown_Meeting";
}

function normalizeName(s) {
  if (!s) return "";
  return s.toString().toLowerCase().replace(/[^\w\s]/g, "").replace(/\s+/g, " ").trim();
}

async function fetchProjects() {
  console.log("🔎 Fetching project list...");
  try {
    const resp = await fetch('http://localhost:8000/projects');
    const data = await resp.json();
    const projects = (data.projects || []).map(p => p.name);
    populateProjectDropdown(projects);
  } catch (err) {
    console.error("Failed to fetch projects:", err);
    populateProjectDropdown([]);
  }
}

function populateProjectDropdown(projects) {
  projectSelect.innerHTML = "";
  const defaultOption = document.createElement('option');
  defaultOption.value = "";
  defaultOption.textContent = "-- Select project --";
  projectSelect.appendChild(defaultOption);

  for (const p of projects) {
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = p;
    projectSelect.appendChild(opt);
  }

  // restore previous selection if any
  const saved = localStorage.getItem("selectedProject");
  if (saved) {
    const match = Array.from(projectSelect.options).find(o => o.value === saved);
    if (match) projectSelect.value = saved;
  }
}

projectSelect.addEventListener('change', () => {
  const val = projectSelect.value;
  localStorage.setItem("selectedProject", val);
});

// Try to match meeting title to a project list (simple fuzzy using includes)
function findMatchingProject(meetingTitle, projects) {
  const nMeeting = normalizeName(meetingTitle);
  if (!nMeeting) return null;
  for (const p of projects) {
    const np = normalizeName(p);
    if (!np) continue;
    if (np === nMeeting) return p;
    if (np.includes(nMeeting) || nMeeting.includes(np)) return p;
  }
  // secondary pass: token overlap
  const mtokens = new Set(nMeeting.split(" "));
  let best = null; let bestScore = 0;
  for (const p of projects) {
    const np = normalizeName(p);
    const ptokens = np.split(" ");
    let score = 0;
    for (const t of ptokens) if (mtokens.has(t)) score++;
    if (score > bestScore) { bestScore = score; best = p; }
  }
  if (bestScore >= 1) return best;
  return null;
}

// Create project on server
async function createProjectOnServer(name) {
  try {
    const resp = await fetch('http://localhost:8000/projects/create', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ name })
    });
    return await resp.json();
  } catch (err) {
    console.error("Failed to create project:", err);
    return { error: err.message };
  }
}

async function ensureProjectForMeeting(meetingTitle) {
  // load current project list
  let resp, data;
  try {
    resp = await fetch('http://localhost:8000/projects');
    data = await resp.json();
  } catch (err) {
    console.error("Failed to fetch projects:", err);
    data = { projects: [] };
  }
  const projects = (data.projects || []).map(p => p.name);
  // try match
  const match = findMatchingProject(meetingTitle, projects);
  if (match) {
    projectSelect.value = match;
    localStorage.setItem("selectedProject", match);
    return match;
  }
  // no match -> create new
  const createResp = await createProjectOnServer(meetingTitle);
  if (createResp && createResp.project && createResp.project.name) {
    // refresh projects and select
    await fetchProjects();
    projectSelect.value = createResp.project.name;
    localStorage.setItem("selectedProject", createResp.project.name);
    return createResp.project.name;
  } else {
    // fallback: use raw meeting title (sanitization will occur server-side too)
    projectSelect.value = meetingTitle;
    localStorage.setItem("selectedProject", meetingTitle);
    return meetingTitle;
  }
}

function setupControlSocket() {
  try {
    controlSocket = new WebSocket('ws://localhost:8000/ws/control');

    controlSocket.onopen = () => {
      console.log("Control socket connected");
    };

    controlSocket.onmessage = async (evt) => {
      try {
        const payload = JSON.parse(evt.data);
        if (payload && payload.command) {
          console.log("Control command received:", payload.command);
          if (payload.command === 'start') {
            // only start if not already recording
            // but first, detect meeting title and match project
            try {
              const mt = await getMeetingTitle();
              meetingTitleDiv.textContent = mt;
              // try to match/create project in background (non-blocking start)
              ensureProjectForMeeting(mt).then((projName) => {
                console.log("Project selected for meeting:", projName);
              }).catch(err => console.error(err));

            } catch (e) {
              console.error("Error during meeting detection:", e);
            }

            if (!isRecording) {
              startRecording();
            } else {
              console.log("Already recording — ignoring start command");
            }
          } else if (payload.command === 'stop') {
            if (isRecording) {
              stopRecording();
            } else {
              console.log("Not recording — ignoring stop command");
            }
          }
        }
      } catch (err) {
        console.error("Invalid control message", err);
      }
    };

    controlSocket.onclose = () => {
      console.log("Control socket closed — will retry in 5s");
      setTimeout(setupControlSocket, 5000);
    };

    controlSocket.onerror = (err) => {
      console.error("Control socket error:", err);
      controlSocket.close();
    };
  } catch (e) {
    console.error("Failed to create control socket:", e);
  }
}

// Call setup on load
window.addEventListener('DOMContentLoaded', async () => {
  setupControlSocket();
  await fetchProjects();

  // set meeting title display from localStorage if present
  const savedTitle = localStorage.getItem("currentMeetingName");
  if (savedTitle) meetingTitleDiv.textContent = savedTitle;
});

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
    
    // Capture current Teams meeting title (if any)
    const meetingTitle = await getMeetingTitle();
    localStorage.setItem("currentMeetingName", meetingTitle);
    meetingTitleDiv.textContent = meetingTitle;
    console.log("🎯 Current Meeting:", meetingTitle);


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
      // Use selected project if available else stored meeting title
      const selectedProject = projectSelect.value || localStorage.getItem("currentMeetingName") || "Unknown_Meeting";
      const meetingTitle = localStorage.getItem("currentMeetingName") || selectedProject;
      const response = await fetch(
        `http://localhost:8000/save?session_id=${sessionId}&meeting_name=${encodeURIComponent(meetingTitle)}`,
        { method: 'POST' }
      );
    
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

async function loadProjectOverview(projectName) {
  try {
    const resp = await fetch(`http://localhost:8000/project/overview?name=${encodeURIComponent(projectName)}`);
    const data = await resp.json();
    if (data && data.project_name) {
      overviewContent.innerHTML = `
        <h3>${data.project_name}</h3>
        <p><strong>Action Items:</strong> ${window.marked.parse(data.action_items || 'N/A')}</p>
        <p><strong>Gaps:</strong> ${window.marked.parse(data.gaps || 'N/A')}</p>
        <p><strong>Questions:</strong> ${window.marked.parse(data.questions || 'N/A')}</p>
        <p><strong>Status:</strong> ${window.marked.parse(data.current_status || 'N/A')}</p>
        <p><strong>Next Steps:</strong> ${window.marked.parse(data.next_steps || 'N/A')}</p>
        <p><strong>Last Updated:</strong> ${new Date(data.last_updated).toLocaleString()}</p>
      `;
    } else {
      overviewContent.innerHTML = `<p>No overview available for this project.</p>`;
    }
  } catch (err) {
    overviewContent.innerHTML = `<p style="color:red;">Failed to load project overview.</p>`;
    console.error(err);
  }
}

projectSelect.addEventListener('change', (e) => {
  const selected = e.target.value;
  const viewBtn = document.getElementById('viewFullSummaryBtn');
  
  if (!selected) {
    overviewContent.innerHTML = `<p>Select a project to view details.</p>`;
    viewBtn.classList.remove('active');
    viewBtn.disabled = true;
    return;
  }

  viewBtn.classList.add('active');
  viewBtn.disabled = false;
  setTimeout(() => loadProjectOverview(selected), 300);
});


viewFullSummaryBtn.addEventListener('click', async () => {
  const selected = projectSelect.value;
  if (!selected) {
    alert("Please select a project first.");
    return;
  }

  try {
    const resp = await fetch(`http://localhost:8000/project/full_report?name=${encodeURIComponent(selected)}`);
    const data = await resp.json();
    if (data.error) {
      alert(data.error);
      return;
    }

    // Create modal or simple popup window to display full markdown
    const mdWindow = window.open("", "_blank", "width=900,height=700,scrollbars=yes");
    mdWindow.document.write(`
      <html>
      <head>
        <title>${data.name} – Full Summary</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/github-markdown-css/github-markdown.min.css">
        <style>body{margin:20px;font-family:Inter,Segoe UI,sans-serif;}pre{background:#f7f7f7;padding:10px;}</style>
      </head>
      <body class="markdown-body">
        ${window.marked.parse(data.content)}
      </body>
      </html>
    `);
    mdWindow.document.close();
  } catch (err) {
    console.error("Failed to load full summary:", err);
    alert("Failed to load full summary.");
  }
});
