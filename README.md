# 🧠 Meeting AI Recorder

An intelligent Teams-integrated meeting recorder and summarizer that automatically detects your **meeting presence**, records meetings, transcribes audio, performs speaker diarization, and generates **AI-driven meeting insights** using **Llama3.2 (via Ollama)**.

---

## 🚀 Features

- 🎙️ **Automatic meeting detection** — starts/stops recording based on your Microsoft Teams presence.
- 🧩 **Real-time transcription** with Whisper & WhisperX alignment.
- 🗣️ **Speaker diarization** using PyAnnote (with `HF_TOKEN`).
- 🤖 **Insight generation** via **Llama3.2** using Ollama (local LLM).
- 💾 **Incremental insight reports** (non-repetitive) stored per project.
- 🧠 **Project-based summaries** accessible through a simple web frontend.
- 🔐 **Silent Microsoft login** after first sign-in.

---

## ⚙️ Requirements

- **Python 3.12.x** (⚠️ *3.13+ causes dependency conflicts*)
- **Node.js + npm**
- **Ollama** installed locally (for running `llama3.2` model)
- Microsoft Azure Entra ID access for Teams presence monitoring

---

## 🧩 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone <your_repo_url>
cd <your_repo_name>
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate    # on Windows
# or
source .venv/bin/activate   # on Linux/Mac
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Prepare the `.env` file

Create a `.env` file in the project root with the following content:

```ini
HF_TOKEN=
AZURE_CLIENT_ID=
MONITOR_USER_UPN=         # Microsoft Teams user UPN (your Teams account email)
POLL_INTERVAL_SECONDS=5
REDIRECT_URI=http://localhost:8000/auth/callback
AUTHORITY=https://login.microsoftonline.com/common
MONITOR_SCOPE=https://graph.microsoft.com/Presence.Read
```

---

## 🧠 Llama3.2 Setup via Ollama

Install and run **Ollama** (https://ollama.ai).  
Then pull the model:
```bash
ollama pull llama3.2
```

The backend automatically connects to `http://localhost:11434` by default.

---

## 🧩 Running the Application

### Backend (FastAPI + MS Presence Monitor)
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

When you first run the backend, it will open a Microsoft login page.  
After the first successful login, silent login is handled automatically.

---

### Frontend (Electron / Node Interface)

```bash
npm install
npm start
```

This launches the desktop app interface where you can:
- View active project overview.
- Start/stop recording manually (or let presence detection control it).
- View diarized transcripts and insights.

---

## 🧠 AI Insight Generation

- Insights are generated automatically after each recorded meeting using **Llama3.2**.
- Incremental logic ensures *non-repetitive updates* per project.
- You can manually trigger insight generation for testing using:
  ```bash
  POST /test_insight
  ```
  Example:
  ```bash
  curl -X POST "http://localhost:8000/test_insight"        -d "project_name=Farm2Door"        -d "transcript_path=sample.txt"
  ```

---

## 📁 Directory Overview

```
📂 recordings/                - raw meeting recordings
📂 diarized_transcripts/      - diarized transcripts grouped by meeting
  └── 📂 <project_name>/
       ├── diarized_*.txt
       ├── insights/
       │    ├── insight_*.md
       │    └── latest.md
       └── project_state.json
📜 main.py                    - backend API and presence logic
📜 insights_generator.py       - LLM insight pipeline
📜 renderer.js / index.html    - frontend logic
```

---

## 💡 How It Works

1. The app monitors your **Teams presence** via Microsoft Graph API.  
2. When a meeting starts → recording begins automatically.  
3. Audio is transcribed, diarized, and analyzed for key decisions and updates.  
4. **Llama3.2** (via Ollama) generates an incremental insight report per project.  
5. The UI displays the latest summary and allows viewing full reports.

---

## 🧪 Quick Tips

- If diarization fails, a basic transcript is still saved.
- You can test insight logic separately using `/test_insight` endpoint.
- All data is stored locally under `diarized_transcripts/`.

---

## 🏁 Done!

After first setup, everything works automatically:  
✅ Detect meeting → 🎙️ Record → 🧠 Summarize → 📄 Generate Insights.

---

