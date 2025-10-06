# VoiceScribe AI 🎙️

**AI-Powered Meeting Transcription with Speaker Diarization**

VoiceScribe AI is a desktop application that records system audio and generates intelligent transcripts with speaker identification. Built with Electron.js and FastAPI, it leverages WhisperX for accurate speech recognition and speaker diarization, making it perfect for meetings, interviews, podcasts, and any multi-speaker audio content.

---

## ✨ Features

- **🎯 Real-time Audio Recording**: Capture system audio directly from your computer
- **🤖 AI-Powered Transcription**: Utilizes OpenAI's Whisper model via WhisperX for accurate speech-to-text
- **👥 Speaker Diarization**: Automatically identifies and labels different speakers in the conversation
- **📊 Dual Output Format**: 
  - Basic transcript (plain text, no speaker labels)
  - Diarized transcript (with speaker labels and timestamps)
- **⚡ Real-time Transcription Preview**: Live transcription during recording
- **💾 Easy Export**: Download both audio files and transcripts
- **🔒 Secure**: HuggingFace tokens stored securely in environment variables

---

## 🎯 Use Cases

The diarized transcripts are perfect for various NLP tasks:

- **📝 Meeting Summarization**: Generate concise summaries of lengthy meetings
- **🔍 Action Item Extraction**: Identify tasks and action items assigned to specific speakers
- **📊 Sentiment Analysis**: Analyze speaker sentiment and engagement levels
- **🗂️ Topic Modeling**: Identify key discussion topics and themes
- **📈 Speaker Analytics**: Track speaker participation, talk time, and interaction patterns
- **🔎 Keyword Extraction**: Extract important terms and concepts
- **📚 Content Classification**: Categorize meeting types and discussion topics
- **🎯 Question-Answer Detection**: Identify Q&A patterns in the conversation

---

## 🛠️ Tech Stack

### Frontend
- **Electron.js**: Cross-platform desktop application framework
- **HTML/CSS/JavaScript**: User interface

### Backend
- **FastAPI**: High-performance Python web framework
- **WhisperX**: Advanced speech recognition with alignment and diarization
- **PyTorch**: Deep learning framework for model inference
- **Pyannote.audio**: Speaker diarization models

---

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python Version**: `>= 3.9` and `< 3.13` (Recommended: **3.12**)
  - ⚠️ **Important**: Python 3.13+ is not supported due to WhisperX dependency issues (removed internal libraries)
- **Node.js**: v16 or higher
- **GPU (Optional)**: NVIDIA GPU with CUDA support for faster processing

### Required Accounts
- **HuggingFace Account**: Required for speaker diarization
  - Create account at [https://huggingface.co](https://huggingface.co)
  - Accept terms at [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
  - Generate access token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AryanGupta-12/VoiceScribe-AI.git
cd voicescribe-ai
```

### 2. Backend Setup

#### Create Python Virtual Environment
```bash
# Windows
py -3.12 -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### For GPU Support (NVIDIA CUDA)
If you have an NVIDIA GPU with CUDA support:

```bash
# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install cuDNN (if not already installed)
conda install -c conda-forge cudnn
```

#### Configure Environment Variables
Create a `.env` file in the root directory:

```env
HF_TOKEN=your_huggingface_token_here
```

Replace `your_huggingface_token_here` with your actual HuggingFace access token.

### 3. Frontend Setup

#### Install Node Dependencies
```bash
npm install
```

---

## 📦 Required Python Packages

Create a `requirements.txt` file:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
whisperx
torch>=2.0.0
torchaudio>=2.0.0
pyannote.audio>=3.1.0
python-dotenv==1.0.0
pydub==0.25.1
soundfile==0.12.1
numpy>=1.24.0
```

---

## ▶️ Running the Application

### 1. Start the Backend Server

```bash
# Activate virtual environment first
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Run FastAPI server
python main.py
```

The backend will start at `http://localhost:8000`

### 2. Start the Electron App

In a new terminal:

```bash
npm start
```

The desktop application will launch automatically.

---

## 🎮 Usage

1. **Launch the Application**: Start both backend and frontend as described above
2. **Start Recording**: Click the record button to begin capturing system audio
3. **Real-time Preview**: View live transcription as you speak
4. **Stop Recording**: Click stop when finished
5. **Save & Process**: The application will:
   - Save the audio file (.wav)
   - Generate basic transcript (plain text)
   - Generate diarized transcript (with speaker labels)
6. **Download Results**: Download audio and both transcript formats

---

## 📁 Project Structure

```
voicescribe-ai/
├── main.py                    # FastAPI backend server
├── recordings/                # Saved audio files and basic transcripts
├── diarized_transcripts/      # Speaker-labeled transcripts
├── .env                       # Environment variables (HF token)
├── requirements.txt           # Python dependencies
├── package.json               # Node.js dependencies
├── package-lock.json                                # Electron frontend source
├── index.html
├── main.js
├── renderer.js
├── venv/                      # Python virtual environment
└── README.md                  # This file
```

---

## 🐛 Troubleshooting

### Python Version Issues
```bash
# Check Python version
python --version

# If using wrong version, create venv with specific Python
python3.12 -m venv venv
```

### WhisperX Installation Errors
If you encounter errors installing WhisperX:
```bash
pip install --upgrade setuptools wheel
pip install whisperx --no-cache-dir
```

### CUDA/GPU Issues
```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CPU version
pip install torch torchvision torchaudio
```

### cuDNN Missing (Windows)
Download cuDNN from NVIDIA website or install via conda:
```bash
conda install -c conda-forge cudnn
```

### HuggingFace Token Issues
- Ensure you've accepted terms at pyannote/speaker-diarization
- Verify token has proper permissions
- Check `.env` file format (no quotes around token)

---

## 🔧 Configuration

### Change Whisper Model Size
In `main.py`, modify the model size for accuracy/speed tradeoff:

```python
# Options: tiny, base, small, medium, large
whisperx_model = whisperx.load_model("small", device=device, compute_type=compute_type)
```

| Model  | Speed | Accuracy | GPU Memory |
|--------|-------|----------|------------|
| tiny   | ⚡⚡⚡ | ⭐⭐    | ~1 GB      |
| base   | ⚡⚡   | ⭐⭐⭐  | ~1 GB      |
| small  | ⚡⚡   | ⭐⭐⭐⭐ | ~2 GB      |
| medium | ⚡     | ⭐⭐⭐⭐⭐ | ~5 GB      |
| large  | ⚡     | ⭐⭐⭐⭐⭐ | ~10 GB     |

---

## 📊 Output Format

### Basic Transcript (`transcript_TIMESTAMP.txt`)
```
This is a sample meeting discussion about project updates and deadlines.
```

### Diarized Transcript (`diarized_transcript_TIMESTAMP.txt`)
```
[0.0s - 5.2s] SPEAKER_00: This is a sample meeting discussion.
[5.5s - 12.3s] SPEAKER_01: Great, let's talk about project updates.
[12.8s - 18.4s] SPEAKER_00: We need to review the deadlines.
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

---

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [WhisperX](https://github.com/m-bain/whisperX) - Enhanced Whisper with alignment and diarization
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Electron](https://www.electronjs.org/) - Desktop application framework

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

## 🌟 Star this Repository

If you find this project useful, please give it a ⭐ on GitHub!

---

**Made with ❤️ for better meeting productivity**