# 🎬 AI Video Assistant

> **An End-to-End Multimodal Intelligence Pipeline for Video/Audio Transcription, Multilingual Translation, Map-Reduce Summarization, Structured Decision Extraction, and Grounded Conversational RAG.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/LangChain-LCEL-1C3C3C.svg?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![LLM](https://img.shields.io/badge/LLM-Mistral--Small-FD6F00.svg?logo=mistralai&logoColor=white)](https://mistral.ai/)
[![ASR](https://img.shields.io/badge/ASR-OpenAI%20Whisper-412991.svg?logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![Indic ASR](https://img.shields.io/badge/Indic%20ASR-Sarvam%20AI-4F46E5.svg)](https://www.sarvam.ai/)
[![Vector DB](https://img.shields.io/badge/Vector%20DB-ChromaDB-FF6600.svg)](https://www.trychroma.com/)
[![Frontend](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Table of Contents

- [Overview & Motivation](#-overview--motivation)
- [Key Features](#-key-features)
- [Core Pipeline Modules](#-core-pipeline-modules)
  - [1. Audio Ingestion & Chunking](#1-audio-ingestion--chunking)
  - [2. Hybrid Speech-to-Text Engine](#2-hybrid-speech-to-text-engine)
  - [3. Map-Reduce Summarization & Title Generation](#3-map-reduce-summarization--title-generation)
  - [4. Structured Intelligence Extraction](#4-structured-intelligence-extraction)
  - [5. Conversational RAG Engine](#5-conversational-rag-engine)
  - [6. Modern Cyberpunk Web Interface](#6-modern-cyberpunk-web-interface)
- [Tech Stack](#-tech-stack)
- [Directory Structure](#-directory-structure)
- [Prerequisites & System Setup](#-prerequisites--system-setup)
- [Installation Guide](#-installation-guide)
- [Usage Guide](#-usage-guide)
  - [1. Streamlit Interactive Web Application](#1-streamlit-interactive-web-application)
  - [2. Headless CLI Mode](#2-headless-cli-mode)
  - [3. Component Verification Script](#3-component-verification-script)
- [Environment Configuration](#-environment-configuration)
- [Engineering Design Decisions & Trade-Offs](#-engineering-design-decisions--trade-offs)
- [Future Roadmap](#-future-roadmap)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 📖 Overview & Motivation

In modern academic and corporate workflows, hours of high-value knowledge remain trapped inside recorded meetings, webinars, research lectures, and presentations. Manually reviewing multi-hour videos to extract action items, critical decisions, and answers to specific follow-up questions is time-prohibitive.

**AI Video Assistant** is an end-to-end multimodal software engineering project developed to solve this problem. It autonomously ingests media from YouTube URLs or local video/audio files, normalizes and segments the audio streams, transcribes both English and code-mixed Hinglish speech, synthesizes executive summaries via map-reduce LLM chains, extracts structured action items with assignees and deadlines, and indexes the entire dialogue into a vector database for semantic retrieval-augmented question answering.

---

## ✨ Key Features

- **🌐 Multi-Source Ingestion**: Ingests direct YouTube links (via `yt-dlp`) or any local video/audio file (`.mp4`, `.mkv`, `.mp3`, `.wav`, `.m4a`).
- **🎛️ Audio Normalization & Chunking**: Standardizes audio to single-channel (mono) 16 kHz WAV format and applies 10-minute sliding window chunking to ensure stability and low memory footprint on long inputs.
- **🗣️ Hybrid Multilingual ASR**:
  - **English Mode**: Offline, zero-cloud-cost transcription using OpenAI's **Whisper** model family (`small` by default, configurable).
  - **Hinglish Mode**: Code-mixed Hindi-English transcription and real-time translation powered by **Sarvam AI** (`saaras:v2.5`), utilizing an automated 25-second chunking scheduler with safety margins to adhere to API constraints.
- **📝 Map-Reduce Summarization**: Overcomes LLM context window limits by recursively splitting transcripts into 3,000-character chunks with overlap, generating per-chunk syntheses, and reducing them into a structured executive brief using **Mistral AI** (`mistral-small-latest`).
- **🎯 Structured Extraction**:
  - **Action Items**: Identifies tasks, assigned owners, and stated deadlines.
  - **Key Decisions**: Catalogs all firm organizational and project decisions.
  - **Open Questions**: Surfaces unresolved topics and pending follow-ups.
- **🧠 Zero-Hallucination Conversational RAG**:
  - Chunks transcripts (500 chars, 50 overlap) and generates dense vector embeddings locally using `sentence-transformers/all-MiniLM-L6-v2`.
  - Stores vectors in a persistent **ChromaDB** collection.
  - Executes similarity search ($k=4$) with a strictly grounded LangChain LCEL chain to answer user queries with verifiable citations.
- **💻 Dual Interaction Modes**:
  - **Streamlit Web UI**: Glassmorphic dark-theme interface with custom CSS typography (`Syne` + `JetBrains Mono`), animated background grid, dynamic step-by-step pipeline tracker, and conversational chat bubble interface.
  - **Terminal CLI**: Headless command-line execution (`main.py`) for automated batch processing and terminal lovers.
---

## 🔬 Core Pipeline Modules

### 1. Audio Ingestion & Chunking
- **Location**: `utils/audio_processor.py`
- Handles source differentiation: downloads audio tracks from YouTube with `yt-dlp` using best-quality streams, or reads local container formats (`.mp4`, `.mov`, `.mkv`, `.wav`, etc.) via `pydub`.
- Converts inputs to uniform single-channel 16,000 Hz WAV files, optimized for acoustic models.
- Divides long recordings into 10-minute segments to avoid PyTorch GPU/RAM Out-Of-Memory exceptions during transcription.

### 2. Hybrid Speech-to-Text Engine
- **Location**: `core/transcriber.py`
- **Whisper Pipeline**: Loads `openai-whisper` (default model: `small`) onto CPU or CUDA for offline, cost-free transcription.
- **Sarvam AI Pipeline**: Designed for Indian bilingual contexts (Hinglish/Hindi). Since Sarvam's synchronous STT-translate endpoint enforces a 30-second duration limit, the module dynamically dices each 10-minute audio chunk into 25-second micro-pieces with safe buffer margins, concurrently translates and transcribes them, and reassembles the complete English transcript.

### 3. Map-Reduce Summarization & Title Generation
- **Location**: `core/summarizer.py`
- Employs **LangChain Expression Language (LCEL)** to build modular runnable chains.
- Uses `RecursiveCharacterTextSplitter` (chunk size: 3000, overlap: 200) to map individual sections to partial summaries, and reduces them into an executive bulleted brief.
- Generates a concise, context-aware title (maximum 8 words) from the opening transcript segments.

### 4. Structured Intelligence Extraction
- **Location**: `core/extractor.py`
- Powered by `mistral-small-latest` running at a low temperature ($0.2$) for deterministic, accurate parsing.
- Extracts:
  - **Action Items**: Tabulates concrete tasks, designated owners, and committed deadlines.
  - **Key Decisions**: Extracts strategic and architectural agreements reached during the meeting.
  - **Open Questions**: Extracts unresolved concerns and unanswered questions requiring follow-up.

### 5. Conversational RAG Engine
- **Location**: `core/vector_store.py` & `core/rag_engine.py`
- Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` embeddings for lightweight, high-performance vector transformations without cloud API overhead.
- Persists document embeddings into a local **ChromaDB** database (`vector_db/`).
- Enforces strict grounding prompts: if an answer is not explicitly contained within the retrieved context chunks ($k=4$), the model replies with an explicit disclaimer, preventing LLM hallucinations.

### 6. Modern Cyberpunk Web Interface
- **Location**: `app.py`
- Built using **Streamlit** styled with custom CSS variables, dark glassmorphic cards, glowing accent borders, and retro monospace fonts (`Syne` headers with `JetBrains Mono` body).
- Features live animated pipeline status indicators, expandable raw transcript views, three-column metric distributions, and a conversational chat interface with session history management.

---

## 🛠️ Tech Stack

| Domain | Technology / Library | Role in System |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core language environment |
| **User Interface** | Streamlit, HTML5, Custom CSS | Interactive web dashboard & chat client |
| **Media Extraction** | yt-dlp, FFmpeg, pydub | Audio downloading, 16kHz resampling, chunking |
| **Speech-to-Text** | OpenAI Whisper | Local acoustic transcription (English) |
| **Indic Translation** | Sarvam AI (`saaras:v2.5`) | Hinglish/Hindi transcription & translation |
| **Orchestration** | LangChain Core (LCEL) | Modular prompt templates, runnables & chains |
| **LLM Provider** | Mistral AI (`mistral-small-latest`)| Map-Reduce summarization & semantic QA |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) | Local dense semantic embedding model |
| **Vector Database** | ChromaDB | Local vector indexing and similarity retrieval |
| **Configuration** | python-dotenv | Secure secret and environment management |

---

## 📁 Directory Structure

```text
AI_Video_Assistant/
├── core/
│   ├── __init__.py
│   ├── extractor.py         # LLM chains for action items, decisions, open questions
│   ├── rag_engine.py        # LangChain LCEL RAG pipeline & Q&A handler
│   ├── summarizer.py        # Map-Reduce summarization and title synthesis
│   ├── transcriber.py       # Hybrid ASR router (OpenAI Whisper + Sarvam AI)
│   └── vector_store.py      # ChromaDB vector store initialization & retrievers
├── utils/
│   ├── __init__.py
│   └── audio_processor.py   # Media downloading, format conversion, and chunking
├── app.py                   # Streamlit web application with modern custom UI
├── main.py                  # Command-line interface (CLI) entry point
├── test.py                  # Pipeline verification script with sample input
├── Requirements.txt         # Project dependencies with version constraints
├── .env.example             # Template for required environment variables
├── .gitignore               # Ignored artifacts, virtual environments, and keys
└── README.md                # Comprehensive project documentation
```

---

## ⚙️ Prerequisites & System Setup

### 1. Python 3.10 or Higher
Verify your Python installation:
```bash
python --version
```

### 2. FFmpeg Binary
FFmpeg is **required** by `pydub` and `yt-dlp` for audio extraction and transcoding.

- **Windows (via winget or chocolatey)**:
  ```powershell
  winget install Gyan.FFmpeg
  # or
  choco install ffmpeg
  ```
- **macOS (via Homebrew)**:
  ```bash
  brew install ffmpeg
  ```
- **Linux (Ubuntu / Debian)**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```

---

## 🚀 Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/pp150204/AI-Video-Assistant.git
cd AI-Video-Assistant
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r Requirements.txt
```

> **Note for PyTorch**: If you have an NVIDIA GPU and wish to leverage CUDA acceleration for Whisper, install the CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).

### Step 4: Configure Environment Variables
Copy `.env.example` to create your local `.env` file:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and fill in your API credentials:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
WHISPER_MODEL=small
SARVAM_STT_MODEL=saaras:v2.5
```

---

## 🖥️ Usage Guide

### 1. Streamlit Interactive Web Application
Launch the rich dashboard in your default browser:
```bash
streamlit run app.py
```
1. Paste a **YouTube URL** or a **Local File Path** (e.g., `C:\recordings\meeting.mp4`) in the sidebar.
2. Select the spoken language: **`english`** (processed locally via Whisper) or **`hinglish`** (processed via Sarvam AI).
3. Click **⚡ Analyse** to begin processing.
4. Review the generated session title, summary, action items, decisions, and open questions.
5. Use the **💬 Chat with your Meeting** section to ask specific questions directly against the indexed transcript!

---

### 2. Headless CLI Mode
For automated execution or headless environments:
```bash
python main.py
```
Follow the interactive prompts:
```text
Enter YouTube URL or local file path: https://www.youtube.com/watch?v=...
Language (english/hinglish): english
```
Once the analysis completes, an interactive terminal session will start where you can query your meeting in real-time.

---

### 3. Component Verification Script
To run an automated test against a sample URL and verify that all pipeline components (audio ingestion, speech-to-text, summarization, extraction) work correctly:
```bash
python test.py
```

---

## 🔐 Environment Configuration

| Variable | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `MISTRAL_API_KEY` | **Required** | — | Mistral AI API key for title, summary, extraction, and RAG QA. |
| `SARVAM_API_KEY` | *Optional* | — | Required only if transcribing code-mixed Hinglish / Indic content. |
| `WHISPER_MODEL` | *Optional* | `small` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`). |
| `SARVAM_STT_MODEL` | *Optional* | `saaras:v2.5` | Target speech model on Sarvam's API. |

---

## 🧠 Engineering Design Decisions & Trade-Offs

1. **Local Whisper vs. Hosted Cloud Transcription**:
   - *Decision*: Adopted local OpenAI Whisper for English transcription.
   - *Rationale*: Eliminates recurring per-minute cloud API transcription bills, protects meeting data privacy, and enables fully offline ASR processing.
2. **Sarvam AI Sub-Chunking Algorithm**:
   - *Challenge*: Sarvam's synchronous translation API restricts uploads to audio clips $\le 30$ seconds.
   - *Solution*: Engineered a custom chunking scheduler in `transcribe_chunk_sarvam()` that splits 10-minute master chunks into 25-second slices (5s safety margin), sends them sequentially with automatic temporary file cleanup, and concatenates the resulting stream.
3. **Map-Reduce vs. Single-Pass Summarization**:
   - *Challenge*: Long meeting transcripts can easily exceed context windows or lead to the "lost in the middle" attention degradation problem.
   - *Solution*: Implemented a LangChain Map-Reduce pipeline. Transcripts are segmented into 3,000-character windows, summarized individually, and then aggregated into a cohesive executive report.
4. **Local Embeddings for Zero-Cost RAG**:
   - *Decision*: Selected `sentence-transformers/all-MiniLM-L6-v2` rather than commercial embedding APIs.
   - *Rationale*: Runs lightning-fast on CPU, generates compact 384-dimensional embeddings, and guarantees zero additional cost when generating or querying vector stores.
5. **Strict Context-Bound RAG Prompting**:
   - *Decision*: Prompt guards instructing the LLM to explicitly state when information is absent.
   - *Rationale*: In meeting intelligence, hallucinations can result in missed deadlines or phantom decisions. Strict grounding ensures truthful, verifiable answers.

---

## 🗺️ Future Roadmap

- [ ] **Speaker Diarization**: Integrate `pyannote.audio` to attribute quotes to specific participants (e.g., *Speaker 1*, *Speaker 2*).
- [ ] **Automated Report Export**: Implement one-click export of summaries, action items, and transcripts to formatted PDF/DOCX reports using `reportlab`.
- [ ] **Real-Time Live Microphone Capture**: Stream microphone audio from live meetings (Zoom/Google Meet) for real-time transcription.
- [ ] **Multi-Meeting Knowledge Base**: Support persistent indexing across multiple meetings, enabling global cross-meeting queries.
- [ ] **Docker Containerization**: Provide a multi-stage `Dockerfile` with pre-compiled FFmpeg and GPU drivers for one-command container deployment.

---

## 👨‍💻 Author & Acknowledgments

Developed by **[Prathmesh Pimpare](https://github.com/pp150204)**

Special thanks to:
- The **OpenAI Whisper** and **PyTorch** teams for state-of-the-art open-source speech recognition.
- **Mistral AI** for high-efficiency, reasoning-capable LLMs.
- **LangChain** and **ChromaDB** for powerful RAG orchestration tools.
- **Sarvam AI** for cutting-edge Indian language speech technology.

---

<p align="center">
  <b>⭐ If you found this project helpful, please consider starring the repository! ⭐</b>
</p>
