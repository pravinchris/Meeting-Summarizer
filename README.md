# 📋 Meeting-Summarizer

A powerful AI-powered tool that automatically transcribes meeting recordings, identifies speakers, and generates comprehensive summaries with multiple export options.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Whisper](https://img.shields.io/badge/Whisper-OpenAI-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

### 🎯 Core Functionality
- **🎥 Audio/Video Processing**: Convert MP4 meeting recordings to audio automatically
- **🗣️ Advanced Transcription**: Multiple Whisper engine options with model selection
- **👥 Intelligent Speaker Detection**: AI-powered speaker identification and separation
- **📝 Smart Summarization**: Generate concise summaries for each speaker's contributions
- **📤 Multi-Format Export**: Export summaries as Word, PDF, Excel, or CSV documents

### 🚀 Transcription Engines
- **faster-whisper**: Optimized implementation with lower memory usage and faster processing
- **openai-whisper**: Original OpenAI implementation with full feature set
- **whisperx**: Enhanced version with improved timestamp accuracy and speaker diarization

### 🎭 Speaker Detection Methods
- **🤖 LLM-Only**: Pure AI text analysis using advanced language models
- **🎯 Pyannote**: State-of-the-art audio-based speaker diarization
- **🧠 SpeechBrain**: Speaker recognition using deep learning embeddings
- **🔊 Resemblyzer**: Voice verification and clustering technology

### 🧠 Supported AI Models
- **Whisper Models**: tiny, base, small, medium, large (with English-only .en variants)
- **LLM Models**: Qwen2, Phi3, Llama2, Gemma, Mistral (via Ollama)

## 🚀 Quick Start

### Prerequisites
- **Python**: 3.8 or higher
- **FFmpeg**: For audio/video conversion
- **Ollama**: For local LLM inference (recommended)

### 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Meeting-Summarizer
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg**

   **macOS (using Homebrew):**
   ```bash
   brew install ffmpeg
   ```

   **Ubuntu/Debian:**
   ```bash
   sudo apt update && sudo apt install ffmpeg
   ```

   **Windows:**
   Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

4. **Install Ollama (for LLM functionality)**

   **macOS/Linux:**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

   **Windows:**
   Download from [https://ollama.ai/download](https://ollama.ai/download)

5. **Pull required LLM models**
   ```bash
   ollama pull qwen2:latest
   ollama pull phi3:latest
   ollama pull llama2:latest
   ```

### 🎬 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload your meeting recording** (MP4 format supported)

4. **Configure settings** using the intuitive button interface:
   - 🚀 Select transcription engine (faster-whisper recommended)
   - 📏 Choose Whisper model size (small.en for balance of speed/accuracy)
   - 🎭 Pick speaker detection method (LLM-Only for most cases)
   - 🧠 Configure LLM settings (Qwen2 recommended)

5. **Process the meeting** and view real-time progress

6. **Review results** with expandable transcript and speaker breakdowns

7. **Export summaries** in your preferred format(s)

## 📁 Project Structure

```
Meeting-Summarizer/
├── 📄 app.py                    # Main Streamlit application with button interface
├── 🔧 utils.py                  # Core processing functions and algorithms
├── 📋 requirements.txt          # Python dependencies
├── 📚 README.md                 # This comprehensive guide
├── 📖 EXPORT_GUIDE.md           # Detailed export functionality guide
├── 🔬 export_example.py         # Example usage of export functions
├── 📤 app_with_export.py        # Enhanced app version (legacy)
├── 📊 exports/                  # Generated export files
│   ├── meeting_summary_*.docx
│   ├── meeting_summary_*.pdf
│   ├── meeting_summary_*.xlsx
│   └── meeting_summary_*.csv
└── 🎛️ pretrained_models/        # Downloaded AI models cache
    ├── spkrec-ecapa-voxceleb/
    └── sb_asr_crdnn/
```

## ⚙️ Configuration Guide

### 🚀 Transcription Engines Comparison

| Engine           | Memory Usage | Speed      | Accuracy | Best For                  |
|------------------|-------------|------------|----------|---------------------------|
| **faster-whisper** | ⭐⭐⭐ Low   | ⭐⭐⭐ Fast   | ⭐⭐ Good | Production, resource-constrained |
| **openai-whisper** | ⭐ High    | ⭐⭐ Medium  | ⭐⭐⭐ Excellent | Development, maximum accuracy |
| **whisperx**       | ⭐⭐ Medium | ⭐⭐ Medium  | ⭐⭐⭐ Excellent | Precise timestamps needed |

### 📏 Whisper Models Comparison

| Model         | Size     | Speed        | Accuracy          | Memory | Language Support  |
|---------------|----------|--------------|-------------------|--------|------------------|
| **tiny.en**   | 39 MB    | ~32x faster  | ⭐⭐ Basic         | 1GB    | English only     |
| **small.en**  | 244 MB   | ~6x faster   | ⭐⭐⭐ Good         | 2GB    | English only     |
| **medium.en** | 769 MB   | ~2x faster   | ⭐⭐⭐⭐ Very Good   | 5GB    | English only     |
| **large-v3**  | 1550 MB  | 1x baseline  | ⭐⭐⭐⭐⭐ Excellent  | 10GB   | 99+ languages    |

### 🎭 Speaker Detection Methods

| Method         | Accuracy   | Speed      | Setup Complexity | Best For            |
|----------------|------------|------------|------------------|---------------------|
| **LLM-Only**   | ⭐⭐⭐ Good   | ⭐⭐⭐ Fast   | ⭐⭐⭐ Easy         | General meetings, quick setup |
| **Pyannote**   | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐ Medium | ⭐⭐ Medium | High accuracy requirements |
| **SpeechBrain**| ⭐⭐⭐⭐ Very Good | ⭐⭐ Medium | ⭐⭐ Medium | Speaker recognition focus |
| **Resemblyzer**| ⭐⭐⭐ Good   | ⭐⭐⭐ Fast   | ⭐⭐⭐ Easy         | Voice clustering     |

## 📦 Dependencies

### 🔨 Core Requirements
```
streamlit>=1.28.0          # Web interface framework
faster-whisper>=0.10.0     # Optimized speech recognition
openai-whisper>=20231117   # Original Whisper implementation
ollama>=0.1.0              # Local LLM inference
pandas>=1.5.0              # Data manipulation
ffmpeg-python>=0.2.0       # Audio/video processing
```

### 📄 Export Dependencies (Optional)
```
python-docx>=0.8.11        # Microsoft Word export
reportlab>=4.0.0           # PDF generation
openpyxl>=3.1.0            # Excel spreadsheet export
```

### 🎭 Advanced Speaker Detection (Optional)
```
pyannote.audio>=3.1.0      # Professional speaker diarization
speechbrain>=0.5.0         # Deep learning speaker recognition
resemblyzer>=0.1.0         # Voice verification clustering
torch>=1.9.0               # Deep learning framework
torchaudio>=0.9.0          # Audio processing for PyTorch
```

## 💡 Usage Examples

### 🔧 Basic API Usage
```python
from utils import (
    convert_mp4_to_wav,
    transcribe_audio,
    extract_speakers_and_updates,
    summarize_updates
)

# Process a meeting recording step by step
wav_path = convert_mp4_to_wav(uploaded_file)
transcript = transcribe_audio(wav_path, "faster-whisper", "small.en")
speakers = extract_speakers_and_updates(transcript, "LLM", "qwen2:latest")
summaries = summarize_updates(speakers, "qwen2:latest")

print(f"Processed {len(speakers)} speakers")
for speaker, summary in summaries.items():
    print(f"{speaker}: {summary}")
```

### 📤 Export Functions
```python
from app import export_to_word, export_to_pdf, export_to_excel, export_to_csv
from datetime import datetime

# Export summaries to multiple formats
meeting_date = datetime.now()

# Individual exports
word_file = export_to_word(summaries, meeting_date, "team_meeting.docx")
pdf_file = export_to_pdf(summaries, meeting_date, "team_meeting.pdf")
excel_file = export_to_excel(summaries, meeting_date, "team_meeting.xlsx")
csv_file = export_to_csv(summaries, meeting_date, "team_meeting.csv")

print(f"Exported to: {word_file}, {pdf_file}, {excel_file}, {csv_file}")
```

### 🔄 Batch Processing
```python
import os
from pathlib import Path

# Process multiple meeting files
meeting_files = Path("meetings/").glob("*.mp4")

for meeting_file in meeting_files:
    print(f"Processing {meeting_file.name}...")
    
    # Convert and transcribe
    wav_path = convert_mp4_to_wav(meeting_file)
    transcript = transcribe_audio(wav_path, "faster-whisper", "small.en")
    
    # Extract speakers and summarize
    speakers = extract_speakers_and_updates(transcript, "LLM", "qwen2:latest")
    summaries = summarize_updates(speakers, "qwen2:latest")
    
    # Export results
    output_name = meeting_file.stem
    export_to_word(summaries, datetime.now(), f"{output_name}_summary.docx")
    
    print(f"✅ Completed {meeting_file.name}")
```

## 🔍 Troubleshooting

### 🚨 Common Issues & Solutions

**1. FFmpeg not found error**
```bash
# Verify FFmpeg installation
ffmpeg -version

# If not found, install:
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

**2. Ollama connection failed**
```bash
# Start Ollama service
ollama serve

# Verify models are available
ollama list

# Pull missing models
ollama pull qwen2:latest
```

**3. Out of memory errors**
```python
# Use smaller models for limited memory
transcribe_audio(wav_path, "faster-whisper", "tiny.en")  # Instead of large models
```

**4. Export dependencies missing**
```bash
# Install all export dependencies
pip install python-docx reportlab openpyxl

# Or install selectively
pip install python-docx  # For Word export only
```

**5. Speaker detection not working**
```python
# Try different detection methods
speakers = extract_speakers_and_updates(transcript, "Smart", "qwen2:latest")
# Or use audio-based detection
speakers = transcribe_with_speakers(wav_path)  # Requires pyannote
```

### ⚡ Performance Optimization Tips

- **For Speed**: Use `tiny.en` or `small.en` models with `faster-whisper`
- **For Accuracy**: Use `medium.en` or `large-v3` with `openai-whisper`
- **For Memory**: Stick with `faster-whisper` and smaller models
- **For Speaker Accuracy**: Use `pyannote` method with sufficient memory
- **For Long Meetings**: Process in chunks or use background processing

## 📊 Supported Formats

### 📥 Input Formats
- **Video**: MP4 (automatically converted to audio)
- **Audio**: WAV (16kHz mono, auto-generated from video)

### 📤 Output Formats

| Format    | Extension | Features                        | Best For               |
|-----------|-----------|---------------------------------|------------------------|
| **Word**  | .docx     | Professional formatting, editable | Reports, documentation |
| **PDF**   | .pdf      | Universal format, consistent layout | Final reports, sharing |
| **Excel** | .xlsx     | Structured data, word counts, analysis | Data analysis, metrics |
| **CSV**   | .csv      | Simple format, database-friendly | Data processing, integration |

## 🛣️ Roadmap

### 🔜 Coming Soon
- [ ] Real-time transcription capabilities
- [ ] Support for more input formats (MOV, AVI, MP3, WAV)
- [ ] Integration with Zoom, Teams, Google Meet
- [ ] Advanced analytics and speaker insights
- [ ] Custom model fine-tuning options

### 🔮 Future Plans
- [ ] Multi-language interface support
- [ ] Cloud deployment options
- [ ] REST API for programmatic access
- [ ] Mobile app companion
- [ ] Integration with productivity tools (Slack, Notion, etc.)

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes** and add tests
4. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
5. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

### 🧪 Development Setup
```bash
# Clone your fork
git clone https://github.com/pravinchris/Meeting-Summarizer.git

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/

# Start development server
streamlit run app.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[OpenAI Whisper](https://github.com/openai/whisper)** - Revolutionary speech recognition technology
- **[faster-whisper](https://github.com/guillaumekln/faster-whisper)** - Optimized Whisper implementation
- **[Ollama](https://ollama.ai/)** - Simplified local LLM deployment
- **[Streamlit](https://streamlit.io/)** - Rapid web app development framework
- **[Pyannote](https://github.com/pyannote/pyannote-audio)** - Advanced speaker diarization
- **[SpeechBrain](https://speechbrain.github.io/)** - Conversational AI toolkit

## 📞 Support & Community

- **🐛 Issues**: [GitHub Issues](https://github.com/pravinchris/Meeting-Summarizer/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/pravinchris/Meeting-Summarizer/discussions)
- **📧 Email**: mailto:pravindchristopher@gmail.com
- **🐦 Linkedin**: https://www.linkedin.com/in/pravindchristopher/

## 📈 Statistics

- **🎯 Accuracy**: Up to 95% transcription accuracy with large models
- **⚡ Speed**: 10x faster than real-time with optimized models
- **💾 Memory**: As low as 1GB RAM with tiny models
- **🌍 Languages**: 99+ languages supported with multilingual models
- **📱 Platforms**: Windows, macOS, Linux compatible

---

**Made with ❤️ by Pravin**

*Transform your meetings into actionable insights with the power of AI!*