import warnings
import streamlit as st
import datetime

# Suppress all warnings for clean UI        # Step 3: Summarization (always use Ollama)
        with st.spinner(f"🤖 Summarizing with Ollama ({llm_model})..."):
            summaries = summarize_updates(speaker_updates, llm_model, use_name_extraction)rnings.filterwarnings("ignore")

from utils import (
    convert_mp4_to_wav,
    transcribe_audio,
    transcribe_with_speakers,
    transcribe_with_speechbrain,
    transcribe_with_resemblyzer,
    extract_speakers_and_updates,
    summarize_updates,
    format_summary_report
)

st.set_page_config(page_title="Meeting Progress Summarizer", layout="centered")

st.title("📋 Meeting Transcript Summarizer")

uploaded_file = st.file_uploader("Upload MP4 meeting recording", type=["mp4"])

# Separate speaker detection and summarization choices
st.subheader("🔧 Configuration")
col1, col2 = st.columns(2)

with col1:
    st.write("**Speaker Detection Method:**")
    speaker_method = st.selectbox("How to identify speakers", ["pyannote", "speechbrain", "resemblyzer", "ollama-llm"], key="speaker")
    
with col2:
    st.write("**Text Processing Mode:**")
    if speaker_method == "ollama-llm":
        mode = st.selectbox("LLM Processing Mode", ["LLM", "Strict", "Smart"], key="mode")
        llm_model = st.selectbox("Ollama Model", ["mistral", "llama3", "gemma"], key="llm")
    else:
        mode = "Audio"  # Not used for audio-based methods
        llm_model = st.selectbox("Ollama Model for Summarization", ["mistral", "llama3", "gemma"], key="summary_llm")

meeting_date = st.date_input("Meeting Date (optional)", value=datetime.date.today())

# Add option for name extraction
st.sidebar.header("Advanced Options")
use_name_extraction = st.sidebar.checkbox(
    "Extract speaker names from speech", 
    value=True, 
    help="Attempt to identify speakers by their actual names mentioned in the conversation (e.g., 'Hi, I'm Sarah')"
)

if uploaded_file is not None:
    with st.spinner("🔄 Converting video to audio..."):
        wav_path = convert_mp4_to_wav(uploaded_file)

    # Step 1: Transcription (always use Whisper)
    with st.spinner("🗣️ Transcribing audio with Whisper..."):
        transcript = transcribe_audio(wav_path)
    
    st.success("✅ Transcription complete")
    with st.expander("📝 View Raw Transcript"):
        st.text_area("Raw Transcript", transcript, height=150)

    # Step 2: Speaker Detection/Diarization
    if speaker_method == "pyannote":
        with st.spinner("🎯 Performing speaker diarization with pyannote..."):
            speaker_updates = transcribe_with_speakers(wav_path)
        method_name = "Pyannote speaker diarization"
        
    elif speaker_method == "speechbrain":
        with st.spinner("🎯 Performing speaker recognition with SpeechBrain..."):
            speaker_updates = transcribe_with_speechbrain(wav_path)
        method_name = "SpeechBrain speaker recognition"
        
    elif speaker_method == "resemblyzer":
        with st.spinner("🎯 Performing speaker verification with Resemblyzer..."):
            speaker_updates = transcribe_with_resemblyzer(wav_path)
        method_name = "Resemblyzer speaker verification"
        
    else:  # ollama-llm
        with st.spinner("🎯 Extracting speakers with Ollama LLM..."):
            speaker_updates = extract_speakers_and_updates(transcript, mode, llm_model)
        method_name = f"Ollama {llm_model} text analysis"
    
    st.success(f"✅ Speaker detection complete using {method_name}")
    
    # Display speaker-separated transcript
    if speaker_updates:
        speaker_transcript = "\n\n".join([f"**{speaker}:**\n{text}" for speaker, text in speaker_updates.items()])
        with st.expander("👥 View Speaker-Separated Transcript"):
            st.markdown(speaker_transcript)

        # Step 3: Summarization (always use Ollama)
        with st.spinner(f"� Summarizing with Ollama ({llm_model})..."):
            summaries = summarize_updates(speaker_updates, llm_model)

        st.success("✅ Summarization complete")

        # Step 4: Generate and display final report
        with st.spinner("📄 Generating summary report..."):
            report = format_summary_report(summaries, meeting_date)

        st.subheader("📋 Meeting Summary Report")
        st.download_button("📥 Download Summary (TXT)", data=report, file_name="meeting_summary.txt", mime="text/plain")
        st.text_area("📝 Final Meeting Summary", report, height=300)
    
    else:
        st.error("❌ No speakers detected. Please try a different detection method.")
