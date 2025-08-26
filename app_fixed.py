import streamlit as st
import datetime
from utils import (
    convert_mp4_to_wav,
    transcribe_audio,
    extract_speakers_and_updates,
    summarize_updates,
    format_summary_report
)

st.set_page_config(page_title="Meeting Progress Summarizer", layout="centered")

st.title("📋 Meeting Transcript Summarizer")

uploaded_file = st.file_uploader("Upload MP4 meeting recording", type=["mp4"])
mode = st.selectbox("Speaker Detection Mode", ["LLM", "Strict", "Smart"])
llm_model = st.selectbox("Choose LLM for extraction & summarization", ["mistral", "llama3", "gemma", "pyannote"])
meeting_date = st.date_input("Meeting Date (optional)", value=datetime.date.today())

if uploaded_file:
    with st.spinner("🔄 Converting video to audio..."):
        wav_path = convert_mp4_to_wav(uploaded_file)

    if llm_model == "pyannote":
        with st.spinner("🗣️ Transcribing audio with speaker diarization..."):
            speaker_updates = extract_speakers_and_updates(wav_path, mode, llm_model)
        
        # Create a combined transcript for preview
        transcript = "\n\n".join([f"{speaker}: {text}" for speaker, text in speaker_updates.items()])
        st.success("✅ Transcription and speaker diarization complete")
        st.text_area("Transcript Preview (with speakers)", transcript, height=200)
    else:
        with st.spinner("🗣️ Transcribing audio..."):
            transcript = transcribe_audio(wav_path)

        st.success("✅ Transcription complete")
        st.text_area("Transcript Preview", transcript, height=200)

        with st.spinner("🧑‍💼 Extracting speaker updates..."):
            speaker_updates = extract_speakers_and_updates(transcript, mode, llm_model)

    with st.spinner("🧠 Summarizing each speaker's progress..."):
        summaries = summarize_updates(speaker_updates, llm_model)

    with st.spinner("📄 Generating summary report..."):
        report = format_summary_report(summaries, meeting_date)

    st.download_button("📥 Download Summary (TXT)", data=report, file_name="meeting_summary.txt", mime="text/plain")
    st.text_area("📝 Weekly Progress Summary", report, height=300)
