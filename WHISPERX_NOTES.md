# WhisperX Compatibility Notes

## Current Issue
WhisperX has dependency conflicts with:
- NumPy 2.0+ (uses deprecated np.NaN instead of np.nan)
- TensorFlow library compatibility issues on macOS ARM64

## Solution Applied
1. Downgraded NumPy to version <2.0 in requirements.txt
2. Temporarily disabled WhisperX in the UI until dependency conflicts are resolved
3. Application works with faster-whisper and openai-whisper engines

## To Re-enable WhisperX in the Future
1. Update pyannote-audio to a newer version that supports NumPy 2.0+
2. Resolve TensorFlow library dependencies  
3. Add "whisperx" back to transcription engine options in app.py
4. Add "whisperx-diarization" back to speaker detection methods

## Current Working Features
- ✅ faster-whisper transcription
- ✅ openai-whisper transcription  
- ✅ pyannote speaker diarization
- ✅ speechbrain speaker recognition
- ✅ resemblyzer speaker verification
- ✅ ollama LLM models: yi:9b, gemma:7b, llama3:8b, phi3:3.8b, mistral:latest
- ✅ All three speaker detection modes: LLM, Strict, Smart

## Dependencies Working
- numpy<2.0 (for pyannote compatibility)
- All other packages stable with current versions
