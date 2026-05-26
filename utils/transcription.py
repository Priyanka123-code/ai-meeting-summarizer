import streamlit as st
import whisper
from pydub import AudioSegment
import os
import json
import wave
from datetime import datetime
import torch
import torchaudio

# ====================== PHASE 3: TIMESTAMP HELPER ======================
def format_timestamp(seconds: float) -> str:
    """Converts seconds into professional [MM:SS] format for the UI"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

# ====================== CACHED MODEL LOADING ======================
@st.cache_resource
def load_whisper_model(model_size="medium"):
    """Loads the model once and keeps it in memory"""
    return whisper.load_model(model_size)

# ====================== PHASE 1: VIDEO & AUDIO PROCESSING ======================
def convert_to_wav(audio_path: str, for_vosk=False) -> str:
    """
    Converts any audio/video format to optimized WAV.
    Supports: mp4, mkv, m4a, mov, avi, flac, ogg
    """
    audio = AudioSegment.from_file(audio_path)
    
    # Vosk requires 16kHz Mono
    if for_vosk:
        audio = audio.set_channels(1).set_frame_rate(16000)
        
    wav_path = audio_path.rsplit('.', 1)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

# ====================== VOSK OFFLINE LOGIC ======================
def transcribe_with_vosk(audio_path: str) -> str:
    """Offline transcription using Vosk"""
    try:
        from vosk import Model, KaldiRecognizer
    except ImportError:
        return "Error: Vosk library not installed. Run 'pip install vosk'"

    model_path = "model" 
    if not os.path.exists(model_path):
        return "Error: Vosk model folder missing at project root."

    wf = wave.open(audio_path, "rb")
    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())
    
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0: break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            results.append(res.get("text", ""))

    res = json.loads(rec.FinalResult())
    results.append(res.get("text", ""))
    return " ".join(results).strip()

# ====================== WHISPER CORE LOGIC ======================
def transcribe_with_whisper(audio_path: str, model_size="medium") -> dict:
    """Returns both full text and segments for diarization"""
    model = load_whisper_model(model_size)
    result = model.transcribe(audio_path, language="en")
    return result

def transcribe_audio(uploaded_file, use_vosk=False) -> str:
    """Main entry point for app.py upload tab"""
    if uploaded_file is None:
        return ""

    os.makedirs("data/audio", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = f"data/audio/{timestamp}_{uploaded_file.name}"
    
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    wav_path = convert_to_wav(audio_path, for_vosk=use_vosk)

    try:
        if use_vosk:
            transcript = transcribe_with_vosk(wav_path)
        else:
            result = transcribe_with_whisper(wav_path, model_size="medium")
            transcript = result["text"].strip()
    finally:
        if os.path.exists(audio_path): os.remove(audio_path)
        if os.path.exists(wav_path): os.remove(wav_path)

    return transcript

# ====================== CACHED DIARIZATION PIPELINE ======================
@st.cache_resource
def load_diarization_pipeline(hf_token):
    """Loads the speaker diarization model from Hugging Face"""
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline

# ====================== PHASE 3: DIARIZATION + TIMESTAMPS ======================
def get_diarized_transcript(audio_path, whisper_segments, hf_token):
    """Merges Whisper text with Speaker IDs and [MM:SS] Timestamps"""
    pipeline = load_diarization_pipeline(hf_token)
    
    waveform, sample_rate = torchaudio.load(audio_path)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    
    final_transcript = []
    for segment in whisper_segments:
        start_t, end_t, text = segment['start'], segment['end'], segment['text']
        timestamp_label = format_timestamp(start_t)
        
        speaker_counts = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            intersection = min(end_t, turn.end) - max(start_t, turn.start)
            if intersection > 0:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + intersection
        
        dominant_speaker = max(speaker_counts, key=speaker_counts.get) if speaker_counts else "Unknown"
        final_transcript.append(f"[{timestamp_label}] [{dominant_speaker}]: {text}")
    
    return "\n".join(final_transcript)