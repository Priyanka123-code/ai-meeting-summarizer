import streamlit as st
import whisper
from pydub import AudioSegment
import os
import json
import wave
import shutil
from datetime import datetime
import torch

# ====================== PHASE 3: TIMESTAMP HELPER ======================
def format_timestamp(seconds: float) -> str:
    """Converts seconds into professional [MM:SS] format for the UI"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

# ====================== CACHED MODEL LOADING ======================
@st.cache_resource
def load_whisper_model(model_size="base"):
    """Loads the model once and keeps it in memory"""
    return whisper.load_model(model_size)

# ====================== PHASE 1: VIDEO & AUDIO PROCESSING ======================
def _configure_ffmpeg() -> bool:
    """Point pydub at ffmpeg/ffprobe when they are available on PATH."""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
    if ffprobe_path:
        AudioSegment.ffprobe = ffprobe_path

    return bool(ffmpeg_path and ffprobe_path)


def convert_to_wav(audio_path: str, for_vosk=False) -> str:
    """
    Converts any audio/video format to optimized WAV.
    Supports: mp4, mkv, m4a, mov, avi, flac, ogg
    """
    if not _configure_ffmpeg():
        raise RuntimeError(
            "FFmpeg is not installed on this server. On Streamlit Cloud, add "
            "a packages.txt file with ffmpeg and redeploy the app."
        )

    try:
        audio = AudioSegment.from_file(audio_path)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg/ffprobe could not be found while reading this audio/video file. "
            "Add ffmpeg to packages.txt and redeploy."
        ) from exc
    
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
    audio_path = f"data/audio/{timestamp}_{os.path.basename(uploaded_file.name)}"
    wav_path = None
    
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        wav_path = convert_to_wav(audio_path, for_vosk=use_vosk)

        if use_vosk:
            transcript = transcribe_with_vosk(wav_path)
        else:
            result = transcribe_with_whisper(wav_path, model_size="medium")
            transcript = result["text"].strip()
    except RuntimeError as exc:
        transcript = f"Error: {exc}"
    finally:
        if os.path.exists(audio_path): os.remove(audio_path)
        if wav_path and os.path.exists(wav_path): os.remove(wav_path)

    return transcript

# ====================== CACHED DIARIZATION PIPELINE ======================
@st.cache_resource
def load_diarization_pipeline(hf_token):
    """Loads the speaker diarization model from Hugging Face"""
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Speaker identification is disabled on this deployment because "
            "pyannote.audio is not installed. Turn off 'Identify speakers' and "
            "process the meeting again."
        ) from exc

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

    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError(
            "Speaker identification needs torchaudio, which is not installed in "
            "the lightweight deployment. Turn off 'Identify speakers' and process "
            "the meeting again."
        ) from exc
    
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
