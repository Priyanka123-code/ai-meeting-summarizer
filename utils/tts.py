import streamlit as st
import os
from importlib.util import find_spec


def is_tts_available() -> bool:
    """Returns True when the optional Kokoro voice dependencies are installed."""
    return find_spec("kokoro") is not None and find_spec("soundfile") is not None

def speak_text(text: str, voice: str = "af_heart"):
    """
    Kokoro-82M Text-to-Speech (High Quality Offline TTS)
    Fixed version - handles generator correctly
    """
    if not text or len(text.strip()) < 3:
        st.warning("Nothing to speak")
        return

    if not is_tts_available():
        st.info("Voice playback is optional and is not installed in the lightweight deployment.")
        return

    try:
        from kokoro import KPipeline
        import soundfile as sf

        with st.spinner("Generating voice with Kokoro-82M..."):
            pipeline = KPipeline(lang_code='a')   # 'a' = American English

            # Generate audio (Kokoro returns a generator)
            generator = pipeline(text, voice=voice, speed=1.0)

            # Take the first (and usually only) chunk
            for i, (graphemes, phonemes, audio) in enumerate(generator):
                temp_file = f"temp_kokoro_{abs(hash(text)) % 100000}.wav"
                sf.write(temp_file, audio, 24000)   # 24kHz sample rate

                # Play directly in Streamlit
                st.audio(temp_file, format="audio/wav", autoplay=True)

                # Cleanup
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                break  # We usually need only the first chunk

        st.success("🔊 Kokoro-82M Speaking...")

    except Exception as e:
        st.error(f"Voice playback error: {str(e)}")
