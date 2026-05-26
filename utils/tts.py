import json
import streamlit as st
import streamlit.components.v1 as components
import os
from importlib.util import find_spec


def is_tts_available() -> bool:
    """Browser speech synthesis is available without Python audio dependencies."""
    return True


def _is_kokoro_available() -> bool:
    return find_spec("kokoro") is not None and find_spec("soundfile") is not None


def _speak_with_browser(text: str):
    safe_text = json.dumps(text)
    components.html(
        f"""
        <button id="play-summary" style="
          width: 100%;
          min-height: 44px;
          border: 0;
          border-radius: 8px;
          background: linear-gradient(135deg, #3157d5, #0f9f8a);
          color: white;
          font: 700 16px system-ui, sans-serif;
          cursor: pointer;
        ">Play Voice</button>
        <script>
          const text = {safe_text};
          const button = document.getElementById("play-summary");

          function speak() {{
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1;
            utterance.pitch = 1;
            window.speechSynthesis.speak(utterance);
          }}

          button.addEventListener("click", speak);
          setTimeout(speak, 250);
        </script>
        """,
        height=56,
    )

def speak_text(text: str, voice: str = "af_heart"):
    """
    Kokoro-82M Text-to-Speech (High Quality Offline TTS)
    Fixed version - handles generator correctly
    """
    if not text or len(text.strip()) < 3:
        st.warning("Nothing to speak")
        return

    if not _is_kokoro_available():
        st.info("Using your browser's built-in voice playback.")
        _speak_with_browser(text)
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
