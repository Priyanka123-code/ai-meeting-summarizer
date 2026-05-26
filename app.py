import json
import os
import tempfile
from datetime import datetime

import streamlit as st
import torch
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from utils.export import generate_word_minutes
from utils.llm_analysis import MeetingAnalysis, analyze_meeting
from utils.rag import add_to_rag, rag_search
from utils.transcription import (
    get_diarized_transcript,
    is_vosk_available,
    load_whisper_model,
    transcribe_audio,
    transcribe_with_whisper,
)
from utils.tts import is_tts_available, speak_text
# from unsloth import FastLanguageModel

load_dotenv()

HAS_GPU = torch.cuda.is_available()
TRANSCRIPT_DIR = "data/transcripts"
AUDIO_DIR = "data/audio"


# @st.cache_resource
# def load_finetuned_model():
#     if not HAS_GPU:
#         return None, None

#     from unsloth import FastLanguageModel

#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name="priyankas123/priyanka-meeting-llama3-8b",
#         max_seq_length=2048,
#         load_in_4bit=True,
#     )
#     FastLanguageModel.for_inference(model)
#     return model, tokenizer


def save_uploaded_file(uploaded_file, prefix="meeting"):
    os.makedirs(AUDIO_DIR, exist_ok=True)
    safe_name = os.path.basename(uploaded_file.name)
    temp_path = os.path.join(AUDIO_DIR, f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}")
    with open(temp_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return temp_path


def build_analysis_with_finetuned_model(transcript, model, tokenizer):
    prompt = f"### Transcript:\n{transcript}\n\n### Structured Meeting Minutes:"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512)
    raw_result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    minutes_text = raw_result.split("### Structured Meeting Minutes:")[-1].strip()

    return MeetingAnalysis(
        summary=minutes_text,
        key_decisions=["Decisions included in summary"],
        open_questions=[],
        action_items=[],
    )


def persist_meeting(transcript_data, analysis):
    transcript_data["analysis"] = analysis.model_dump()
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    path = os.path.join(TRANSCRIPT_DIR, f"{transcript_data['meeting_id']}.json")
    with open(path, "w") as file:
        json.dump(transcript_data, file, indent=2)
    add_to_rag(transcript_data, analysis)


def load_meeting_record(filename):
    with open(os.path.join(TRANSCRIPT_DIR, filename), "r") as file:
        data = json.load(file)

    st.session_state.transcript = data["transcript"]
    st.session_state.transcript_data = data
    if "analysis" in data:
        st.session_state.analysis = MeetingAnalysis(**data["analysis"])
    elif "analysis" in st.session_state:
        del st.session_state.analysis


def render_action_items(analysis):
    if not analysis.action_items:
        st.caption("No action items were detected.")
        return

    for item in analysis.action_items:
        st.markdown(
            f"""
            <div class="action-row">
                <div>
                    <div class="action-task">{item.task}</div>
                    <div class="action-meta">Owner: {item.owner} | Deadline: {item.deadline or "Not specified"}</div>
                </div>
                <span>{item.priority or "Medium"}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.set_page_config(page_title="Meeting Intelligence Suite", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #050505;
  --surface: #101010;
  --surface-strong: #171717;
  --ink: #f5f5f5;
  --muted: #a3a3a3;
  --line: rgba(255,255,255,0.14);
  --primary: #3157d5;
  --primary-2: #0f9f8a;
  --accent: #e0574f;
  --gold: #f4c76b;
  --shadow: none;
}

.stApp {
  background: var(--bg);
  color: var(--ink);
  font-family: 'Inter', sans-serif;
}

.block-container {
  max-width: 1280px;
  padding-top: 1.1rem;
  padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
  background: #050505;
  border-right: 1px solid var(--line);
  box-shadow: none;
}

[data-testid="stSidebar"] * {
  color: var(--ink);
}

.hero {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr);
  gap: 1.2rem;
  align-items: stretch;
  margin: 0.75rem 0 1.6rem;
}

.hero-main, .status-panel, .glass-panel, .library-row {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: none;
  backdrop-filter: none;
}

.hero-main {
  position: relative;
  overflow: hidden;
  padding: 2rem 2.2rem;
  min-height: 250px;
  background:
    linear-gradient(120deg, rgba(20,32,51,0.98), rgba(28,53,96,0.94) 56%, rgba(15,159,138,0.74)),
    linear-gradient(135deg, #142033, #3157d5);
}

.hero-main::after {
  content: "";
  position: absolute;
  inset: auto -5rem -6rem auto;
  width: 18rem;
  height: 18rem;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,0.28), transparent 62%);
}

.hero-copy {
  position: relative;
  z-index: 1;
  max-width: 760px;
}

.eyebrow {
  color: #9be7d7;
  font-size: 0.82rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.hero h1 {
  color: #ffffff;
  font-size: clamp(2.25rem, 4.4vw, 4.25rem);
  line-height: 1.02;
  letter-spacing: 0;
  margin: 0.65rem 0 0.9rem;
}

.hero p {
  color: rgba(255,255,255,0.78);
  font-size: 1.02rem;
  line-height: 1.7;
  max-width: 680px;
}

.muted {
  color: var(--muted);
  font-size: 1rem;
  line-height: 1.58;
}

.hero-steps {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
  margin-top: 1.35rem;
}

.hero-steps span {
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 999px;
  color: #ffffff;
  font-size: 0.84rem;
  font-weight: 800;
  padding: 0.52rem 0.78rem;
}

.status-panel {
  padding: 1rem;
  display: grid;
  gap: 0.75rem;
}

.status-card {
  background: var(--surface-strong);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 1rem;
}

.status-label {
  color: var(--muted);
  font-size: 0.78rem;
  text-transform: uppercase;
  font-weight: 700;
  letter-spacing: 0.08em;
}

.status-value {
  color: var(--ink);
  font-size: 1.45rem;
  font-weight: 800;
  margin-top: 0.25rem;
}

.glass-panel {
  padding: 1.05rem 1.15rem;
  margin: 0.95rem 0;
}

.settings-heading {
  background:
    linear-gradient(90deg, rgba(49,87,213,0.14), rgba(15,159,138,0.1)),
    var(--surface);
  border-color: rgba(49,87,213,0.24);
  margin-bottom: 0.8rem;
}

.section-title {
  color: var(--ink);
  font-size: 1.22rem;
  font-weight: 800;
  margin-bottom: 0.35rem;
}

.stButton > button {
  background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
  color: #ffffff !important;
  border: 0 !important;
  border-radius: 8px !important;
  min-height: 2.9rem;
  font-weight: 800 !important;
  box-shadow: none !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}

.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: none !important;
}

.stTabs [data-baseweb="tab-list"] {
  background: #101010;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.5rem;
  gap: 0.5rem;
  box-shadow: none;
}

.stTabs [data-baseweb="tab"] {
  border-radius: 7px;
  color: var(--muted);
  font-size: 1.02rem;
  font-weight: 700;
  min-height: 3.05rem;
  padding: 0.72rem 1.15rem;
}

.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
  color: #ffffff !important;
}

.stRadio > label, .stTextInput > label {
  color: var(--ink) !important;
  font-weight: 800 !important;
}

div[role="radiogroup"] {
  gap: 0.7rem;
}

div[role="radiogroup"] label {
  background: #171717;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.65rem 0.9rem;
  transition: border-color 0.2s ease, background 0.2s ease;
}

div[role="radiogroup"] label:hover {
  background: #202020;
  border-color: rgba(49,87,213,0.45);
}

input[type="radio"] {
  accent-color: var(--primary);
}

.stTextArea textarea, .stTextInput input {
  background: #0b0b0b !important;
  border: 1px solid var(--line) !important;
  border-radius: 8px !important;
  color: var(--ink) !important;
  box-shadow: none !important;
}

.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: rgba(49,87,213,0.68) !important;
  box-shadow: none !important;
}

div[data-testid="stFileUploader"] {
  background: #101010;
  border: 1px dashed rgba(49,87,213,0.55);
  border-radius: 8px;
  padding: 0.35rem;
}

.decision-pill {
  display: inline-block;
  background: rgba(17,197,166,0.12);
  border: 1px solid rgba(17,197,166,0.28);
  border-radius: 999px;
  padding: 0.55rem 0.8rem;
  margin: 0.25rem 0.3rem 0.25rem 0;
  color: #7df0dc;
  font-weight: 700;
}

.action-row {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: center;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.9rem 1rem;
  margin: 0.55rem 0;
  background: #171717;
}

.action-task {
  font-weight: 800;
  color: var(--ink);
}

.action-meta {
  color: var(--muted);
  font-size: 0.86rem;
  margin-top: 0.2rem;
}

.action-row span {
  background: rgba(244,199,107,0.12);
  color: var(--gold);
  border: 1px solid rgba(244,199,107,0.26);
  border-radius: 999px;
  padding: 0.35rem 0.65rem;
  font-size: 0.78rem;
  font-weight: 800;
}

.mode-status {
  display: inline-flex;
  align-items: center;
  background: rgba(17,197,166,0.12);
  border: 1px solid rgba(17,197,166,0.24);
  border-radius: 999px;
  color: #7df0dc;
  font-weight: 800;
  padding: 0.62rem 0.9rem;
  margin: 0.45rem 0 1rem;
}

.library-row {
  padding: 0.95rem 1rem;
  margin: 0.2rem 0 0.55rem;
}

.library-date {
  color: var(--ink);
  font-weight: 800;
  margin-bottom: 0.18rem;
}

.library-file {
  color: var(--muted);
  font-size: 0.9rem;
}

.stAlert {
  border-radius: 8px;
}

@media (max-width: 900px) {
  .hero {
    grid-template-columns: 1fr;
  }
  .hero-main {
    padding: 1.45rem;
  }
}
</style>
    """,
    unsafe_allow_html=True,
)


meeting_count = 0
if os.path.exists(TRANSCRIPT_DIR):
    meeting_count = len([name for name in os.listdir(TRANSCRIPT_DIR) if name.endswith(".json")])

transcript_status = "Ready" if "transcript" in st.session_state else "Waiting"
analysis_status = "Created" if "analysis" in st.session_state else "Not yet"

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-main">
        <div class="hero-copy">
          <div class="eyebrow">Meeting intelligence workspace</div>
          <h1>From raw calls to clear decisions.</h1>
          <p>
            Upload a meeting, generate structured minutes, search previous conversations,
            and export a polished Word document from one focused workspace.
          </p>
          <div class="hero-steps">
            <span>Upload</span>
            <span>Transcribe</span>
            <span>Analyze</span>
            <span>Search</span>
            <span>Export</span>
          </div>
        </div>
      </div>
      <div class="status-panel">
        <div class="status-card">
          <div class="status-label">Transcript</div>
          <div class="status-value">{transcript_status}</div>
        </div>
        <div class="status-card">
          <div class="status-label">Minutes</div>
          <div class="status-value">{analysis_status}</div>
        </div>
        <div class="status-card">
          <div class="status-label">Archive</div>
          <div class="status-value">{meeting_count} meetings</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

model_options = ["Groq API (Llama-3.3-70b)"]
if HAS_GPU:
    model_options.append("My Fine-Tuned Model (Llama-3.1-8b)")

st.markdown(
    """
    <div class="glass-panel settings-heading">
      <div class="section-title">Workspace Settings</div>
      <div class="muted">Choose the minutes engine and add a Hugging Face token only when speaker identification is needed.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

settings_engine_col, settings_token_col = st.columns([1, 1])
with settings_engine_col:
    model_choice = st.radio("Minutes engine", model_options, horizontal=True)
    if not HAS_GPU:
        st.caption("Local fine-tuned model is hidden because no GPU was detected.")

with settings_token_col:
    hf_token = st.text_input("Hugging Face token", type="password")

ft_model, ft_tokenizer = None, None
# if model_choice.startswith("My Fine-Tuned"):
#     with st.spinner("Loading local model..."):
#         ft_model, ft_tokenizer = load_finetuned_model()
#     st.markdown('<div class="mode-status">Local model ready</div>', unsafe_allow_html=True)
# else:
#     st.markdown('<div class="mode-status">Groq mode active</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="mode-status">Groq mode active</div>',
    unsafe_allow_html=True
)

tab_capture, tab_library, tab_ask = st.tabs(["Capture", "Library", "Meeting Search"])


with tab_capture:
    st.markdown(
        """
        <div class="glass-panel">
          <div class="section-title">Capture Meeting</div>
          <div class="muted">Upload audio or video, then choose between fast offline transcription and richer Whisper processing.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_col, settings_col = st.columns([2, 1])
    with upload_col:
        uploaded_file = st.file_uploader(
            "Meeting file",
            type=["mp3", "wav", "m4a", "mp4", "mkv", "mov"],
        )

    with settings_col:
        enable_diarization = st.checkbox("Identify speakers", value=False)
        vosk_available = is_vosk_available()
        use_vosk = st.checkbox(
            "Fast Vosk offline mode",
            value=False,
            disabled=not vosk_available,
        )
        if not vosk_available:
            st.caption("Fast Vosk mode is disabled in the lightweight deployment.")

    if uploaded_file and st.button("Start Processing", type="primary", use_container_width=True):
        with st.spinner("Transcribing meeting..."):
            if enable_diarization and hf_token:
                temp_path = save_uploaded_file(uploaded_file, prefix="diarize")
                try:
                    model_whisper = load_whisper_model()
                    result = model_whisper.transcribe(temp_path)
                    try:
                        transcript = get_diarized_transcript(temp_path, result["segments"], hf_token)
                    except RuntimeError as exc:
                        transcript = f"Error: {exc}"
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                transcript = transcribe_audio(uploaded_file, use_vosk=use_vosk)

        if transcript:
            st.session_state.transcript = transcript
            st.session_state.transcript_data = {
                "meeting_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "filename": uploaded_file.name,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "transcript": transcript,
            }
            if "analysis" in st.session_state:
                del st.session_state.analysis
            st.rerun()

    if "transcript" in st.session_state:
        st.markdown('<div class="glass-panel"><div class="section-title">Transcript Review</div></div>', unsafe_allow_html=True)
        st.text_area("Transcript text", st.session_state.transcript, height=260, label_visibility="collapsed")

        if st.button("Create Meeting Minutes", type="primary", use_container_width=True):
            if "Error:" in st.session_state.transcript:
                st.error("Cannot generate minutes because transcription failed. Switch mode or fix the model setup first.")
                st.stop()

            if model_choice == "Groq API (Llama-3.3-70b)":
                analysis = analyze_meeting(st.session_state.transcript)
            else:
                with st.spinner("Analyzing with fine-tuned Llama model..."):
                    analysis = build_analysis_with_finetuned_model(
                        st.session_state.transcript,
                        ft_model,
                        ft_tokenizer,
                    )

            st.session_state.analysis = analysis
            persist_meeting(st.session_state.transcript_data, analysis)
            st.rerun()

    if "analysis" in st.session_state:
        analysis = st.session_state.analysis
        st.markdown('<div class="glass-panel"><div class="section-title">Executive Minutes</div></div>', unsafe_allow_html=True)
        st.write(analysis.summary)

        st.markdown("#### Key Decisions")
        if analysis.key_decisions:
            st.markdown(
                "".join([f'<span class="decision-pill">{decision}</span>' for decision in analysis.key_decisions]),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No key decisions were detected.")

        st.markdown("#### Action Items")
        render_action_items(analysis)

        col_read, col_export = st.columns(2)
        with col_read:
            tts_available = is_tts_available()
            if st.button("Listen to Summary", use_container_width=True, disabled=not tts_available):
                speak_text(analysis.summary)
            if not tts_available:
                st.caption("Voice playback is disabled in the lightweight deployment.")
        with col_export:
            if st.button("Export to Word", use_container_width=True):
                path = generate_word_minutes(st.session_state.transcript_data, analysis)
                if path:
                    with open(path, "rb") as word_file:
                        st.download_button(
                            "Download Word Summary",
                            data=word_file,
                            file_name=os.path.basename(path),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                        )
                    st.caption(f"Saved locally: {path}")


with tab_library:
    st.markdown(
        """
        <div class="glass-panel">
          <div class="section-title">Meeting Library</div>
          <div class="muted">Reload saved transcripts and minutes from your local archive.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if os.path.exists(TRANSCRIPT_DIR):
        files = sorted([name for name in os.listdir(TRANSCRIPT_DIR) if name.endswith(".json")], reverse=True)
        if not files:
            st.info("No saved meetings yet.")

        for filename in files:
            with open(os.path.join(TRANSCRIPT_DIR, filename), "r") as file:
                data = json.load(file)

            st.markdown(
                f"""
                <div class="library-row">
                  <div class="library-date">{data.get('date', 'Unknown date')}</div>
                  <div class="library-file">{data.get('filename', 'Unknown file')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Load meeting", key=f"load_{filename}", use_container_width=True):
                load_meeting_record(filename)
                st.rerun()
    else:
        st.info("No history found. Process a meeting to start the archive.")


with tab_ask:
    st.markdown(
        """
        <div class="glass-panel">
          <div class="section-title">Ask the Archive</div>
          <div class="muted">Search across saved meeting memory using text or a recorded question.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "rag_answer" not in st.session_state:
        st.session_state.rag_answer = None

    query_col, voice_col = st.columns([3, 1])
    with query_col:
        query = st.text_input(
            "Question",
            placeholder="Example: What did we decide about the launch plan?",
            key="rag_query_input",
        )
    with voice_col:
        voice_input = st.audio_input("Record question")

    if voice_input:
        with st.spinner("Transcribing your voice..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_voice:
                temp_voice.write(voice_input.getbuffer())
                temp_voice_path = temp_voice.name

            try:
                query_result = transcribe_with_whisper(temp_voice_path)
                query = query_result["text"].strip()
            finally:
                if os.path.exists(temp_voice_path):
                    os.remove(temp_voice_path)

            st.info(f"Heard: {query}")

    if st.button("Search Archive", type="primary", use_container_width=True) and query:
        with st.spinner("Searching vector database..."):
            docs, context = rag_search(query)
            if docs:
                llm_rag = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
                prompt_rag = ChatPromptTemplate.from_template(
                    """
                    Use the following meeting context to answer the question.
                    Context: {context}
                    Question: {question}
                    Answer:
                    """
                )
                chain = prompt_rag | llm_rag
                response = chain.invoke({"question": query, "context": context})
                st.session_state.rag_answer = response.content
            else:
                st.session_state.rag_answer = None
                st.warning("No matching information found.")

    if st.session_state.rag_answer:
        st.markdown('<div class="glass-panel"><div class="section-title">Answer</div></div>', unsafe_allow_html=True)
        st.info(st.session_state.rag_answer)

        tts_available = is_tts_available()
        if st.button("Hear Answer", use_container_width=True, disabled=not tts_available):
            speak_text(st.session_state.rag_answer)
        if not tts_available:
            st.caption("Voice playback is disabled in the lightweight deployment.")
