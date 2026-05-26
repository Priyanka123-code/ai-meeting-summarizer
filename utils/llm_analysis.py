from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
import streamlit as st

# ====================== Pydantic Models ======================
class ActionItem(BaseModel):
    task: str = Field(..., description="Clear description of the task")
    owner: str = Field(..., description="Person responsible")
    deadline: Optional[str] = Field(None, description="Deadline if mentioned")
    priority: Optional[str] = Field("Medium", description="High/Medium/Low")

class MeetingAnalysis(BaseModel):
    summary: str
    key_decisions: List[str]
    open_questions: List[str]
    action_items: List[ActionItem]

def analyze_meeting(transcript: str) -> MeetingAnalysis:
    """
    Analyze transcript using Groq LLM and return structured output
    """
    if not transcript or len(transcript.strip()) < 20:
        st.warning("Transcript too short for analysis")
        return MeetingAnalysis(
            summary="No meaningful transcript provided.",
            key_decisions=[],
            open_questions=[],
            action_items=[]
        )

    with st.spinner("Analyzing meeting with Groq LLM..."):
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )

        prompt = ChatPromptTemplate.from_template("""
You are an expert meeting analyst.
Transcript:
{transcript}

Return only valid JSON with this exact structure:
{{
  "summary": "overall summary in 3-4 sentences",
  "key_decisions": ["decision 1", "decision 2"],
  "open_questions": ["question 1", "question 2"],
  "action_items": [
    {{"task": "task description", "owner": "person name", "deadline": "date or null", "priority": "High/Medium/Low"}}
  ]
}}
""")

        chain = prompt | llm.with_structured_output(MeetingAnalysis)
        analysis = chain.invoke({"transcript": transcript})

        st.success("✅ Groq Analysis Complete!")
        return analysis