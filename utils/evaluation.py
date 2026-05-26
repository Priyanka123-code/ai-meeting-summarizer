from jiwer import wer
from rouge_score import rouge_scorer
import pandas as pd

def calculate_stt_metrics(reference_text, hypothesis_text):
    """
    Calculates Word Error Rate (WER). 
    Lower is better (0.0 is perfect).
    """
    if not reference_text or not hypothesis_text:
        return 1.0
    error_rate = wer(reference_text, hypothesis_text)
    return round(error_rate, 4)

def calculate_llm_metrics(reference_summary, generated_summary):
    """
    Calculates ROUGE scores for summarization.
    Higher is better (1.0 is perfect).
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    
    return {
        "ROUGE-1": round(scores['rouge1'].fmeasure, 4),
        "ROUGE-L": round(scores['rougeL'].fmeasure, 4)
    }