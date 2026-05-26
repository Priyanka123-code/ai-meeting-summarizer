from unsloth import FastLanguageModel

def analyze_meeting_finetuned(transcript):
    # Load your model from Hugging Face[cite: 13]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "priyankas123/priyanka-meeting-llama3-8b",
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    # Format the prompt exactly like your training[cite: 3, 13]
    prompt = f"### Transcript:\n{transcript}\n\n### Structured Meeting Minutes:"
    
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 256)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]