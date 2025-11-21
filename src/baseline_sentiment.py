#!/usr/bin/env python3
"""
Baseline Sentiment Analysis using Mistral-7B-Instruct (4-bit)
Week 1–2 Deliverable – Luke McCabe
"""
from transformers import pipeline
import json
from datetime import datetime

# Load model (cached after first run)
pipe = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    torch_dtype="auto",
    model_kwargs={"load_in_4bit": True}
)

def get_sentiment(text: str) -> dict:
    prompt = f"<s>[INST] Return only valid JSON with keys 'sentiment' (positive/negative/neutral) and 'confidence' (0–100). Text: \"{text}\" [/INST]"
    result = pipe(prompt, max_new_tokens=60, do_sample=False, temperature=0.1)[0]['generated_text']
    # Extract JSON part after [/INST]
    try:
        json_str = result.split('[/INST]')[-1].strip()
        return json.loads(json_str)
    except:
        return {"sentiment": "error", "confidence": 0, "raw": result}

if __name__ == "__main__":
    headlines = [
        "Apple reports record revenue and beats estimates by 12%",
        "Tesla misses delivery targets for third consecutive quarter",
        "Federal Reserve signals potential rate cut in 2026",
        "NVIDIA shares soar after blockbuster AI chip demand",
        "Bank of America announces 2,000 job cuts amid restructuring"
    ]

    print(f"Baseline Sentiment Analysis – {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    results = []
    for h in headlines:
        out = get_sentiment(h)
        print(f"Text: {h}")
        print(f"→ {out}\n")
        results.append({"text": h, **out})

    # Save results
    with open("data/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to data/baseline_results.json")
