"""
SST-5 Sentiment Classifier — Inference
Run after training:  python inference.py
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR  = "./sst5_model"   # path where trainer saved the model
MAX_LEN    = 128
LABEL_NAMES = ["very negative", "negative", "neutral", "positive", "very positive"]
EMOJI = ["😡", "😞", "😐", "🙂", "🤩"]


def load_model(model_dir: str = MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def predict(texts: list[str], tokenizer, model, device) -> list[dict]:
    """
    Run inference on a list of strings.
    Returns a list of dicts: {text, label, label_id, confidence, scores}
    """
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred_ids = np.argmax(probs, axis=-1)

    results = []
    for text, pid, prob_row in zip(texts, pred_ids, probs):
        results.append({
            "text":       text,
            "label":      LABEL_NAMES[pid],
            "emoji":      EMOJI[pid],
            "label_id":   int(pid),
            "confidence": float(prob_row[pid]),
            "scores":     {l: float(p) for l, p in zip(LABEL_NAMES, prob_row)},
        })
    return results


def print_result(r: dict):
    bar = "█" * int(r["confidence"] * 20)
    print(f"\n  Text       : {r['text'][:80]}")
    print(f"  Prediction : {r['emoji']}  {r['label'].upper()}  ({r['confidence']*100:.1f}%)")
    print(f"  Confidence : [{bar:<20}]")
    print("  All scores :")
    for label, score in r["scores"].items():
        filled = int(score * 15)
        print(f"    {label:<16} {'▪'*filled}{'·'*(15-filled)} {score*100:5.1f}%")


# ── Demo ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading model …")
    tokenizer, model, device = load_model()
    print(f"Running on: {device}\n")

    demo_sentences = [
        "This is an absolutely stunning masterpiece — one of the best films I've ever seen.",
        "A dull and predictable movie that wasted two hours of my life.",
        "It was okay, nothing special but not terrible either.",
        "The acting was decent but the plot had several glaring holes.",
        "A triumph of storytelling, visually breathtaking and emotionally powerful.",
        "Boring, unimaginative, and painfully slow.",
        "I have mixed feelings — some scenes shine while others fall completely flat.",
    ]

    print("=" * 60)
    print("SST-5 Sentiment Predictions")
    print("=" * 60)
    results = predict(demo_sentences, tokenizer, model, device)
    for r in results:
        print_result(r)

    print("\n" + "=" * 60)
    print("Interactive mode  (type 'quit' to exit)")
    print("=" * 60)
    while True:
        text = input("\nEnter a sentence: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        r = predict([text], tokenizer, model, device)[0]
        print_result(r)
