"""
Inference engine — Sentiment prediction & stock price prediction.
Label mapping from training notebook:
  0 = Negative, 1 = Neutral, 2 = Positive
"""

import torch
import numpy as np
from dataclasses import dataclass

# Correct mapping from notebook: label_map = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_EMOJIS = {0: "😟", 1: "😐", 2: "😊"}


@dataclass
class SentimentResult:
    label: str
    label_id: int
    confidence: float
    probabilities: dict
    emoji: str


def predict_sentiment(text: str, model, tokenizer, device) -> SentimentResult:
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    return SentimentResult(
        label=LABEL_NAMES[pred_id],
        label_id=pred_id,
        confidence=float(probs[pred_id]),
        probabilities={LABEL_NAMES[i]: float(probs[i]) for i in range(len(probs))},
        emoji=LABEL_EMOJIS[pred_id],
    )


def predict_sentiment_batch(texts: list[str], model, tokenizer, device) -> list[SentimentResult]:
    return [predict_sentiment(t, model, tokenizer, device) for t in texts]
