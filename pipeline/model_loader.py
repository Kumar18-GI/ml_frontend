"""
Model loader — Cached, secure loading of RoBERTa and TLSTM models.
Lazy imports to avoid slow startup from transformers source scanning.
"""

import os
import json
import torch
import torch.nn as nn
import streamlit as st
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # paper/

ROBERTA_DIR = BASE_DIR / "roberta_final_checkpoint-20260808T120900Z-1-001" / "roberta_final_checkpoint"
TLSTM_DIR = BASE_DIR / "tlstm"

TLSTM_MODEL_PATH = TLSTM_DIR / "tlstm_model.pth"
TLSTM_CONFIG_PATH = TLSTM_DIR / "tlstm_config.json"
TLSTM_SCALER_PATH = TLSTM_DIR / "tlstm_scaler.pkl"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ──────────────────────────────────────────────
# TLSTM Architecture (must match training code)
# ──────────────────────────────────────────────
class TLSTM(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# ──────────────────────────────────────────────
# Loaders (cached by Streamlit, lazy imports)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_roberta():
    """Load RoBERTa. Imports transformers lazily."""
    from transformers import RobertaForSequenceClassification, RobertaTokenizer

    model_dir = str(ROBERTA_DIR)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"RoBERTa checkpoint not found at: {model_dir}")

    device = get_device()
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


@st.cache_resource(show_spinner=False)
def load_tlstm():
    """Load TLSTM model, config, and scaler. Uses joblib for pickle compatibility."""
    import joblib  # joblib handles cross-version pickle better than raw pickle

    if not TLSTM_MODEL_PATH.exists():
        raise FileNotFoundError(f"TLSTM model not found at: {TLSTM_MODEL_PATH}")

    with open(TLSTM_CONFIG_PATH, "r") as f:
        config = json.load(f)

    device = get_device()
    model = TLSTM(
        input_dim=config["input_dim"],
        hidden_dim=config["lstm_hidden"],
        num_layers=config["lstm_layers"],
    )
    model.load_state_dict(torch.load(str(TLSTM_MODEL_PATH), map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Use joblib to load scaler (handles Python version mismatches)
    scaler = joblib.load(str(TLSTM_SCALER_PATH))
    return model, config, scaler, device


def get_model_status() -> dict:
    return {
        "roberta_available": ROBERTA_DIR.exists() and (ROBERTA_DIR / "model.safetensors").exists(),
        "tlstm_available": TLSTM_MODEL_PATH.exists(),
        "roberta_path": str(ROBERTA_DIR),
        "tlstm_path": str(TLSTM_DIR),
        "device": str(get_device()),
    }
