"""
ClauseGuard — HuggingFace Model Upload Script.
Pushes fine-tuned LegalBERT classifier weights to Hugging Face Hub.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_latest_checkpoint(artifacts_dir: Path) -> Path:
    """Finds the highest numbered checkpoint directory or falls back to root."""
    checkpoints = list(artifacts_dir.glob("checkpoint-*"))
    if checkpoints:
        checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
        latest = checkpoints[-1]
        print(f"[*] Found latest checkpoint: {latest.name}")
        return latest
    return artifacts_dir


def push_model_to_hub(repo_id: str, token: str = None):
    """Pushes fine-tuned LegalBERT model and tokenizer to HuggingFace Hub."""
    project_root = Path(__file__).resolve().parents[2]
    artifacts_dir = project_root / "ml" / "artifacts" / "legalbert_classifier"

    if not artifacts_dir.exists():
        print(f"[!] Error: Model artifacts directory not found at {artifacts_dir}")
        sys.exit(1)

    # Automatically resolve token from workspace environment variables if not passed explicitly
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if hf_token:
        print("[*] Using HuggingFace Token detected in environment variables.")

    model_path = get_latest_checkpoint(artifacts_dir)
    print(f"[*] Loading model from: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        sys.exit(1)

    print(f"[*] Pushing model to Hugging Face Hub: https://huggingface.co/{repo_id} ...")
    
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token

    model.push_to_hub(repo_id, **kwargs)
    tokenizer.push_to_hub(repo_id, **kwargs)

    print("\n" + "=" * 70)
    print(f"[✓] SUCCESS! Model successfully published to Hugging Face Hub.")
    print(f"    Hub URL: https://huggingface.co/{repo_id}")
    print("=" * 70)
    print("\nTo use this model in your legal-ai classifier, set:")
    print(f'   LegalBertClassifier(model_name_or_path="{repo_id}")')
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push fine-tuned LegalBERT classifier to HuggingFace Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repository ID (e.g. 'username/clauseguard-legalbert')")
    parser.add_argument("--token", type=str, default=None, help="Optional Hugging Face Access Token (defaults to HF_TOKEN env var)")

    args = parser.parse_args()
    push_model_to_hub(args.repo_id, args.token)
