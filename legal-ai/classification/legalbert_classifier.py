"""
ClauseGuard — Fine-Tuned LegalBERT Clause Classifier.
Classifies legal contract text into one of 10 employment contract categories.
"""

import os
import torch
import warnings
from typing import Dict, Any, List, Optional, Union

LABEL_MAP = {
    0: "confidentiality",
    1: "exclusivity",
    2: "governing_law",
    3: "ip_assignment",
    4: "liability_indemnity",
    5: "no_solicit_customers",
    6: "no_solicit_employees",
    7: "non_compete",
    8: "other_general",
    9: "termination_notice"
}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class LegalBertClassifier:
    """
    Production-ready classifier leveraging fine-tuned LegalBERT (nlpaueb/legal-bert-base-uncased)
    for multi-class legal clause classification across 10 categories.
    """

    def __init__(
        self,
        model_name_or_path: str = "AmirLGass/clauseguard-legalbert",
        num_labels: int = 10,
        device: Optional[str] = None,
        use_fallback: bool = False
    ):
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fallback = use_fallback
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load_model(self):
        """Loads model weights and tokenizer from HuggingFace Hub or local path."""
        if self.use_fallback:
            print("[*] LegalBertClassifier operating in heuristic fallback mode.")
            self.is_loaded = True
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            print(f"[*] Loading LegalBERT Classifier ({self.model_name_or_path}) on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels
            ).to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"[✓] Successfully loaded LegalBERT Classifier.")
        except Exception as e:
            print(f"[!] Failed to load model from {self.model_name_or_path} ({e}). Switching to heuristic mode.")
            self.use_fallback = True
            self.is_loaded = True

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classifies a single legal clause text.
        Returns dictionary containing predicted category, confidence score, and class probabilities.
        """
        if not self.is_loaded:
            self.load_model()

        if self.use_fallback or self.model is None:
            return self._heuristic_predict(text)

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_class_id = torch.argmax(probs).item()
            confidence = probs[pred_class_id].item()

        category = LABEL_MAP.get(pred_class_id, "other_general")
        probabilities = {LABEL_MAP[i]: probs[i].item() for i in range(self.num_labels)}

        return {
            "category": category,
            "label_id": pred_class_id,
            "confidence": round(confidence, 4),
            "probabilities": probabilities
        }

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classifies a list of legal clauses."""
        return [self.predict(t) for t in texts]

    def _heuristic_predict(self, text: str) -> Dict[str, Any]:
        """Heuristic fallback rule engine for classification."""
        lower = text.lower()
        category = "other_general"
        confidence = 0.85

        if "confidential" in lower or "trade secret" in lower or "proprietary data" in lower:
            category = "confidentiality"
        elif "exclusively" in lower or "exclusive distributor" in lower or "sole distributor" in lower:
            category = "exclusivity"
        elif "governed by" in lower or "jurisdiction" in lower or "choice of law" in lower:
            category = "governing_law"
        elif "assigns" in lower or "intellectual property" in lower or "inventions" in lower:
            category = "ip_assignment"
        elif "indemnify" in lower or "hold harmless" in lower or "defend" in lower:
            category = "liability_indemnity"
        elif "solicit" in lower and ("customer" in lower or "client" in lower or "account" in lower):
            category = "no_solicit_customers"
        elif "solicit" in lower and ("employee" in lower or "personnel" in lower or "engineer" in lower or "hire" in lower):
            category = "no_solicit_employees"
        elif "compete" in lower or "competitors" in lower or "competing business" in lower:
            category = "non_compete"
        elif "terminate" in lower or "notice period" in lower or "without cause" in lower or "for cause" in lower:
            category = "termination_notice"

        label_id = REVERSE_LABEL_MAP.get(category, 8)
        return {
            "category": category,
            "label_id": label_id,
            "confidence": confidence,
            "probabilities": {cat: (0.90 if cat == category else 0.01) for cat in LABEL_MAP.values()}
        }
