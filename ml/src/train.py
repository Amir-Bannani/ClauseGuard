"""
ClauseGuard LegalBERT Classifier Training Script
Fine-tunes 'nlpaueb/legal-bert-base-uncased' on the 10-class Employment Contract Dataset using DataCollatorWithPadding.
Saves fine-tuned model weights to ml/artifacts/legalbert_classifier/.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def train_legalbert(project_root: Path):
    data_dir = project_root / "ml" / "data" / "employment_processed"
    output_dir = project_root / "ml" / "artifacts" / "legalbert_classifier"
    
    print(f"[*] Reading dataset from: {data_dir.resolve()}")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    with open(data_dir / "label_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
        
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    label2id = {k: int(v) for k, v in mapping["label2id"].items()}
    
    # Prepare HuggingFace Datasets
    def prep_df(df):
        return df.rename(columns={"clause_text": "text", "label_id": "labels"})[["text", "labels"]]
        
    train_dataset = Dataset.from_pandas(prep_df(train_df), preserve_index=False)
    val_dataset = Dataset.from_pandas(prep_df(val_df), preserve_index=False)
    test_dataset = Dataset.from_pandas(prep_df(test_df), preserve_index=False)
    
    model_name = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)
        
    tokenized_train = train_dataset.map(tokenize_fn, batched=True)
    tokenized_val = val_dataset.map(tokenize_fn, batched=True)
    tokenized_test = test_dataset.map(tokenize_fn, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
        m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        return {"accuracy": acc, "f1": w_f1, "macro_f1": m_f1, "precision": w_prec, "recall": w_rec}
        
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    print("[*] Starting LegalBERT training with DataCollatorWithPadding...")
    trainer.train()
    
    print("[*] Evaluating on test set...")
    predictions = trainer.predict(tokenized_test)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(labels, preds, target_names=target_names, digits=4)
    print("\n=== Classification Report ===")
    print(report)
    
    # Save model & tokenizer
    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")
    print(f"\n[✓] Saved fine-tuned LegalBERT model to: {(output_dir / 'final_model').resolve()}")

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    train_legalbert(root)
