"""
CUAD Dataset Processor & Exporter
Loads local CUADv1.json, flattens question/answer text spans into (clause_text, category) pairs,
cleans noise, deduplicates, performs document-grouped 0-leakage train/val/test splits,
and exports clean CSV files and label_mapping.json for LegalBERT fine-tuning.
"""

import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def process_cuad(cuad_json_path: Path, output_dir: Path):
    print(f"[*] Reading CUAD dataset from: {cuad_json_path.resolve()}")
    if not cuad_json_path.exists():
        raise FileNotFoundError(f"CUAD dataset not found at {cuad_json_path}")
        
    with open(cuad_json_path, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
        
    records = []
    for doc in data_json.get("data", []):
        title = doc.get("title", "")
        for para in doc.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                q_id = qa.get("id", "")
                question = qa.get("question", "")
                
                if "__" in q_id:
                    category = q_id.split("__")[-1].replace("_", " ").strip()
                else:
                    category = question
                    
                answers = qa.get("answers", [])
                ans_texts = [a.get("text", "") for a in answers]
                ans_starts = [a.get("answer_start", -1) for a in answers]
                
                records.append({
                    "title": title,
                    "context": context,
                    "id": q_id,
                    "question": question,
                    "category": category,
                    "answers": {"text": ans_texts, "answer_start": ans_starts},
                    "has_answer": len(ans_texts) > 0 and any(len(t.strip()) > 0 for t in ans_texts)
                })
                
    df_raw = pd.DataFrame(records)
    print(f"[✓] Extracted {len(df_raw):,} SQuAD Q&A records across {df_raw['title'].nunique()} contracts.")
    
    # Flatten into clause-level entries
    df_positive = df_raw[df_raw["has_answer"]].copy()
    clauses = []
    
    for idx, row in df_positive.iterrows():
        contract_title = row["title"]
        category = row["category"]
        answers = row["answers"]
        
        texts = answers.get("text", [])
        starts = answers.get("answer_start", [])
        
        for text_span, start_pos in zip(texts, starts):
            if text_span and text_span.strip():
                clauses.append({
                    "contract_title": contract_title,
                    "category": category,
                    "clause_text": text_span.strip(),
                    "start_char": int(start_pos),
                    "end_char": int(start_pos + len(text_span))
                })
                
    df_clauses = pd.DataFrame(clauses)
    print(f"[✓] Extracted {len(df_clauses):,} positive clause text spans.")
    
    # Clean whitespace and filter noise (< 15 chars)
    df_clauses["clause_text"] = df_clauses["clause_text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df_clean = df_clauses[df_clauses["clause_text"].str.len() >= 15].copy()
    
    # Deduplicate exact (clause_text, category) pairs
    df_dedup = df_clean.drop_duplicates(subset=["clause_text", "category"]).copy()
    print(f"[✓] Cleaned & Deduplicated dataset size: {len(df_dedup):,} clauses.")
    
    # Label mapping
    unique_cats = sorted(df_dedup["category"].unique())
    label2id = {cat: idx for idx, cat in enumerate(unique_cats)}
    id2label = {idx: cat for idx, cat in enumerate(unique_cats)}
    df_dedup["label_id"] = df_dedup["category"].map(label2id)
    
    # Grouped Train / Val / Test split by contract_title
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(df_dedup, groups=df_dedup["contract_title"]))
    df_train_val = df_dedup.iloc[train_val_idx].copy()
    df_test = df_dedup.iloc[test_idx].copy()

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val["contract_title"]))
    df_train = df_train_val.iloc[train_idx].copy()
    df_val = df_train_val.iloc[val_idx].copy()
    
    print("\n[✓] Split summary (0 document leakage):")
    print(f" - Train:      {len(df_train):,} clauses ({df_train['contract_title'].nunique()} contracts)")
    print(f" - Validation: {len(df_val):,} clauses ({df_val['contract_title'].nunique()} contracts)")
    print(f" - Test:       {len(df_test):,} clauses ({df_test['contract_title'].nunique()} contracts)")
    
    # Export files
    output_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(output_dir / "train.csv", index=False)
    df_val.to_csv(output_dir / "val.csv", index=False)
    df_test.to_csv(output_dir / "test.csv", index=False)
    df_dedup.to_csv(output_dir / "full_clean_clauses.csv", index=False)
    
    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)
        
    print(f"\n[✓] Exported clean dataset files to: {output_dir.resolve()}")
    return df_train, df_val, df_test, label2id

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    cuad_path = project_root / "data" / "CUADv1.json"
    output_path = project_root / "ml" / "data" / "cuad_processed"
    process_cuad(cuad_path, output_path)
