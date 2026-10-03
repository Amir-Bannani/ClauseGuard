"""
Employment Dataset Processor & Exporter
Refines master CUAD clauses into a targeted 10-class Employment Contract Dataset.
Outputs train.csv, val.csv, test.csv, and label_mapping.json to ml/data/employment_processed/.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

CUAD_TO_EMPLOYMENT_MAP = {
    "Non-Compete": "non_compete",
    "No-Solicit Of Employees": "no_solicit_employees",
    "No-Solicit Of Customers": "no_solicit_customers",
    "Termination For Convenience": "termination_notice",
    "Notice Period To Terminate Renewal": "termination_notice",
    "Ip Ownership Assignment": "ip_assignment",
    "Joint Ip Ownership": "ip_assignment",
    "Non-Disparagement": "confidentiality",
    "Governing Law": "governing_law",
    "Exclusivity": "exclusivity",
    "Cap On Liability": "liability_indemnity",
    "Uncapped Liability": "liability_indemnity"
}

def process_employment_dataset(project_root: Path):
    master_path = project_root / "ml" / "data" / "cuad_processed" / "full_clean_clauses.csv"
    cuad_json_path = project_root / "data" / "CUADv1.json"
    output_dir = project_root / "ml" / "data" / "employment_processed"
    
    print(f"[*] Loading master clauses from: {master_path.resolve()}")
    if not master_path.exists():
        raise FileNotFoundError(f"Master clauses file not found at {master_path}. Run 01_cuad_exploration first.")
        
    df_master = pd.read_csv(master_path)
    
    # Filter to target categories
    df_filtered = df_master[df_master["category"].isin(CUAD_TO_EMPLOYMENT_MAP.keys())].copy()
    df_filtered["employment_category"] = df_filtered["category"].map(CUAD_TO_EMPLOYMENT_MAP)
    
    print(f"[✓] Filtered {len(df_filtered):,} positive employment-relevant clauses.")
    
    # Negative/general samples via character span gaps
    negative_samples = []
    if cuad_json_path.exists():
        print(f"[*] Extracting unannotated text gaps (`other_general`) from: {cuad_json_path.resolve()}")
        with open(cuad_json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        for doc in raw_data.get("data", []):
            title = doc.get("title", "")
            for para in doc.get("paragraphs", []):
                context = para.get("context", "")
                
                spans = []
                for qa in para.get("qas", []):
                    for ans in qa.get("answers", []):
                        st = ans.get("answer_start", -1)
                        txt = ans.get("text", "")
                        if st >= 0 and txt:
                            spans.append((st, st + len(txt)))
                            
                spans = sorted(spans, key=lambda x: x[0])
                
                last_end = 0
                for st, end in spans:
                    if st > last_end + 80:
                        gap_text = context[last_end:st].strip()
                        for snippet in gap_text.split("\n\n"):
                            clean_snip = snippet.strip()
                            if 60 <= len(clean_snip) <= 350 and not clean_snip.startswith("Source:") and not clean_snip.startswith("EXHIBIT"):
                                negative_samples.append({
                                    "contract_title": title,
                                    "category": "General / Unannotated",
                                    "clause_text": clean_snip,
                                    "start_char": -1,
                                    "end_char": -1,
                                    "employment_category": "other_general"
                                })
                    if end > last_end:
                        last_end = end
                        
    df_negatives = pd.DataFrame(negative_samples).drop_duplicates(subset=["clause_text"])
    print(f"[✓] Extracted {len(df_negatives):,} clean `other_general` negative clauses.")
    
    df_employment = pd.concat([df_filtered, df_negatives], ignore_index=True)
    
    # Cleaning & deduplication
    df_employment["clause_text"] = df_employment["clause_text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df_employment = df_employment[df_employment["clause_text"].str.len() >= 15].copy()
    df_employment = df_employment.drop_duplicates(subset=["clause_text", "employment_category"]).copy()
    
    print(f"[✓] Cleaned & deduplicated total dataset size: {len(df_employment):,} clauses.")
    
    # Label mapping
    sorted_categories = sorted(df_employment["employment_category"].unique())
    label2id = {cat: idx for idx, cat in enumerate(sorted_categories)}
    id2label = {idx: cat for idx, cat in enumerate(sorted_categories)}
    df_employment["label_id"] = df_employment["employment_category"].map(label2id)
    
    # Grouped splits (70 / 15 / 15)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(df_employment, groups=df_employment["contract_title"]))
    df_train_val = df_employment.iloc[train_val_idx].copy()
    df_test = df_employment.iloc[test_idx].copy()

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val["contract_title"]))
    df_train = df_train_val.iloc[train_idx].copy()
    df_val = df_train_val.iloc[val_idx].copy()
    
    print("\n[✓] Split summary (0 document leakage):")
    print(f" - Train:      {len(df_train):,} clauses ({df_train['contract_title'].nunique()} contracts)")
    print(f" - Validation: {len(df_val):,} clauses ({df_val['contract_title'].nunique()} contracts)")
    print(f" - Test:       {len(df_test):,} clauses ({df_test['contract_title'].nunique()} contracts)")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(output_dir / "train.csv", index=False)
    df_val.to_csv(output_dir / "val.csv", index=False)
    df_test.to_csv(output_dir / "test.csv", index=False)
    df_employment.to_csv(output_dir / "employment_full_clauses.csv", index=False)
    
    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)
        
    print(f"\n[✓] Exported clean employment dataset to: {output_dir.resolve()}")
    return df_train, df_val, df_test, label2id

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    process_employment_dataset(root)
