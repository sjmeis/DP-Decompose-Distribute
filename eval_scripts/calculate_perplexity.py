#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss

DATASET_COLUMNS = {
    "enron": ("text", "privatized_text"),
    "trustpilot": ("text", "privatized_text"),
    "yelp": ("review", "privatized_review")
}

OUTPUT_FILE = "perplexity_results.csv"
DATASET_FOLDER = "privatized_datasets"

# Use cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading model...")

ppl_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
print("Model loaded!")
    
# Perplexity code copied from DPST/DPST.py
def compute_ppl(predictions, batch_size=16, add_start_token=True, max_length=32):
    
    if ppl_tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(ppl_tokenizer.special_tokens_map_extended.values())
        assert len(existing_special_tokens) > 0, "Model must have at least one special token"
        ppl_tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        assert ppl_tokenizer.bos_token is not None, "Model must have a BOS token"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = ppl_tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in range(0, len(encoded_texts), batch_size):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[ppl_tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = ppl_model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


results_list = []
processed_files = set()

# Look for results file, if it exists, skip processed datasets. Else process every dataset
if Path(OUTPUT_FILE).exists():
    existing_df = pd.read_csv(OUTPUT_FILE)
    results_list = existing_df.to_dict('records')
    processed_files = set(existing_df['file'].tolist())
    print(f"Found {len(results_list)} processed datasets")


# Get all csv files from folder
datasets_path = Path(DATASET_FOLDER)
all_files = list(datasets_path.glob("**/*.csv"))

print(f"Found {len(all_files)} datasets to process")


# Start processing
for idx, file_path in enumerate(all_files, 1):
    file_name = file_path.name
    
    # Skip if already processed datasets
    if file_name in processed_files:
        print(f"Skipping {file_name}")
        continue
    
    # Parse filename: privatized_{distribution}_{chunking}_{epsilon}_{dataset}_10000_sample.csv
    print(f"Processing: {file_name}")
    parts = file_name.replace('.csv', '').replace('_10000_sample', '').split('_')
    distribution = parts[1]
    chunking = parts[2]
    epsilon = parts[3]
    dataset_from_filename = parts[4]
    
    print(f"Distribution: {distribution}, Chunking: {chunking}, Epsilon: {epsilon}, Dataset: {dataset_from_filename}")
    
    # Determine dataset type from path, at this position the name of the csv file contains the dataset type (enron, trustpilot, or yelp)
    dataset_type = file_path.parts[1]
    orig_col, priv_col = DATASET_COLUMNS[dataset_type]
    
    df = pd.read_csv(file_path)
    
    print(f"Calculate perplexity...")
    ppl_results = compute_ppl(predictions=df[priv_col].tolist(), batch_size=16, max_length=32)
    
    mean_ppl = ppl_results["mean_perplexity"]
    
    result_row = {
        "file": file_name,
        "dataset": dataset_from_filename,
        "distribution": distribution,
        "chunking_method": chunking,
        "epsilon": epsilon,
        "mean_perplexity": round(mean_ppl, 3),
        "processed_at": datetime.now().isoformat()
    }
    
    results_list.append(result_row)
    processed_files.add(file_name)
    
    # Save results after each file
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(OUTPUT_FILE, index=False)

print(f"Processing done. Results saved to: {OUTPUT_FILE}")