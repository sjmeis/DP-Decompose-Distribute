import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
from sentence_transformers import SentenceTransformer, util

DATASET_COLUMNS = {
    "enron": ("text", "privatized_text"),
    "trustpilot": ("text", "privatized_text"),
    "yelp": ("review", "privatized_review")
}

OUTPUT_FILE = "cs_results.csv"
DATASET_FOLDER = "privatized_datasets"

# Use cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading sentence transformer models...")
e1 = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device=device)
e2 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
e3 = SentenceTransformer("thenlper/gte-small", device=device)
print("All models loaded")

# Method copied from DPST/evalutation/CS.py
def calculate_cs(original_texts, private_texts):
    all_scores = []
    for model in [e1, e2, e3]:
        orig_embed = model.encode(original_texts, convert_to_tensor=True, show_progress_bar=False)
        priv_embed = model.encode(private_texts, convert_to_tensor=True, show_progress_bar=False)
        scores = util.pairwise_cos_sim(orig_embed, priv_embed).cpu()
        all_scores.append(float(scores.numpy().mean()))
    
    return round(float(np.mean(all_scores)), 3), [round(s, 3) for s in all_scores]


results_list = []
processed_files = set()
if Path(OUTPUT_FILE).exists():
    existing_df = pd.read_csv(OUTPUT_FILE)
    results_list = existing_df.to_dict('records')
    processed_files = set(existing_df['file'].tolist())
    print(f"Found {len(results_list)} processed datasets")

# Get all csv files from folder
datasets_path = Path(DATASET_FOLDER)
all_files = list(datasets_path.glob("**/*.csv"))

print(f"Found {len(all_files)} CSV files to process\n")


# Start processing
for idx, file_path in enumerate(all_files, 1):
    file_name = file_path.name
    
    # Skip if already processed datasets
    if file_name in processed_files:
        print(f"Skipping {file_name}")
        continue
    
    # Determine dataset type from path, at this position the name of the csv file contains the dataset type (enron, trustpilot, or yelp)
    dataset_type = file_path.parts[1]
    
    orig_col, priv_col = DATASET_COLUMNS[dataset_type]
    
    # Parse filename: privatized_{method}_{chunking}_{epsilon}_{dataset}_10000_sample.csv
    print(f"Processing: {file_name}")
    parts = file_name.replace('.csv', '').replace('_10000_sample', '').split('_')
    distribution = parts[1]
    chunking = parts[2]
    epsilon = parts[3]
    dataset_from_filename = parts[4]

    print(f"Distribution: {distribution}, Chunking: {chunking}, Epsilon: {epsilon}, Dataset: {dataset_from_filename}")
    
    df = pd.read_csv(file_path)
    
    print(f"Calculate CS...")
    mean_score, individual_scores = calculate_cs(
        df[orig_col].tolist(),
        df[priv_col].tolist()
    )
    
    result_row = {
        "file": file_name,
        "dataset": dataset_from_filename,
        "distribution_method": distribution,
        "chunking_method": chunking,
        "epsilon": epsilon,
        "mean_cs": mean_score,
        "model1_score": individual_scores[0],
        "model2_score": individual_scores[1],
        "model3_score": individual_scores[2],
        "processed_at": datetime.now().isoformat()
    }
    
    results_list.append(result_row)
    processed_files.add(file_name)
    
    # Save results after each file
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
print(f"Processing done. Results saved to: {OUTPUT_FILE}")