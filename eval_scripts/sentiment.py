import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import os
import glob
import gc
import shutil
from datetime import datetime

MODEL_NAME = "microsoft/deberta-v3-base"
DATASET_FOLDER = "privatized_datasets"
BASE_OUTPUT_DIR = "sentiment_model_deberta"
RESULTS_FILE = "sentiment_analysis_scores.csv"
BATCH_SIZE = 8
TEST_SPLIT_SIZE = 0.1
NUM_RUNS = 3
SPLIT_SEED = 42


class CustomTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def focal_loss(logits, labels, gamma=2.0, alpha=None):
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_weight = (1 - pt) ** gamma
    
    if alpha is not None:
        loss = alpha * focal_weight * ce_loss
    else:
        loss = focal_weight * ce_loss
    
    return loss.mean()


def save_results_to_csv(data, filename):
    df_to_append = pd.DataFrame([data])
    file_exists = os.path.exists(filename)
    df_to_append.to_csv(filename, index=False, mode='a', header=not file_exists)


def prepare_dataset(df):
    if "review" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"review": "text"})
    
    if "sentiment_id" in df.columns:
        label_col = "sentiment_id"
    elif "sentiment" in df.columns:
        label_col = "sentiment"
    else:
        label_col = "label"
    
    if label_col != "labels":
        df = df.rename(columns={label_col: "labels"})
    
    non_null_label = df["labels"].dropna().iloc[0]
    if isinstance(non_null_label, str):
        label_map = {"positive": 1, "negative": 0}
        df["labels"] = df["labels"].str.lower().str.strip().map(label_map)
    else:
        df["labels"] = df["labels"].astype(int)
    
    df = df[["text", "labels"]].copy()
    df = df.reset_index(drop=True)
    
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
    }
    
    # Add per-class metrics
    f1_scores = f1_score(labels, predictions, average=None, zero_division=0)
    precision = precision_score(labels, predictions, average=None, zero_division=0)
    recall = recall_score(labels, predictions, average=None, zero_division=0)
    
    metrics["f1_negative"] = f1_scores[0]
    metrics["f1_positive"] = f1_scores[1]
    metrics["precision_negative"] = precision[0]
    metrics["precision_positive"] = precision[1]
    metrics["recall_negative"] = recall[0]
    metrics["recall_positive"] = recall[1]
    
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check processed datasets
    processed_datasets = []
    if os.path.exists(RESULTS_FILE):
        results_df = pd.read_csv(RESULTS_FILE)
        if 'dataset_name' in results_df.columns:
            processed_datasets = results_df['dataset_name'].unique().tolist()
    
    if processed_datasets:
        print(f"Found {len(processed_datasets)} processed datasets.")
    
    dataset_paths = glob.glob(os.path.join(DATASET_FOLDER, "*.csv"))
    
    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        
        if dataset_name in processed_datasets:
            print(f"Skipping {dataset_name}")
            continue
        
        print(f"Processing: {dataset_name}...")
        
        df = pd.read_csv(dataset_path)
        df = prepare_dataset(df)
        
        dataset = Dataset.from_pandas(df)
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenize_fn = lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512)
        
        dataset_split = dataset.train_test_split(test_size=TEST_SPLIT_SIZE, shuffle=True, seed=SPLIT_SEED)
        original_train = dataset_split["train"]
        test_dataset = dataset_split["test"]
        
        tokenized_test = test_dataset.map(tokenize_fn, batched=True)
        
        all_run_metrics = []
        
        for i in range(NUM_RUNS):
            print(f"Run {i+1}/{NUM_RUNS}")
            
            # Shuffle dataset
            current_seed = SPLIT_SEED + i
            shuffled_train = original_train.shuffle(seed=current_seed)
            tokenized_train = shuffled_train.map(tokenize_fn, batched=True)
            tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, 
                num_labels=2, 
                id2label={0: "NEGATIVE", 1: "POSITIVE"}, 
                label2id={"NEGATIVE": 0, "POSITIVE": 1}
            )
            model.to(device)
            
            run_output_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name.replace('.csv', ''), f"run_{i+1}")
            
            # Standard parameters for Deberta, except for epoch
            training_args = TrainingArguments(
                output_dir=run_output_dir, 
                num_train_epochs=1, 
                learning_rate=2e-5, 
                warmup_steps=500,
                per_device_train_batch_size=BATCH_SIZE, 
                per_device_eval_batch_size=BATCH_SIZE,
                logging_dir=f"./logs/{dataset_name.replace('.csv', '')}/run_{i+1}", 
                seed=current_seed, 
                save_strategy="no"
            )
            
            focal_loss_fn = lambda logits, labels: focal_loss(logits, labels, gamma=2.0, alpha=0.25)
            
            trainer = CustomTrainer(
                model=model, 
                args=training_args, 
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test, 
                tokenizer=tokenizer, 
                compute_metrics=compute_metrics, 
                loss_fn=focal_loss_fn
            )
            
            trainer.train()
            eval_results = trainer.evaluate()
            
            print(f"Run {i+1}:")
            print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
            print(f"F1 Macro: {eval_results['eval_f1_macro']:.4f}")
            print(f"F1 Weighted: {eval_results['eval_f1_weighted']:.4f}")
            print(f"F1 Micro: {eval_results['eval_f1_micro']:.4f}")
            print(f"F1 Positive: {eval_results['eval_f1_positive']:.4f}")
            print(f"F1 Negative: {eval_results['eval_f1_negative']:.4f}")
            
            run_metrics = {key.replace('eval_', ''): value for key, value in eval_results.items()}
            all_run_metrics.append(run_metrics)
            
            # Delete model after each run and clear cuda cache if cuda was used
            del model, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if os.path.exists(run_output_dir):
                shutil.rmtree(run_output_dir)
        
        metrics_df = pd.DataFrame(all_run_metrics)
        
        summary_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            "model_name": MODEL_NAME, 
            "dataset_name": dataset_name,
            "num_runs": NUM_RUNS,
        }
        
        for col in metrics_df.columns:
            summary_data[f"mean_{col}"] = metrics_df[col].mean()
            summary_data[f"std_{col}"] = metrics_df[col].std()
        
        save_results_to_csv(summary_data, RESULTS_FILE)
        print(f"Results appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()