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
BASE_OUTPUT_DIR = "privacy_model_adaptive_attacker"
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
        alpha_t = alpha[labels]
        loss = alpha_t * focal_weight * ce_loss
    else:
        loss = focal_weight * ce_loss
    
    return loss.mean()


def calculate_class_weights(labels, num_classes, device):
    labels_tensor = torch.tensor(labels, dtype=torch.long) if isinstance(labels, list) else torch.as_tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(labels_tensor, minlength=num_classes).float()
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights * num_classes / class_weights.sum()
    return class_weights.to(device)


def prepare_dataset(df, dataset_name):

    if 'author_id' in df.columns:
        # Yelp dataset
        classification_type = 'author'
        target_col = 'author_id'
        privatized_col = 'privatized_review'
        
        # Convert to int to avoid serialization errors
        unique_labels = sorted([int(x) for x in df[target_col].unique()])
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: str(label) for idx, label in enumerate(unique_labels)} 
        df['labels'] = df[target_col].map(label2id)
        
    elif 'author' in df.columns:
        # Enron dataset
        classification_type = 'author'
        target_col = 'author'
        privatized_col = 'privatized_text'
        
        unique_labels = sorted(df[target_col].unique())
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        df['labels'] = df[target_col].map(label2id)
        
    else:
        # Trustpilot dataset
        classification_type = 'gender'
        target_col = 'gender'
        privatized_col = 'privatized_text'
        
        label2id = {'F': 0, 'M': 1}
        id2label = {0: 'F', 1: 'M'}
        df['labels'] = df[target_col].map(label2id)
        
    df = df[[privatized_col, 'labels']].copy()
    df = df.rename(columns={privatized_col: 'privatized_text'})
    df = df.reset_index(drop=True)
    
    return df, classification_type, len(label2id), id2label, label2id


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    num_classes = logits.shape[1]
    
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
    }
    
    # Add per-class metrics for all classes
    f1_scores = f1_score(labels, predictions, average=None, zero_division=0)
    precision = precision_score(labels, predictions, average=None, zero_division=0)
    recall = recall_score(labels, predictions, average=None, zero_division=0)
    
    for class_idx in range(num_classes):
        metrics[f"f1_class_{class_idx}"] = f1_scores[class_idx]
        metrics[f"precision_class_{class_idx}"] = precision[class_idx]
        metrics[f"recall_class_{class_idx}"] = recall[class_idx]
    
    if num_classes == 2:
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check processed datasets
    processed_datasets = []
    for result_file in ["adaptive_attacker_yelp_results.csv", "adaptive_attacker_trustpilot_results.csv", "adaptive_attacker_enron_results.csv"]:
        if os.path.exists(result_file):
            results_df = pd.read_csv(result_file)
            if 'dataset_name' in results_df.columns:
                processed_datasets.extend(results_df['dataset_name'].unique().tolist())
    
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
        df, classification_type, num_labels, id2label, label2id = prepare_dataset(df, dataset_name)
        print(f"Type: {classification_type} | Classes: {num_labels} | Samples: {len(df)}")

        # Determine result file based on dataset type
        if classification_type == 'author' and 'author_id' in pd.read_csv(dataset_path).columns:
            results_file = "adaptive_attacker_yelp_results.csv"
        elif classification_type == 'gender':
            results_file = "adaptive_attacker_trustpilot_results.csv"
        elif classification_type == 'author':
            results_file = "adaptive_attacker_enron_results.csv"
        else:
            results_file = "adaptive_attacker_results.csv"
        
        dataset = Dataset.from_pandas(df)
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenize_fn = lambda x: tokenizer(x["privatized_text"], truncation=True, padding="max_length", max_length=512)
        
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
            cols_to_remove_train = [c for c in shuffled_train.column_names if c != "labels"]
            tokenized_train = shuffled_train.map(tokenize_fn, batched=True, remove_columns=cols_to_remove_train)
            tokenized_train.set_format(type="torch")
            
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id)
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
            
            # Calculate class weights for focal loss
            train_labels = [item['labels'] for item in tokenized_train]
            class_weights = calculate_class_weights(train_labels, num_labels, device)
            focal_loss_fn = lambda logits, labels: focal_loss(logits, labels, gamma=2.0, alpha=class_weights)
            
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
            "classification_type": classification_type, 
            "num_classes": num_labels, 
            "num_runs": NUM_RUNS,
            "attacker_type": "adaptive", 
            "train_on": "privatized_text", 
            "test_on": "privatized_text",
        }
        
        for col in metrics_df.columns:
            summary_data[f"mean_{col}"] = metrics_df[col].mean()
            summary_data[f"std_{col}"] = metrics_df[col].std()
        
        df_to_append = pd.DataFrame([summary_data])
        file_exists = os.path.exists(results_file)
        df_to_append.to_csv(results_file, index=False, mode='a', header=not file_exists)
        print(f"Results appended to {results_file}")


if __name__ == "__main__":
    main()