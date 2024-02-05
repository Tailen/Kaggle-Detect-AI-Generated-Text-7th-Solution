import numpy as np
import pandas as pd
import gc
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from roben.utils import Clustering
from roben.recoverer import ClusterRepRecoverer
from tqdm import tqdm


# Initialize tqdm
tqdm.pandas()
# Paths
DATASET_PATH = "../datasets/feedback.csv"
CHECKPOINT_PATH = "../models/deberta-v3-xsmall"
# Model parameters
METRIC_NAME = "roc_auc"
MODEL_NAME = "deberta-xsmall"
MAX_LEN = 1024
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 1


# Import and clean dataset
def import_dataset(dataset="../datasets/combined.csv"):
    train = pd.read_csv(dataset)
    train = train.drop_duplicates(subset=["text"])
    train = train.sample(frac=1, random_state=42)
    train.reset_index(drop=True, inplace=True)
    X = train["text"]
    y = train["generated"]
    # Map texts to robust encoded text
    cache_dir = ".cache/"
    clustering = Clustering.from_pickle("../clusterers/agglomerative_cluster_gamma0.2.pkl",
                                        max_num_possibilities=None)
    recoverer = ClusterRepRecoverer(cache_dir, clustering)
    X = X.progress_apply(lambda x: recoverer.recover(x))
    print("Dataset shape:", train.shape)
    return X, y

# Train and evaluate DeBERTa model
def deberta(X_train, y_train, X_test, y_test):
    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_PATH, num_labels=2
    )

    # Build train and validation Datasets
    def preprocess_func(row):
        return tokenizer(
            row["text"],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )

    train = Dataset.from_pandas(pd.DataFrame({"text": X_train, "label": y_train}))
    test = Dataset.from_pandas(pd.DataFrame({"text": X_test, "label": y_test}))
    train_enc = train.map(preprocess_func, batched=True)
    test_enc = test.map(preprocess_func, batched=True)

    # Build trainer
    num_steps = len(X_train) * NUM_EPOCHS // (TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS)
    args = TrainingArguments(
        output_dir=f"../models/{MODEL_NAME}-finetuned",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=num_steps // 3,
        save_steps=num_steps // 3,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_NAME,
        greater_is_better=True,
        save_total_limit=3,
        fp16=True,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        auc = roc_auc_score(labels, probs[:, 1], multi_class="ovr")
        return {"roc_auc": auc}

    trainer = Trainer(
        model,
        args,
        train_dataset=train_enc,
        eval_dataset=test_enc,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()
    logits = trainer.predict(test_enc).predictions
    y_pred = (np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True))[:, 1]
    return roc_auc_score(y_test, y_pred)


if __name__ == "__main__":
    # Import dataset
    X, y = import_dataset(DATASET_PATH)

    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Train and evaluate model
    auc = deberta(X_train, y_train, X_test, y_test)
    print(f"ROC AUC: {auc:.5f}")

    # Clean up
    del X, y, X_train, X_test, y_train, y_test
    gc.collect()
    torch.cuda.empty_cache()
