import numpy as np
import pandas as pd
import gc
import torch
import os
import pickle
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# Paths
DATASET_PATH = "../datasets/slimpajama/combined.pkl"
CHECKPOINT_PATH = "../models/deberta-v3-base"
# Model parameters
METRIC_NAME = "roc_auc"
MODEL_NAME = "deberta-base"
MAX_LEN = 1024
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
NUM_EPOCHS = 1


# Import and clean dataset
def import_dataset(dataset):
    if dataset.endswith(".pkl"):
        train = pd.read_pickle(dataset)
    elif dataset.endswith(".csv"):
        train = pd.read_csv(dataset)
    train = train.dropna(subset=["text"])
    train = train.drop_duplicates(subset=["text"])
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    X = train["text"]
    if "label" in train.columns:
        y = train["label"]
    else:
        y = train["generated"]
    print("Dataset shape:", train.shape)
    return X, y


# Train and evaluate DeBERTa model
def deberta(X_train, y_train, X_test, y_test):
    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_PATH,
        num_labels=2,
        max_position_embeddings=512,
    )

    # Build train and validation Datasets
    def preprocess_func(row):
        return tokenizer(
            row["text"],
            truncation=True,
            max_length=MAX_LEN,
        )

    train = Dataset.from_pandas(pd.DataFrame({"text": X_train, "label": y_train}))
    test = Dataset.from_pandas(pd.DataFrame({"text": X_test, "label": y_test}))
    enc_cache_file = f"{CHECKPOINT_PATH}/enc_cache_{MAX_LEN}.pkl"
    if os.path.exists(enc_cache_file):
        train_enc, test_enc = pickle.load(open(enc_cache_file, "rb"))
    else:
        train_enc = train.map(preprocess_func, batched=True, remove_columns=["text"])
        test_enc = test.map(preprocess_func, batched=True, remove_columns=["text"])
        train_enc.set_format(type="torch")
        test_enc.set_format(type="torch")
        pickle.dump((train_enc, test_enc), open(enc_cache_file, "wb"))

    # Build trainer
    num_steps = len(X_train) * NUM_EPOCHS // (TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS)
    training_args = TrainingArguments(
        output_dir=f"../models/{MODEL_NAME}-finetuned",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=num_steps // 5,
        save_steps=num_steps // 5,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        label_smoothing_factor=LABEL_SMOOTHING,
        num_train_epochs=NUM_EPOCHS,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_NAME,
        greater_is_better=True,
        save_total_limit=1,
        fp16=True,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        # roc auc
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        auc = roc_auc_score(labels, probs[:, 1], multi_class="ovr")

        # accuracy
        p_metric = evaluate.load("precision")
        r_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")
        acc_metric = evaluate.load("accuracy")

        preds = np.argmax(logits, axis=-1)
        precision = p_metric.compute(predictions=preds, references=labels)["precision"]
        recall = r_metric.compute(predictions=preds, references=labels)["recall"]
        f1 = f1_metric.compute(predictions=preds, references=labels)["f1"]
        acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]

        return {
            "roc_auc": auc,
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "accuracy": acc,
        }

    model = model.cuda()
    print(f"Model loaded on {model.device}")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_enc,
        eval_dataset=test_enc,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()
    # logits = trainer.predict(test_enc).predictions
    # y_pred = (np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True))[:, 1]
    # return roc_auc_score(y_test, y_pred)


if __name__ == "__main__":
    # Import dataset
    X, y = import_dataset(DATASET_PATH)

    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, stratify=y
    )

    # Train model
    deberta(X_train, y_train, X_test, y_test)

    # Clean up
    del X, y, X_train, X_test, y_train, y_test
    gc.collect()
    torch.cuda.empty_cache()
