import pandas as pd
import numpy as np
import torch
import gc
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from spellchecker import SpellChecker
from roben.utils import Clustering
from roben.recoverer import ClusterRepRecoverer

# Import datasets
test = pd.read_csv("../datasets/feedback.csv")
test = test.sample(frac=0.1).reset_index(drop=True)
X_test = test["text"]
y_test = test["generated"]

# Get index of texts with more than 20 typos
tqdm.pandas()
spell = SpellChecker()


def get_num_typo(text):
    words = re.sub("[^\w]", " ", text).split()
    return len(spell.unknown(words))


typo_idx = X_test.progress_apply(lambda x: get_num_typo(x)) > 20
X_test_typo = X_test[typo_idx]
X_test_clean = X_test[~typo_idx]

# Map texts to robust encoded text
cache_dir = ".cache/"
clustering = Clustering.from_pickle(
    "../clusterers/agglomerative_cluster_gamma0.2.pkl", max_num_possibilities=None
)
recoverer = ClusterRepRecoverer(cache_dir, clustering)
X_test_typo = X_test_typo.apply(lambda x: recoverer.recover(x))

# Inference
ROBEN_CHECKPOINT_PATH = "../models/deberta-xsmall-finetuned/checkpoint-1533"
CLEAN_CHECKPOINT_PATH = "../models/deberta-xsmall-finetuned/checkpoint-510"
MAX_LEN = 1024
BATCH_SIZE = 32


def inference(checkpoint_path, X):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to(
        "cuda"
    )

    y_pred = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            inputs = tokenizer(
                X[i : i + BATCH_SIZE].tolist(),
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to("cuda")
            logits = model(**inputs).logits.cpu().numpy()
            y_pred.extend(
                (np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True))[:, 1]
            )
    gc.collect()
    torch.cuda.empty_cache()
    return y_pred


# Combine predictions based on typo_idx
typo_pred = inference(ROBEN_CHECKPOINT_PATH, X_test_typo)
clean_pred = inference(CLEAN_CHECKPOINT_PATH, X_test_clean)
y_pred = np.zeros(len(X_test))
y_pred[typo_idx] = typo_pred
y_pred[~typo_idx] = clean_pred

print(roc_auc_score(y_test, y_pred))
