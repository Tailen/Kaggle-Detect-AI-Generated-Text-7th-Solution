import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

# Import datasets
test = pd.read_csv("../datasets/train_v2_drcat_02.csv")
test_generated = test[test["label"] == 1]
test_human = test[test["label"] == 0].sample(n=test_generated.shape[0], random_state=42)
test = (
    pd.concat([test_human, test_generated])
    .sample(n=6000, random_state=42)
    .reset_index(drop=True)
)
X_test = test["text"]
y_test = test["label"]

# Inference
CHECKPOINT_PATH = "../models/deberta-large-finetuned-512/deberta-large-512-ckpt-5"
MAX_LEN = 512
BATCH_SIZE = 32
STRIDE = 256

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH).to("cuda")
print("Done loading model")

y_pred = []
with torch.no_grad():
    for i in tqdm(range(0, len(test), BATCH_SIZE)):
        inputs = tokenizer(
            X_test[i : i + BATCH_SIZE].tolist(),
            padding=True,
            truncation=True,
            max_length=MAX_LEN * 2,
            return_tensors="pt",
        ).to("cuda")
        # split inputs into batches of size MAX_LEN and STRIDE (for overlap)
        n_mini_batch = max(
            1, int(np.ceil((inputs["input_ids"].shape[1] - MAX_LEN) / STRIDE)) + 1
        )
        mini_batches = []
        for j in range(n_mini_batch):
            mini_batches.append(
                {k: v[:, j * STRIDE : j * STRIDE + MAX_LEN] for k, v in inputs.items()}
            )
        # run inference on each mini batch
        logits = []
        for mini_batch in mini_batches:
            logits.append(model(**mini_batch).logits.detach().cpu().numpy())
        # average the non-empty logits in the mini batches
        final_logits = np.zeros((BATCH_SIZE, 2))
        for j in range(BATCH_SIZE):
            valid_mini_batches = [
                k
                for k in range(n_mini_batch)
                if mini_batches[k]["attention_mask"][j, 0] == 1
            ]
            final_logits[j] = np.mean(
                np.array([logits[k][j] for k in valid_mini_batches]), axis=0
            )
        # append predictions
        y_pred.extend(
            (
                np.exp(final_logits)
                / np.sum(np.exp(final_logits), axis=-1, keepdims=True)
            )[:, 1]
        )

precision, recall, _ = precision_recall_curve(y_test, y_pred)
print("Precision-Recall AUC: " + str(auc(recall, precision)))
print("ROC AUC: " + str(roc_auc_score(y_test, y_pred)))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()
