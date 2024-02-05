import numpy as np
import pandas as pd
import pickle as pkl
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from nltk import tokenize
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer


MIN_CHUNK_LEN = 8
MAX_CHUNK_LEN = 64
EMBED_SEQ_LEN = 32
LSTM_HIDDEN_DIM = 128
NUM_EPOCHS = 10
BATCH_SIZE = 256
DATASET_NAME = "persuade_combined"
DATASET_PATH = f"../datasets/{DATASET_NAME}.csv"
EMBED_DATA_PATH = f"../datasets/{DATASET_NAME}_emb_{MAX_CHUNK_LEN}.pkl"
TEST_DATASET_NAME = "feedback"
TEST_DATASET_PATH = f"../datasets/{TEST_DATASET_NAME}.csv"
TEST_EMBED_DATA_PATH = f"../datasets/{TEST_DATASET_NAME}_emb_{MAX_CHUNK_LEN}.pkl"
# MODEL_PATH = f"../models/lstm_{LSTM_HIDDEN_DIM}.pt"
MODEL_PATH = f"../models/ffn_{EMBED_SEQ_LEN}.pt"


class LSTM(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, num_layers=1, dropout=0, bidirectional=False
    ):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_dim * (bidirectional + 1), 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out[:, -1, :])
        prob = torch.sigmoid(logits)
        return prob


class FFN(nn.Module):
    def __init__(self, embedding_dim, seq_len):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * seq_len, embedding_dim * seq_len // 2)
        self.fc2 = nn.Linear(embedding_dim * seq_len // 2, embedding_dim * seq_len // 8)
        self.fc3 = nn.Linear(embedding_dim * seq_len // 8, embedding_dim)
        self.fc4 = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        logits = self.fc4(x)
        prob = torch.sigmoid(logits)
        return prob


class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def embed_essay(essay, emb_model):
    # Split essay into sentences
    sentences = tokenize.sent_tokenize(essay)
    # Merge sentences until the chunk length is at least MIN_CHUNK_LEN
    i = 0
    while i < len(sentences) - 1:
        if len(sentences[i].split()) < MIN_CHUNK_LEN:
            sentences[i] = sentences[i] + " " + sentences[i + 1]
            sentences.pop(i + 1)
        else:
            i += 1
    # Split sentences into chunks of MAX_CHUNK_LEN words
    chunks = []
    for sent in sentences:
        words = sent.split()
        chunks.extend(
            [
                " ".join(words[i : i + MAX_CHUNK_LEN])
                for i in range(0, len(words), MAX_CHUNK_LEN)
            ]
        )
    # Embed the chunks
    return emb_model.encode(chunks)
    # return embs[:-1] - embs[1:]


def load_dataset(emb_model, dataset_path=DATASET_PATH, embed_data_path=EMBED_DATA_PATH):
    if os.path.exists(embed_data_path):
        X_emb, y = pkl.load(open(embed_data_path, "rb"))
    else:
        # Read dataset
        df = pd.read_csv(dataset_path)
        df = df.sample(frac=1).reset_index(drop=True)
        X_text = df["text"]
        if "label" in df.columns:
            y = df["label"]
        else:
            y = df["generated"]
        # Embed the essays
        X_emb = []
        for essay in tqdm(X_text):
            X_emb.append(embed_essay(essay, emb_model))
        pkl.dump((X_emb, y), open(embed_data_path, "wb"))
    return X_emb, np.array(y)


def pad_dataset(X_emb, y):
    # Pad and truncate each embedding sequence to EMBED_SEQ_LEN
    for i in tqdm(range(len(X_emb))):
        X_emb[i] = X_emb[i][:EMBED_SEQ_LEN]
        if len(X_emb[i]) < EMBED_SEQ_LEN:
            X_emb[i] = np.concatenate(
                (X_emb[i], np.zeros((EMBED_SEQ_LEN - len(X_emb[i]), emb_dim)))
            )
    X_emb = torch.FloatTensor(np.array(X_emb))
    y = torch.FloatTensor(y)
    return X_emb, y


def collate_pad(batch):
    # Sort the batch by sequence length (descending order)
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    # Separate the inputs and labels
    inputs, labels = zip(*batch)
    # Pad the inputs to the length of the longest sequence
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, torch.stack(labels)


def train(model, X_train, y_train, X_val, y_val):
    # Prepare dataloader
    train_dataset = EmbeddingDataset(X_train, torch.FloatTensor(y_train))
    val_dataset = EmbeddingDataset(X_val, torch.FloatTensor(y_val))
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    print("Done preparing dataloader")
    # Train model
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()
    for epoch in range(NUM_EPOCHS):
        model.train()
        for X, y in tqdm(train_loader):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(X).squeeze()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            y_preds = []
            for X, y in tqdm(val_loader):
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X).squeeze()
                y_preds.append(y_pred.detach().cpu().numpy())
            y_preds = np.concatenate(y_preds)
            print(f"Epoch {epoch + 1}: {roc_auc_score(y_val, y_preds)}")
    # Save model
    torch.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)
    emb_dim = emb_model.get_sentence_embedding_dimension()
    ffn_model = FFN(emb_dim, EMBED_SEQ_LEN).to(device)
    # lstm_model = LSTM(emb_dim, LSTM_HIDDEN_DIM, num_layers=2, dropout=0.1).to(device)

    # Load dataset
    X_emb, y = load_dataset(emb_model)
    X_emb, y = pad_dataset(X_emb, y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_emb, y, test_size=0.1, random_state=42
    )
    # Train model
    print("Training FFN model...")
    train(ffn_model, X_train, y_train, X_val, y_val)

    # Evaluate model
    X_test_emb, y_test = load_dataset(
        emb_model, dataset_path=TEST_DATASET_PATH, embed_data_path=TEST_EMBED_DATA_PATH
    )
    X_test_emb, y_test = pad_dataset(X_test_emb, y_test)
    test_dataset = EmbeddingDataset(X_test_emb, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    ffn_model.load_state_dict(torch.load(MODEL_PATH))
    ffn_model.eval()
    with torch.no_grad():
        y_preds = []
        for X, y in tqdm(test_loader):
            X = X.to(device)
            y = y.to(device)
            y_pred = ffn_model(X).squeeze()
            y_preds.append(y_pred.detach().cpu().numpy())
        y_preds = np.concatenate(y_preds)
        print(f"Test AUC: {roc_auc_score(y_test, y_preds)}")
