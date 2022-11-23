import torch
from torch import nn
import pandas as pd
import json
import pickle
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

from syllables_splitting import split
from models import LemmaUsingNetWithPositionalEmbeddings
from build_vocab import build_vocab

BATCH_SIZE = 256
EPOCHS = 1500
LR = 0.001
LOG_INTERVAL = 10


class CustomDataset(Dataset):
    def __init__(self, words, labels):
        self.words = list(words["word"])
        self.lemma = list(words["lemma"])
        self.labels = list(labels)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        word = [vocab[i] if i in vocab else vocab["<unk>"] for i in split(self.words[item])]
        lemma = [lemma_vocab[i] if i in lemma_vocab else lemma_vocab["<unk>"] for i in split(self.lemma[item])]
        if len(word) < max_length:
            word += [vocab["<pad>"]] * (max_length - len(word))
        if len(lemma) < max_lemma_length:
            lemma += [lemma_vocab["<pad>"]] * (max_lemma_length - len(lemma))
        return {"word": torch.tensor(word, dtype=torch.long),
                "lemma": torch.tensor(lemma, dtype=torch.long),
                "label": torch.tensor([self.labels[item] - 1])}


df = pd.read_csv("data/cleaned_dataset.csv")
df2 = pd.read_csv("data/test.csv")

vocab = build_vocab(list(df["word"]), list(df2["word"]))
lemma_vocab = build_vocab(list(df["lemma"]), list(df2["lemma"]), tag="lemma")
x_train, x_test, y_train, y_test = train_test_split(df[["word", "lemma"]], df["stress"], test_size=0.1, shuffle=True,
                                                    random_state=42)

train_dataloader = DataLoader(CustomDataset(x_train, y_train), batch_size=BATCH_SIZE)
test_dataloader = DataLoader(CustomDataset(x_test, y_test), batch_size=BATCH_SIZE)

model = LemmaUsingNetWithPositionalEmbeddings()
max_length = model.get_parameter("MAX_LENGTH")
max_lemma_length = model.get_parameter("MAX_LEMMA_LENGTH")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
metrics = {"test_loss": [], "train_loss": [], "test_acc": [], "train_acc": [], "test_precision": [], "test_recall": []}
for epoch in range(1, EPOCHS + 1):
    model.train()
    mean_loss, mean_acc = 0, 0
    data_count = 0
    for i, data in enumerate(train_dataloader):
        y_pred = model(data["word"], data["lemma"])
        y_test = data["label"].reshape(len(data["label"]))
        loss = criterion(y_pred, y_test)
        mean_loss += loss.tolist() * len(y_test)
        mean_acc += accuracy_score(y_test.tolist(), y_pred.argmax(dim=1).tolist()) * len(y_test)
        data_count += len(y_test)
        if i % LOG_INTERVAL == LOG_INTERVAL - 1:
            print(f"{i}/{len(x_train) // BATCH_SIZE} loss = {loss}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    metrics["train_loss"].append(mean_loss / data_count)
    metrics["train_acc"].append(mean_acc / data_count)
    torch.save(model.state_dict(), f"weights/bert_net{epoch}.pt")
    model.eval()
    loss, acc, precision, recall = 0, 0, 0, 0
    data_count = 0
    for i, data in enumerate(test_dataloader):
        y_pred = model(data["word"], data["lemma"])
        y_test = data["label"].reshape(len(data["label"]))
        loss += criterion(y_pred, y_test).tolist() * len(y_pred)
        y_pred = y_pred.argmax(dim=1).tolist()
        y_test = y_test.tolist()
        acc += accuracy_score(y_test, y_pred) * len(y_pred)
        precision += precision_score(y_test, y_pred, average="macro", zero_division=True) * len(y_pred)
        recall += recall_score(y_test, y_pred, average="macro", zero_division=True) * len(y_pred)
        data_count += len(y_pred)
    metrics["test_acc"].append(acc / data_count)
    metrics["test_recall"].append(recall / data_count)
    metrics["test_precision"].append(precision / data_count)
    metrics["test_loss"].append(loss / data_count)
    print(f"After epoch {epoch}/{EPOCHS}")
    print(f"Acc: {metrics['test_acc'][-1]}, "
          f"Precision: {metrics['test_precision'][-1]}, "
          f"Recall: {metrics['test_recall'][-1]}")
    with open("checkpoints/bert_metrics.json", "w") as f:
        json.dump(metrics, f)
