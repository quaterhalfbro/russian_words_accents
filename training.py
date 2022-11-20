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
from models import DenseNet


BATCH_SIZE = 512
EPOCHS = 1500
LR = 0.001
LOG_INTERVAL = 10


class CustomDataset(Dataset):
    def __init__(self, words, labels):
        self.words = list(words)
        self.labels = list(labels)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        word = [vocab[i] if i in vocab else vocab["<unk>"] for i in split(self.words[item])]
        if len(word) < max_length:
            word += [vocab["<pad>"]] * (max_length - len(word))
        return {"word": torch.tensor(word, dtype=torch.long), "label": torch.tensor([self.labels[item] - 1])}


def tokens_iterator(words):
    for i in words:
        yield [j if wc[j] > 1 or j in test_wc.keys() else "<unk>" for j in split(i)]


df = pd.read_csv("data/train.csv")
df2 = pd.read_csv("data/test.csv")
wc, test_wc = {}, {}
for j in df["word"]:
    for i in split(j):
        if i in wc.keys():
            wc[i] += 1
        else:
            wc[i] = 1
for j in df2["word"]:
    for i in split(j):
        if i in test_wc.keys():
            test_wc[i] += 1
        else:
            test_wc[i] = 1

vocab = build_vocab_from_iterator(tokens_iterator(df["word"]), specials=["<unk>", "<pad>"])
pickle.dump(vocab, open(f"checkpoints/vocab{len(vocab)}.pth", "wb"))

x_train, x_test, y_train, y_test = train_test_split(df["word"], df["stress"], test_size=0.001, shuffle=True,
                                                    random_state=42)

train_dataloader = DataLoader(CustomDataset(x_train, y_train), batch_size=BATCH_SIZE)
test_dataloader = DataLoader(CustomDataset(x_test, y_test), batch_size=BATCH_SIZE)

model = DenseNet()
max_length = model.get_parameter("MAX_LENGTH")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
metrics = {"test_loss": [], "train_loss": [], "test_acc": [], "train_acc": [], "test_precision": [], "test_recall": []}
for epoch in range(1, EPOCHS + 1):
    model.train()
    mean_loss, mean_acc = 0, 0
    data_count = 0
    for i, data in enumerate(train_dataloader):
        y_pred = model(data["word"])
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
    torch.save(model.state_dict(), f"weights/DenseNet{epoch}.pt")
    model.eval()
    loss, acc, precision, recall = 0, 0, 0, 0
    data_count = 0
    for i, data in enumerate(test_dataloader):
        y_pred = model(data["word"])
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
    with open("checkpoints/dense_net_metrics.json", "w") as f:
        json.dump(metrics, f)
