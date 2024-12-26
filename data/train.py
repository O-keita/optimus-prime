import json
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
from preprocess import tokenize, bag_of_words, stem
from models.model import NeuralNet


# Add the parent directory to sys.path
nltk.download('punkt')


with open('intents.json') as f:

    intents = json.load(f)



all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


all_words = sorted(set(stem(w) for w in all_words if w not in ['?', '!', '.', ','] ))
tags = sorted(set(tags))


X_train = []
y_train = []


for (pattern_sentence, tag) in xy:

    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    labels = tags.index(tag)
    y_train.append(labels)

X_train = np.array(X_train)
y_train = np.array(y_train)




class ChatDataset(Dataset):

    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y

    def __getitem__(self, index):
        return  self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples


batch_size = 16


dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3)

input_size = len(X_train[0])
hidden_size = 12
output_size = len(tags)
lr = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(1000):
    for (words, labels) in train_loader:

        words = words
        labels = labels.to(dtype=torch.long)

        ouput = model(words)
        loss = criterion(ouput, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'{epoch+1}/{1000}, Loss: {loss.item():.4f}')


data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE ="optimus.pth"
torch.save(data, FILE)


print(f'Training completed and file saved to {FILE}')

#export PYTHONPATH=/home/omar/coding/work/optimus-prime


