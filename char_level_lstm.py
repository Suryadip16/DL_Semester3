import string
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

paragraph = "Artificial intelligence (AI) has transformed the way humans interact with technology, offering capabilities" \
            "that were once the realm of science fiction. From personalized recommendations on streaming platforms to " \
            "advancements in healthcare diagnostics, AI has seamlessly integrated into everyday life. At its core, AI " \
            "mimics human intelligence by learning from data, recognizing patterns, and making decisions. Deep learning," \
            " a subset of AI, has fueled breakthroughs in fields like natural language processing and image recognition," \
            " enabling applications such as chatbots and autonomous vehicles. However, alongside these advancements, AI " \
            "also raises ethical questions about bias, transparency, and its impact on the workforce. Striking a balance" \
            " between innovation and responsibility is key to ensuring AI's potential is harnessed for the greater good."

# adding start and end tokens
paragraph = "START" + paragraph + "END"

# convert entire para to idx_array
all_characters = string.ascii_letters + string.punctuation + " "
print(all_characters)
vocab = {"START": 0,
         "END": 1}
for char in all_characters:
    vocab[char] = all_characters.find(char) + 2


def tokenize(sentence):
    i = 0
    tokens = []
    while i < len(sentence):  # Ensure we don't exceed the length of the sentence
        if i + 5 <= len(sentence) and sentence[i:i + 5] == "START":
            tokens.append("START")
            i += 5  # Skip the length of "START"
        elif i + 3 <= len(sentence) and sentence[i:i + 3] == "END":
            tokens.append("END")
            i += 3  # Skip the length of "END"
        else:
            tokens.append(sentence[i])  # Add individual character
            i += 1
    return tokens


print(tokenize("Art"))


def text2indexarray(text):
    tokens = tokenize(text)
    idx_list = [vocab[token] for token in tokens]  # Simplified list comprehension
    idx_array = np.array(idx_list)  # Ensure integer type
    return idx_array


class ParagraphDataset(Dataset):
    def __init__(self, text, sequence_length=100):
        self.text = text
        self.sequence_length = sequence_length
        self.encoded_text = text2indexarray(self.text)

    def __len__(self):
        return len(self.encoded_text) - self.sequence_length

    def __getitem__(self, idx):
        x = self.encoded_text[idx: idx + self.sequence_length]
        y = self.encoded_text[idx + 1: idx + self.sequence_length + 1]
        tensor_x = torch.tensor(x)
        tensor_y = torch.tensor(y)

        return tensor_x, tensor_y


class myLSTM(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_layers):
        super(myLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden=None):
        embedded_x = self.embedding(x)
        out, h = self.lstm(embedded_x, hidden)
        out = self.fc(out)

        return out, h


# initialize dataset:
dataset = ParagraphDataset(paragraph)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# initialize model
vocab_size = len(vocab)
embedding_size = 64
hidden_size = 256
num_layers = 3

model = myLSTM(vocab_size=vocab_size, input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers)

# initialise loss and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = Adam(params=model.parameters(), lr=0.001)

# training loop:
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        batch_size = x.size(0)
        # Initialize hidden state based on the current batch size
        hidden_state = (
            torch.zeros(num_layers, batch_size, hidden_size),
            torch.zeros(num_layers, batch_size, hidden_size)
        )
        # fwd pass
        output, hidden = model(x, hidden_state)

        # detach hidden state to avoid backprop through entire sequence
        hidden = tuple([h.detach() for h in hidden])

        # reshape outputs and targets for CrossEntropyLoss
        output = output.view(-1, vocab_size)
        y = y.view(-1)

        # calculate loss
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch{epoch + 1}/{num_epochs}: ")
    print(f"Loss = {total_loss}")
#
# torch.save(model.state_dict(), "char_lstm_model.pth")
model.load_state_dict(torch.load("char_lstm_model.pth"))


def generate_text(start_word, length, rnn=model):
    rnn.eval()
    text = text2indexarray(start_word)
    hidden = None
    for _ in range(length):
        x = torch.tensor(text[-1]).unsqueeze(0)
        output, hidden = model(x, hidden)
        last_char = output.argmax(dim=1).item()
        text = np.append(text, last_char)
    idx_to_char = {v: k for k, v in vocab.items()}
    return "".join(idx_to_char[idx] for idx in text)


print(generate_text(start_word="flower", length=200, ))
