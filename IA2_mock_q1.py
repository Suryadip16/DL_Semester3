import numpy as np
import pandas as pd
import random
import string
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam

# Part A:

# # Generate Random Reviews
#
# # define positive and negative words and phrases
#
# positive_words = [
#     "amazing", "incredible", "fantastic", "wonderful", "brilliant", "engaging",
#     "captivating", "entertaining", "moving", "touching", "inspiring", "masterpiece"
# ]
# negative_words = [
#     "boring", "dull", "predictable", "tedious", "disappointing", "unimpressive",
#     "forgettable", "painful", "horrible", "uninspired", "lame", "lackluster"
# ]
# positive_phrases = [
#     "left me speechless", "is a must-watch", "surpassed my expectations",
#     "had me hooked", "is an unforgettable experience", "is a cinematic gem",
#     "is truly remarkable", "is a great watch", "had an amazing storyline",
#     "has outstanding performances", "brilliantly directed"
# ]
# negative_phrases = [
#     "was a waste of time", "left me disappointed", "fell flat",
#     "was a total letdown", "couldn't keep me engaged", "lacks depth",
#     "was poorly executed", "missed the mark", "has unconvincing acting",
#     "was unbearable to sit through", "was an underwhelming experience"
# ]
#
#
# def generate_reviews(sentiment):
#     review = "The movie"
#     if sentiment == "positive":
#         review += f" was {random.choice(positive_words)} and {random.choice(positive_phrases)}."
#     else:
#         review += f" was {random.choice(negative_words)} and {random.choice(negative_phrases)}."
#
#     return review
#
#
# movie_reviews = []
# for _ in range(500):
#     movie_reviews.append({"Review": generate_reviews(sentiment="positive"), "Label": "positive"})
#     movie_reviews.append({"Review": generate_reviews(sentiment="negative"), "Label": "negative"})
#
# print(movie_reviews[1])
# random.shuffle(movie_reviews)
# reviews_df = pd.DataFrame(movie_reviews)
# reviews_df.to_csv("movie_reviews_data.csv")

# Part B:

# Add start and end tokens to all reviews:
df = pd.read_csv("movie_reviews_data.csv")
for i in range(len(df)):
    df.iloc[i, 1] = "START" + df.iloc[i, 1] + "END"
print(df)
df['Label'] = np.where(df['Label'] == 'positive', 1, 0)
max_len = df["Review"].str.len().max() - 6  # because we consider START as 1 and END as 1. So 5 + 3 - 2 = 6

# NLP operations
all_characters = string.ascii_letters + string.punctuation + " "
print(all_characters)
vocab = {"START": 0,
         "END": 1}
for char in all_characters:
    vocab[char] = all_characters.find(char) + 2


# print(vocab)

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


print(tokenize("STARTThe movie was uninspired and lacks depth.END"))


#
#
# def token2onehot(token):
#     if token == "START" or token == "END":
#         index = vocab[token]
#         letter_array = np.zeros(shape=(len(vocab), 1))
#         letter_array[index, 0] = 1
#         return letter_array
#     else:
#         letter_list = []
#         for letter in token:
#             index = vocab[letter]
#             letter_array = np.zeros(shape=(len(vocab), 1))
#             letter_array[index, 0] = 1
#             letter_list.append(letter_array)
#         token_vec = np.array(letter_list).reshape(len(token), len(vocab))
#         token_vec = token_vec.T
#         return token_vec
#
#
# def text2onehot(text):
#     tokens = tokenize(text)
#     vec_list = [token2onehot(token) for token in tokens]
#     # Concatenate along the first axis to combine all token vectors
#     text_vec = np.concatenate(vec_list, axis=1)
#     return text_vec
#
#
# print(text2onehot("STARTThe movie was uninspired and lacks depth.END"))


# review_vecs = df["Review"].apply(func=text2onehot).values
# print(review_vecs[4, ])


# def padding_n_onehot(text):
#     review_vec = text2onehot(text)
#     if len(review_vec[1]) < max_len:
#         padding_vec = np.zeros(shape=(len(vocab), (max_len - len(review_vec[1]))))
#         padded_vec = np.concatenate((review_vec, padding_vec), axis=1)
#     else:
#         padded_vec = review_vec
#     return padded_vec
#
#
# print(padding_n_onehot("STARTThe movie was uninspired and lacks depth.END"))

def text2indexarray(text):
    tokens = tokenize(text)
    idx_list = [vocab[token] for token in tokens]  # Simplified list comprehension
    idx_array = np.array(idx_list)  # Ensure integer type

    if len(idx_array) < max_len:
        padding_array = np.zeros(shape=(max_len - len(idx_array),))
        padded_array = np.concatenate((idx_array, padding_array))
    else:
        padded_array = idx_array[:max_len]  # Truncate to max_len if necessary

    return padded_array


print(text2indexarray("STARTThe movie was uninspired and lacks depth.END"))


class ReviewDataset(Dataset):
    def __init__(self, df):
        self.reviews = df["Review"].apply(func=text2indexarray).values
        self.sentiment = torch.tensor(df["Label"].values, dtype=torch.int64)

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review_array = self.reviews[idx]
        review_tensor = torch.tensor(review_array, dtype=torch.long)
        sentiment = self.sentiment[idx]

        return review_tensor, sentiment


dataset = ReviewDataset(df)
train_size = int(0.8 * len(dataset))
test_size = int(len(dataset) - train_size)

train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)


# for batch, (review, sentiment) in enumerate(dataloader):
#     sample_num = batch + 1
#     review_tensor = review
#     sentiment_tensor = sentiment
#     print(f"Sample {sample_num}")
#     print(f"Shape of Review tensor: {review_tensor.shape}")
#     print(f"Sentiment: {sentiment.item()}")
#     print(review_tensor)
#     print("-" * 40)
#
# print("Done")

class RNN(nn.Module):
    def __init__(self, vocab_size, input_size, output_size, hidden_size, droupout, num_layers):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, input_size)

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=droupout)

        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

        self.dropout = nn.Dropout(p=droupout)

    def forward(self, x):
        x_embedded = self.embedding(x)
        out, _ = self.rnn(
            x_embedded)  # _ is a placeholder for hidden state. Since we are using only 1 layer of RNN we don't need it

        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)

        return out


vocab_size = len(vocab)
input_size = vocab_size
num_classes = 2
hidden_size = 50
dropout = 0.3
num_layers = 1

model = RNN(vocab_size=vocab_size, input_size=input_size,
            output_size=num_classes, num_layers=num_layers,
            droupout=dropout, hidden_size=hidden_size)

optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0

    for batch, (reviews, labels) in enumerate(train_loader):
        # reviews.shape: (batch_size, seq_len) -> (batch_size, max_len)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(reviews)  # outputs.shape: (batch_size, output_size)

        # Compute loss
        loss = criterion(outputs, labels)  # labels.shape: (batch_size)
        total_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {total_loss / len(train_loader)}")

# Testing
model.eval()
num_batches = len(test_loader)
num_dp = len(test_loader.dataset)
test_loss, correct_preds = 0.0, 0.0

with torch.no_grad():
    for reviews, labels in test_loader:
        pred = model(reviews)
        test_loss += criterion(pred, labels).item()
        correct_preds += (pred.argmax(1) == labels).type(torch.float).sum().item()
    avg_test_loss = test_loss / num_batches
    model_acc = correct_preds / num_dp
    print(f"Average Test Loss: {avg_test_loss}")
    print(f"Model Accuracy: {model_acc * 100}%")
