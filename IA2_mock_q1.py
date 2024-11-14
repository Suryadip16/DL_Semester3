import numpy as np
import pandas as pd
import random
import string
import torch
import torch.nn as nn

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
    while i <= len(sentence):
        if sentence[i:i + 5] == "START":
            tokens.append("START")
            i += 5

        elif sentence[i:i + 4] == "END":
            tokens.append("END")
            i += 4
        else:
            tokens.append(sentence[i])
            i += 1
    return tokens


def token2onehot(token):
    if token == "START" or token == "END":
        index = vocab[token]
        letter_array = np.zeros(shape=(len(vocab), 1))
        letter_array[index, 0] = 1
        return letter_array
    else:
        letter_list = []
        for letter in token:
            index = vocab[letter]
            letter_array = np.zeros(shape=(len(vocab), 1))
            letter_array[index, 0] = 1
            letter_list.append(letter_array)
        token_vec = np.array(letter_list).reshape(len(token), len(vocab))
        token_vec = token_vec.T
        return token_vec


def text2onehot(text):
    tokens = tokenize(text)
    vec_list = [token2onehot(token) for token in tokens]
    # Concatenate along the first axis to combine all token vectors
    text_vec = np.concatenate(vec_list, axis=1)
    return text_vec


print(text2onehot("STARTThe movie was uninspired and lacks depth.END"))
# review_vecs = df["Review"].apply(func=text2onehot).values
# print(review_vecs[4, ])
