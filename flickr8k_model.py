import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        # we set train_CNN to false at the beginning because we are not going to train the CNN. We will use a pretrained model

        self.inception = models.inception_v3(pretrained=True,
                                             aux_logits=False)  # aux_logits=False disables the auxiliary outputs that Inception v3 can generate during training.
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)

        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN
        # This customization ensures the model reuses the pretrained features (trained on ImageNet) from Inception v3, fine-tuning only the
        # last layer (the custom fc layer) to output embeddings of the desired size. Use this strategy whenever you want to use a CNN as a feature extractor.

        return self.dropout(self.relu(features))


class Decoder_RNN(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, num_layers):
        super(Decoder_RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)  # each token is one hot encoded. so input size is vocab size. uses nn.Embedding of pytorch to form the word embedding of embed_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)  # takes in the embedded tokens, hidden_size=size of hidden state h, num_layers=number og LSTM layers.
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, caption):
        embeddings = self.dropout(self.embedding(caption))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CNN_to_RNN(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, num_layers):
        super(CNN_to_RNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = Decoder_RNN(embed_size, vocab_size, hidden_size, num_layers)

    def forward(self, images, caption):
        x = self.encoderCNN(images)
        x = self.decoderRNN(x, caption)
        return x

    def captionImage(self):
