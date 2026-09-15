# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter
import matplotlib.pyplot as plt


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence, add_to_indexer, lowercase=True):
        feature = Counter()
        for word in sentence:
            if lowercase: word = word.lower()

            # get the index
            index = self.indexer.add_and_get_index(word) if add_to_indexer else self.indexer.index_of(word)
            
            # if the index is -1, skip; otherwise feats[index] += 1
            if index != -1: feature[index] += 1

        return feature


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence, add_to_indexer, lowercase=True):
        feature = Counter()
        for i in range(len(sentence) - 1):
            word_i, word_j = sentence[i], sentence[i+1]
            if lowercase:
                word_i = word_i.lower()
                word_j = word_j.lower()
            bigram = word_i + '|' + word_j

            # get the index
            index = self.indexer.add_and_get_index(bigram) if add_to_indexer else self.indexer.index_of(bigram)
            
            # if the index is -1, skip; otherwise feats[index] += 1
            if index != -1: feature[index] += 1

        return feature

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feature_extractor):
        self.W = weights
        self.feature_extractor = feature_extractor

    def predict(self, sentence):
        features = self.feature_extractor.extract_features(sentence=sentence, add_to_indexer=False)
        idxs, values = list(features.keys()), list(features.values())
        selected_weights = self.W[idxs]

        z = np.dot(np.array(selected_weights), np.array(values))
        pred = 1.0/(1.0 + np.exp(-1 * z))

        return 1 if pred > 0.5 else 0

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # build the indexer (vocabulary)
    for sample in train_exs:
        _ = feat_extractor.extract_features(sample.words, add_to_indexer=True)
    vocab_size = len(feat_extractor.get_indexer())
    weights = np.zeros(vocab_size)

    lr_start = 0.005
    epochs = 10

    dev_exs = read_sentiment_examples("data/dev.txt")
    dev_accuracies = []

    for epoch in range(epochs):
        random.shuffle(train_exs)
        for sample in train_exs:
            # extract features
            words, label = sample.words, sample.label
            features = feat_extractor.extract_features(sentence=words, add_to_indexer=False)

            # compute scores
            idxs, values = list(features.keys()), list(features.values())
            selected_weights = weights[idxs]
            z = np.dot(np.array(selected_weights), np.array(values))
            pred = 1 / (1 + np.exp(-1 * z))
            error = pred - label

            # weight updated
            lr = lr_start / (1 + 0.01 * epoch)
            for idx, val in features.items():
                weights[idx] -= lr * val * error

        tempCLf = LogisticRegressionClassifier(weights=weights, feature_extractor=feat_extractor)
        accuracy = 0
        for sample in dev_exs:
            words, label = sample.words, sample.label
            pred = tempCLf.predict(words)
            if pred == label: accuracy += 1
        dev_accuracies.append(accuracy / len(dev_exs))
        print(f"Epoch: [{epoch+1}/{epochs}], Accuracy: {np.round(accuracy / len(dev_exs), 5)}")

    # plt.plot(dev_accuracies)
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Dev accuracy v/s Epoch')
    # plt.show()

    return LogisticRegressionClassifier(weights=weights, feature_extractor=feat_extractor)
        

def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model

class DAN(nn.Module):
    def __init__(self, word_embeddings, hidden_layer, num_classes=1):
        super(DAN, self).__init__()
        self.embeddings = word_embeddings.get_initialized_embedding_layer()
        self.embed_dim = self.embeddings.embedding_dim
        self.hidden = hidden_layer
        self.classes = num_classes

        self.linear = nn.Linear(self.embed_dim, self.hidden, bias=True)
        self.output = nn.Linear(self.hidden, self.classes, bias=True)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, word_indices):
        if type(word_indices) == list: word_indices = torch.tensor(word_indices, dtype=torch.long)
        embeds = self.embeddings(word_indices)
        embeds_avg = embeds.mean(axis=0)
        x = nn.functional.relu(self.linear(embeds_avg))
        x = self.dropout(x)
        x = torch.sigmoid(self.output(x))
        return x

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        self.embeddings = word_embeddings
        self.network = network

    def predict(self, sentence):
        word_indices = []
        for word in sentence:
            idx = self.embeddings.word_indexer.index_of(word)
            if idx == -1: idx = self.embeddings.word_indexer.index_of("UNK")
            word_indices.append(idx)
        with torch.no_grad():
            prob = self.network(word_indices)
        return 1 if prob.item() > 0.5 else 0

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    np.random.seed(42)
    random.seed(42)

    epochs = args.num_epochs
    lr = args.lr
    hidden_size = args.hidden_size

    model = DAN(word_embeddings=word_embeddings, hidden_layer=hidden_size, num_classes=1)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dev_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        random.shuffle(train_exs)

        for sample in train_exs:
            sentence, label = sample.words, sample.label
            word_indices = []
            for word in sentence:
                idx = word_embeddings.word_indexer.index_of(word)
                if idx == -1: idx = word_embeddings.word_indexer.index_of("UNK")
                word_indices.append(idx)

            # Handle empty sentences gracefully if any exist
            if len(word_indices) == 0: continue
                
            # Clear previous gradients
            model.zero_grad()
            
            # Forward pass to get probability prediction
            prob = model(word_indices)
            
            # Create target label tensor matching the shape of prob (e.g., shape)
            target = torch.tensor([float(label)], dtype=torch.float)
            
            # Compute loss, run backward pass, and update parameters
            loss = loss_function(prob, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        model.eval()
        tempCLf = NeuralSentimentClassifier(network=model, word_embeddings=word_embeddings)
        accuracy = 0
        with torch.no_grad():
            for sample in dev_exs:
                words, label = sample.words, sample.label
                pred = tempCLf.predict(words)
                if pred == label: accuracy += 1
            dev_accuracies.append(accuracy / len(dev_exs))
        print(f"Epoch: [{epoch+1}/{epochs}], Accuracy: {np.round(accuracy / len(dev_exs), 5)}, - Loss: {epoch_loss / len(train_exs):.4f}")

    model.eval()
    return NeuralSentimentClassifier(network=model, word_embeddings=word_embeddings)
        
