# models.py
import numpy as np
import random as rd
#import matplotlib.pyplot  as plt
from sentiment_data import *
from utils import *
from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
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
    
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")
    """
    def __init__(self, Indexer: Indexer):
        self.indexer = Indexer
    def extract_features(self, ex: SentimentExample, add_to_indexer: bool=False):
        if type(ex) == list:
            idx = [self.indexer.index_of(w) for w in ex]
        else:
            idx = [self.indexer.index_of(w) for w in ex.words]
        feat = np.zeros(self.vocab_size())
        for i in idx:
            feat[i] += 1
        return feat
    def vocab_size(self):
        return len(self.indexer)


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")
    """
    def __init__(self, Indexer: Indexer):
        self.indexer = Indexer
    def extract_features(self, ex: SentimentExample, add_to_indexer: bool=False):
        if type(ex) == list:
            arr = [w for w in ex]
            idx = []
            for i in range(len(ex) - 1):
                s = arr[i] + " " + arr[i + 1]
                idx.append(self.indexer.index_of(s))
        else:
            arr = [w for w in ex.words]
            idx = []
            for i in range(len(ex.words) - 1):
                s = arr[i] + " " + arr[i + 1]
                idx.append(self.indexer.index_of(s))
        feat = np.zeros(self.vocab_size())
        for i in idx:
            feat[i] += 1
        return feat
    def vocab_size(self):
        return len(self.indexer)


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")
    """
    def __init__(self, Indexer: Indexer):
        self.indexer = Indexer
    def extract_features(self, ex: SentimentExample, add_to_indexer: bool=False):
        if type(ex) == list:
            idx = [self.indexer.index_of(w.lower()) for w in ex]
        else:
            idx = [self.indexer.index_of(w.lower()) for w in ex.words]
        feat = np.zeros(self.vocab_size())
        for i in idx:
            feat[i] += 1
        return feat
    def vocab_size(self):
        return len(self.indexer)


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    
    def __init__(self):
        raise Exception("Must be implemented")
    """
    def __init__(self, train_exs, feat_extractor):
        self.feat_extractor = feat_extractor
        n = len(train_exs)
        weight_vector = np.zeros([self.feat_extractor.vocab_size()])
        Epoch = 10
        learning_rate = 1
        for i in range(Epoch):
            #acc = np.zeros([n])
            rd.Random(1).shuffle(train_exs)
            for j in range(n):
                feat = self.feat_extractor.extract_features(train_exs[j])
                pred = np.dot(weight_vector,feat) > 0
                if pred == train_exs[j].label:
                    #acc[j] = 1
                    continue
                else:
                    if train_exs[j].label == 1:
                        weight_vector = weight_vector + feat * learning_rate
                    else:
                        weight_vector = weight_vector - feat * learning_rate
            learning_rate /= np.sqrt(i + 1)
            #print('epoch: %s, acc: %.6f' % (i, np.mean(acc)))
        self.weight_vector = weight_vector

    def predict(self, ex: SentimentExample):
        feat = self.feat_extractor.extract_features(ex)
        if np.dot(self.weight_vector, feat) > 0:
            return 1
        else:
            return 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    
    def __init__(self):
        raise Exception("Must be implemented")
    """
    def __init__(self, train_exs, feat_extractor):
        self.feat_extractor = feat_extractor
        #traning the perceptron classifier
        n = len(train_exs)
        
        weight_vector = np.zeros([self.feat_extractor.vocab_size()])
        Epoch = 10
        learning_rate = 1
        for i in range(Epoch):
            acc = np.zeros([n])
            rd.Random(5).shuffle(train_exs)
            for j in range(n):
                y = self.feat_extractor.extract_features(train_exs[j])   
                pred = np.dot(weight_vector,y) > 0
                prob = np.exp(pred) / (1 + np.exp(pred))  
                if pred == train_exs[j].label:
                    acc[j] = 1
                    continue
                else:
                    if train_exs[j].label == 1:
                        weight_vector = weight_vector + learning_rate * y * (1 - prob)
                    else:
                        weight_vector = weight_vector - learning_rate * y * (1 - prob)           
            learning_rate /= np.sqrt(i + 1)
            print('epoch: %s, acc: %.6f' % (i, np.mean(acc)))
        self.weight_vector = weight_vector

    def predict(self, ex: SentimentExample):
        y = self.feat_extractor.extract_features(ex)
        return np.dot(self.weight_vector, y) > 0
        

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    
    raise Exception("Must be implemented")
    """
    model = PerceptronClassifier(train_exs, feat_extractor)
    return model


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    
    raise Exception("Must be implemented")
    """
    model = LogisticRegressionClassifier(train_exs, feat_extractor)
    return model


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
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
        n = len(train_exs)
        my_indexer = Indexer()
        for i in range(n):
            m = len(train_exs[i].words)
            for j in range(m):
                my_indexer.add_and_get_index(train_exs[i].words[j], add=True)
        feat_extractor = UnigramFeatureExtractor(my_indexer)
        print(feat_extractor)
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        n = len(train_exs)
        my_indexer = Indexer()
        for i in range(n):
            m = len(train_exs[i].words) - 1
            for j in range(m):
                s = train_exs[i].words[j] + " " + train_exs[i].words[j + 1]
                my_indexer.add_and_get_index(s, add=True)
        feat_extractor = BigramFeatureExtractor(my_indexer)
    elif args.feats == "BETTER":
        stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'] 
        n = len(train_exs)
        my_indexer = Indexer()
        for i in range(n):
            m = len(train_exs[i].words)
            for j in range(m):
                if train_exs[i].words[j] not in stop_words:
                    my_indexer.add_and_get_index(train_exs[i].words[j].lower(), add=True)
                    
        feat_extractor = BetterFeatureExtractor(my_indexer)
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model