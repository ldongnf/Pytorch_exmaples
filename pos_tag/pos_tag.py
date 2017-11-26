#!/usr/bin/env python

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle
import numpy as np
import matplotlib.pyplot as plt

import argparse

def word2idx(train_sents):
    word_idx_mapping = {}
    for sent in train_sents:
        for word in sent:
            if word not in word_idx_mapping:
                word_idx_mapping[word] = len(word_idx_mapping)
    return word_idx_mapping

def tag2idx(truth_tags):
    tag_idx_mapping = {}
    for tags in truth_tags:
        for tag in tags:
            if tag not in tag_idx_mapping:
                tag_idx_mapping[tag] = len(tag_idx_mapping)
    return tag_idx_mapping

def idx2tag(tag_idxs, idx_tag_mapping):
    # view make sure it's one dim
    # return a seq of tag given by the input
    return [idx_tag_mapping[idx] for idx in tag_idxs.data.tolist()]

def score2tags(tag_scores, idx_tag_mapping):
    _, idxs =  torch.max(tag_scores, 1)
    return idx2tag(idxs, idx_tag_mapping)

def convert2idx(seq, idx_mapping):
    idxs = [idx_mapping[word] for word in seq]
    return Variable(torch.LongTensor(idxs))

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_tags):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # init the hidden states       
        self.hidden = self.init_hidden()

        # LSTM, from num of embed_dim to hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # transfer from hidden to output layer
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)

        # word_embedding
        self.word2vec = nn.Embedding(vocab_size, embedding_dim)

    def init_hidden(self):
        # the semantics: num_layer, batch_size, hidden_dim

        return Variable(torch.zeros(1, 1, self.hidden_dim)), Variable(torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentences):
        # convert to word vec
        #print sentences
        feats = self.word2vec(sentences).view(len(sentences), 1, -1)

        # use the output of lstm
        lstm_output, self.hidden = self.lstm(feats, self.hidden)
        
        # convert hidden layer to tag_space
        tag_space = self.hidden2tag(lstm_output.view(len(sentences), -1))
        
        # get the score of tags
        score = F.log_softmax(tag_space)
        
        return score

def draw_plot(losses):
    plt.figure()
    plt.plot(losses)
    

def train(train_sents, train_truths, word_idx_mapping, tag_idx_mapping, idx_tag_mapping, n_epochs=30, if_draw_plot=False):
    embedding_dim = 16
    hidden_dim = 128
    vocab_size = len(word_idx_mapping)
    num_tags = len(tag_idx_mapping)
    model = LSTMTagger(embedding_dim, hidden_dim, vocab_size, num_tags)
    
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5)

    print "Unit test(Before training): "
    print train_sents[0]
    temp = model(convert2idx(train_sents[0], word_idx_mapping))
    print temp
    print score2tags(temp, idx_tag_mapping)

    all_losses = []
    for epoch in xrange(n_epochs):
        losses = []
        for sent, tags in zip(train_sents, train_truths):
            # clear grad
            model.zero_grad()

            # clear hidden
            model.hidden = model.init_hidden()

            feats = convert2idx(sent, word_idx_mapping)
            truths = convert2idx(tags, tag_idx_mapping)

            score = model(feats)
            loss = loss_function(score, truths)
            loss.backward()
            optimizer.step()

            losses.append(loss.data.mean())
        all_losses.append(np.mean(losses))
        print '[{:d}/{:d}] Loss: {:.3f}'.format(epoch + 1, n_epochs, np.mean(losses))
    
    if if_draw_plot:
        draw_plot(all_losses)
    return model

def test(model, test_sents, test_truths, word_idx_mapping, idx_tag_mapping):
    sent_cnt = 0
    N = len(test_sents)
    tag_cnt = 0
    total_tag = 0

    for i, sent in enumerate(test_sents):
        test = convert2idx(sent, word_idx_mapping)
        score = model(test)
        pred = score2tags(score, idx_tag_mapping)
        str_truth = " ".join(test_truths[i])
        str_pred = " ".join(pred)

        sent_cnt += str_truth == str_pred
        
        for x, y in zip(pred, test_truths[i]):
            tag_cnt += (x == y)
            total_tag += 1

    print "sent level: {}, correct: {}, acc: {}".format(N, sent_cnt, sent_cnt * 1.0 / N)
    print "tag level: {}, correct: {}, acc: {}".format(total_tag, tag_cnt, tag_cnt * 1.0 / total_tag)

def prepare_args():
    parser = argparse.ArgumentParser(description="Multi classification")
    parser.add_argument('-t', '--n_train', default=500, type=int,
            help='Number of training samples')
    parser.add_argument('-e', '--n_test', default=500, type=int,
            help='Number of testing samples')
    parser.add_argument('-n', '--n_epochs', default=30, type=int,
            help='Number of epochs')
    parser.add_argument('-p', '--show_plot', default=False, type=bool,
            help='Number of epochs')
    opts = parser.parse_args()

    return opts

def main():
    opts = prepare_args()

    n_train = opts.n_train
    n_test = opts.n_test

    N = n_train + n_test
    n_epochs = opts.n_epochs
    if_draw_plot = opts.show_plot

    path = "training.p"
    data = pickle.load(open(path, "rb"))[:N]
    sents, truths = zip(*data)
    
    train_sents = sents[: n_train]
    train_truths = truths[: n_train]

    test_sents = sents[-n_test: ]
    test_truths = truths[-n_test: ]

    word_idx_mapping = word2idx(sents)
    tag_idx_mapping = tag2idx(truths)
    idx_tag_mapping = {v: k for k, v in tag_idx_mapping.iteritems()}

    model = train(train_sents, train_truths, word_idx_mapping, tag_idx_mapping, idx_tag_mapping, n_epochs, if_draw_plot)  
    test(model, train_sents, train_truths, word_idx_mapping, idx_tag_mapping)
    test(model, test_sents, test_truths, word_idx_mapping, idx_tag_mapping)
    
    if if_draw_plot:
        plt.show()
if __name__ == '__main__':
    main()
