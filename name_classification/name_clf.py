#!/usr/bin/env python

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import glob
import numpy as np
import matplotlib.pyplot as plt

import random

import argparse

# build the name dictionary
def build_name_dict(path='./data/*.txt'):
    file_names = glob.glob(path)

    name_category = {}
    for fname in file_names:
        with open(fname, "r") as infile:
            category = fname.split('/')[-1].split('.')[0]
            names = [line.strip().lower() for line in infile]
            name_category[category] = names

    return name_category

def one_hot_encode(num_labels, data):
    # num_labels
    # data need to be encoded
    # rtype: encoded data

    # view = np.reshape: -1 means the dimension you not sure 
    idxs = torch.LongTensor(data).view(-1, 1)
    tensor = torch.LongTensor(len(idxs), num_labels).zero_()
    # scatter the 1 to index positions
    tensor.scatter_(1, idxs, 1)
    return tensor

def name2idxs(name):
    return [ord(char) - ord('a') for char in name]

def cat2idxs(category, name_category):
    return [name_category.index(category)]

class RNN(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(RNN, self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        # lstm
        self.lstm = nn.LSTM(input_dims, hidden_dims)

        # init the hidden states       
        self.hidden = self.init_hidden()
        
        # hidden2out
        self.h2o = nn.Linear(hidden_dims, output_dims)

        self.softmax = nn.LogSoftmax()

    def forward(self, name_tensor):
        # lstm
        lstm_output, self.hidden = self.lstm(name_tensor, self.hidden)

        # get the final hidden state
        output = lstm_output[-1]

        # hidden2out
        output = self.h2o(lstm_output[-1])

        # softmax
        output = self.softmax(output)
        return output

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_dims)), Variable(torch.zeros(1, 1, self.hidden_dims))

def train(train_data, train_truths, input_dims, hidden_dims, output_dims, n_epochs, all_category, if_draw_plot=False):
    model = RNN(input_dims, hidden_dims, output_dims)
   
    loss_function = nn.NLLLoss()
    learning_rate = 0.05
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    all_losses = []
    n_samples = len(train_data)

    for epoch in xrange(1, n_epochs + 1):
        curr_loss = 0.0
        for name, truth in zip(train_data, train_truths):
            name_idxs = name2idxs(name)
            name_tensor = one_hot_encode(26, name_idxs).float()
            name_var = Variable(name_tensor.view(name_tensor.size()[0],1,name_tensor.size()[1]))

            truth_idxs = cat2idxs(truth, all_category)
            truth_tensor = torch.LongTensor(truth_idxs)
            truth_var = Variable(truth_tensor)

            model.zero_grad()
            model.hidden = model.init_hidden()

            output = model(name_var)

            # If you have a single element with 1139 possible label, 
            # then output should be 1x1139 and target should be a 
            # LongTensor of size 1 (containing the index of the correct label).
            # maybe not one_hot for target

            loss = loss_function(output, truth_var)
            loss.backward()
            optimizer.step()

            curr_loss += loss.data[0]
        
        mean_loss = curr_loss / n_samples
        all_losses.append(mean_loss)
        print '[{:d}/{:d}] Loss: {:.3f}'.format(epoch, n_epochs, mean_loss)
    
    if if_draw_plot:
        draw_plot(all_losses)
    return model
def draw_plot(losses):
    plt.figure()
    plt.plot(losses)   
def prepare_data(name_category):
    data = []
    for cate, names in name_category.iteritems():
        for n in names:
            data.append((n, cate))
    random.shuffle(data)
    return data

def test(model, test_data, test_truths, all_category):
    correct = 0
    n_test = len(test_data)

    for name, truth in zip(test_data, test_truths):
        name_idxs = name2idxs(name)
        name_tensor = one_hot_encode(26, name_idxs).float()
        name_var = Variable(name_tensor.view(name_tensor.size()[0],1,name_tensor.size()[1]))

        truth_idxs = cat2idxs(truth, all_category)

        output = model(name_var)
        _, pred = torch.max(output.data, 1)
        pred = pred[0]
        
        correct += (pred == truth_idxs[0])

    print "total: {}, correct: {}, acc: {}".format(n_test, correct, correct * 1.0 / n_test)

def prepare_args():
    parser = argparse.ArgumentParser(description="Name classification")
    parser.add_argument('-t', '--n_train', default=1000, type=int,
            help='Number of training samples')
    parser.add_argument('-e', '--n_test', default=1000, type=int,
            help='Number of testing samples')
    parser.add_argument('-n', '--n_epochs', default=30, type=int,
            help='Number of epochs')
    parser.add_argument('-p', '--show_plot', default=False, type=bool,
            help='Number of epochs')
    
    return parser.parse_args()


def main():
    opts = prepare_args()

    name_category = build_name_dict()
    print "{}\t {}".format("category", "count")
    for cate, name in name_category.iteritems():
        print "{}\t {}".format(cate, len(name))
    print

    names, truth = zip(*prepare_data(name_category))
    
    n_train = opts.n_train
    n_test = opts.n_test
    n_epochs = opts.n_epochs
    if_draw_plot = opts.show_plot

    train_data = names[: n_train]
    train_truths = truth[: n_train]

    test_data = names[: n_train]
    test_truths = truth[n_train: n_train + n_test]

    input_dims = 26
    hidden_dims = 128
    output_dims = len(name_category)

    all_category = name_category.keys()

    model = train(train_data, train_truths, input_dims, hidden_dims, output_dims, n_epochs, all_category, if_draw_plot)
    test(model, test_data, test_truths, all_category)

    if if_draw_plot:
        plt.show()
if __name__ == "__main__":
    main()