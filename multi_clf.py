import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import argparse

#multiclassification #pytorch

class Neural_Net(nn.Module):
    def __init__(self, n_in_feats, n_class):
        # define layers

        super(Neural_Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in_feats, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_class),
        )

    def forward(self, input):
        # forward pass

        return self.net(input)

def generate_cases(n_data, n_class):
    # n_data: number of cases
    # n_class: number of classes

    labels = range(n_class)
    features, truth = [], []
    for i in range(n_data):
        category = (np.random.choice(labels))
        features.append([np.random.uniform(-1.2, 1.2) + category, 
                            np.random.uniform(-1.2, 1.2)+ category])
        truth.append(category)
    return features, truth

def lose2dense(loose_data):
    # convert loose labels to dense one

    mapping = {}
    dense_data = []
    index = 0

    for item in loose_data:
        if item not in mapping:
            mapping[item] = index
            index += 1
        dense_data.append(mapping[item])
    return dense_data, mapping

def one_hot(num_labels, data):
    # num_labels
    # data need to be encoded
    # rtype: encoded data

    # view = np.reshape: -1 means the dimension you not sure 
    index = torch.LongTensor(data).view(-1, 1)
    bit_mask = torch.LongTensor(len(data), num_labels).zero_()
    # scatter the 1 to index positions
    bit_mask.scatter_(1, index, 1)
    return bit_mask


def draw_plot(ax, feature, label, title):
    marker_map = {0: 'h', 1: 'o', 2: 'v', 3: '*', 4: 'd', 5: 'H', 6: 'p'}
    color_map = {0: 'b', 1: 'g', 2: 'r', 3: 'c', 4: 'm', 5: 'y', 6: 'k'}
    for feature, l in zip(feature, label):
        x, y = feature
        ax.set_title(title)
        ax.scatter(x, y, marker=marker_map[l], c=color_map[l])
    

def train_classifer(train_feats, train_truths, n_epochs=30, batch_size=20):
    assert len(train_feats) == len(train_truths), "Number of data and labels are not match"
    assert len(train_feats) > 0, "Empty Data"
    
    n_in_feats = len(train_feats[0])
    n_label = len(train_truths[0])
    
    train_feats = torch.FloatTensor(train_feats)
    if type(train_truths) == torch.LongTensor:
        train_truths = train_truths.float()
    else:
        train_truths = torch.FloatTensor(train_truths)

    clf = Neural_Net(n_in_feats, n_label)
    
    # init the optimizer
    optimizer = optim.Adam(clf.parameters())

    # loss function
    criterion = nn.MultiLabelSoftMarginLoss()

    for epoch in range(n_epochs):
        losses = []
        
        for i in xrange(len(train_truths) / batch_size): 
            input_var = Variable(train_feats[i * batch_size: (i + 1) * batch_size]).view(-1, n_in_feats)
            label_var = Variable(train_truths[i * batch_size: (i + 1) * batch_size]).view(-1, n_label)
            
            # zero the parameter gradients
            optimizer.zero_grad()
                
            # forward + backward + optimize
            output = clf(input_var)
            loss = criterion(output, label_var)
            
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean())
        
        # print statistics
        print '[{:d}/{:d}] Loss: {:.3f}'.format(epoch + 1, n_epochs, np.mean(losses))

    return clf

def main():

    parser = argparse.ArgumentParser(description="Multi classification")
    parser.add_argument('-t', '--n_train', default=1000, type=int,
            help='Number of training samples')
    parser.add_argument('-e', '--n_test', default=1000, type=int,
            help='Number of testing samples')
    parser.add_argument('-c', '--n_class', default=5, type=int,
            help='Number of classes')
    parser.add_argument('-n', '--n_epochs', default=30, type=int,
            help='Number of epochs')
    parser.add_argument('-b', '--batch_size', default=10, type=int,
            help='Number of epochs')
    parser.add_argument('-p', '--show_plot', default=False, type=bool,
            help='Number of epochs')
    
    opts = parser.parse_args()

    n_train = opts.n_train
    n_test = opts.n_test
    n_class = opts.n_class
    n_epochs = opts.n_epochs
    batch_size = opts.batch_size
    show_plot = opts.show_plot

    train_data, train_truth = generate_cases(n_train, n_class)
    train_truth, mapping = lose2dense(train_truth)
    train_truth = one_hot(len(mapping), train_truth)
    
    test_data, test_truth = generate_cases(n_test, n_class)
    test_truth = [mapping[t] for t in test_truth]

    clf = train_classifer(train_data, train_truth, n_epochs, batch_size)

    error_x, error_y = [], []
    
    correct = 0
    prediction = []
    for x, y  in zip(test_data, test_truth):
        inputv = Variable(torch.FloatTensor(x)).view(1, -1)
        output = clf(inputv)
        _, pred = torch.max(output.data, 1)

        pred = pred[0]
        prediction.append(pred)
        if y == pred:
            correct += 1
        else:
            error_x.append(x)
            error_y.append(pred)
            #print x, mapping[y], pred

    n_train = opts.n_train
    n_test = opts.n_test
    n_class = opts.n_class
    n_epochs = opts.n_epochs
    batch_size = opts.batch_size
    show_plot = opts.show_plot

    print "number of classes: ", n_class
    print "number of training data: ", n_train
    print "number of testing  data: ", n_test
    print "number of epochs: {}, batch size:, {}".format(n_epochs, batch_size)
    print
    print "total: {}, correct: {}, acc: {}".format(n_test, correct, correct * 1.0 / n_test)

    if show_plot:
        f, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(8, 5))
        if error_x:
            draw_plot(ax1, error_x, error_y, "Error")
        draw_plot(ax2, test_data, prediction, "Pred")
        draw_plot(ax3, test_data, test_truth, "Truth")
        plt.show()

if __name__ == '__main__':
    main()





