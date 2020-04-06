#! /usr/bin/python
# -*- coding: utf-8 -*-
# Author: Wenyuan Wu, 18746867
# Date: 22.03.2020
# Additional Info:
# Programming assignment 2: Perceptron
# Code Skeleton inspired by Sebastian Raschka:
# Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow 2, 3rd Edition
# Raschka, S and Mirjalili, V.
# https://books.google.ch/books?id=sKXIDwAAQBAJ
# Packt Publishing, 2019
# 1 more element is added in the weights vector(first one) to represent the bias unit
# stopping criterion: if each error equals 0 in last 4 epochs, stop fitting.
# GitHub Repository:
# https://github.com/wenyuan-wu/sa_pa2


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron(object):
    """
    Perceptron classifier.

    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    random_state : int
    Random number generator seed for random weight initialization.

    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    errors_ : list
    Number of misclassifications in every epoch.
    """
    def __init__(self, eta=0.01, random_state=1):
        self.eta = eta
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples
        is the number of samples and
        n_features is the number of features.
        y : array-like, shape = [n_samples]
        Target values.

        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        while True:
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            if sum(self.errors_[-4:]) == 0:
                break
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, -1)


def output_weights(weights, file_name):
    with open(file_name, 'w', encoding='utf-8') as outfile:
        for w in weights:
            outfile.write(str(w))
            outfile.write('\n')


def main():
    df = pd.read_csv('pa2_input.txt', sep='\t', index_col=0)
    X = df.iloc[:, :83].values
    y = df.iloc[:, 84].values
    y = np.where(y == 'WAR', -1, 1)
    ppn = Perceptron(eta=0.1)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()
    output_weights(ppn.w_, 'weights_pa2.txt')


if __name__ == '__main__':
    main()
