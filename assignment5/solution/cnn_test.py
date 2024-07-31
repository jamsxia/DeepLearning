#!/usr/bin/python3
import numpy as np
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from cnn import MNISTConvNet

MODEL = 'mnist.pt'

def test(model, test_loader):

    model.eval()

    confusion = np.zeros((10, 10)).astype(int)

    with torch.no_grad():

        for data, target in test_loader:

            out = model(data)

            _, preds = out.max(dim=1)

            for pred, targ in zip(target, preds):
                confusion[pred][targ] += 1

        for k in range(10):
            sys.stdout.write('  %4d' % k)
        sys.stdout.write('\n')
        for j in range(10):
            sys.stdout.write('%d ' % j)
            for k in range(10):
                sys.stdout.write('%4d  ' % confusion[j, k])
            sys.stdout.write('\n')


def main():

    test_dataset = MNIST(".", train=False, 
                         download=True, transform=ToTensor())

    test_loader = DataLoader(test_dataset, 
                             batch_size=64, 
                             shuffle=False)

    model = MNISTConvNet()

    print('Loading model %s ...' % MODEL)

    model.load_state_dict(torch.load(MODEL)) 

    test(model, test_loader)

main()
