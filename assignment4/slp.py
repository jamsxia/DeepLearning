
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch
from torch import optim
import torch.nn as nn
from mlp_train import train
from mlp_test import test

# For reproducibility
torch.manual_seed(0)
class BaseClassifier(nn.Module):
  def __init__(self, in_dim, feature_dim, out_dim):
    super(BaseClassifier, self).__init__()
    self.classifier = nn.Sequential(
        nn.Linear(in_dim, out_dim, bias=True),
        nn.ReLU(),
    )
    
  def forward(self, x):
    return self.classifier(x)
    
# Instantiate model, optimizer, and hyperparameter(s)
in_dim, feature_dim, out_dim = 784, 256, 10
lr=1e-3
loss_fn = nn.CrossEntropyLoss()
epochs=20
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
optimizer = optim.SGD(classifier.parameters(), lr=lr)



# Load in MNIST dataset from PyTorch
train_dataset = MNIST(".", train=True, 
                      download=True, transform=ToTensor())
test_dataset = MNIST(".", train=False, 
                     download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, 
                          batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, 
                         batch_size=64, shuffle=False)




train(train_loader,classifier,optimizer,epochs,loss_fn,lr)
out=BaseClassifier(in_dim, feature_dim, out_dim)
out.load_state_dict(torch.load('mnist.pt'))
test(test_loader, out, loss_fn)