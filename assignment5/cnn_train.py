import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from cnn import MNISTConvNet

from time import time

torch.manual_seed(0)
#dev ="cuda:0" if torch.cuda.is_available() else "cpu"                                                
#print("Running on " + dev)

def train(dev,
          trainloader,
          model,
          optimizer,
          num_epochs,
          loss_fn, 
          ):
    for epochs in range(num_epochs):
        running_loss = 0.0
        num_correct = 0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs.to(dev))
            loss = loss_fn(outputs, labels.to(dev))
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            _, idx = outputs.max(dim=1)
            num_correct += (idx == labels.to(dev)).sum().item()
        print('Epoch: {} Loss: {} Accuracy: {}'.format(epochs+1,running_loss/len(trainloader),
                num_correct/len(trainloader)))

    print('Saving model in minst.pt')

    torch.save(model.state_dict(), 'mnist.pt')

trainset = MNIST('.', train=True, download=True, 
                      transform=ToTensor())
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
lr = 1e-4
num_epochs = 40

device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'

print('Running on ' + device)

model = MNISTConvNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
start = time()
train(device, trainloader,model,optimizer,num_epochs,loss_fn)
print('%d epoch(s) took = %d seconds' % (num_epochs, int(time()-start)))
