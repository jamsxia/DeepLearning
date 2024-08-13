import matplotlib.pyplot as plt
import torch

dev ="cpu"#"cuda:0" if torch.cuda.is_available() else "cpu"                                                
print("Running on " + dev)



def train(train_loader,
          classifier,
          optimizer,
          epochs,
          loss_fn, 
          lr
          ):
  classifier = classifier.to(dev)
  classifier.train()
  loss_lt = []
  #running_loss = 0.0
  for epoch in range(epochs):
  #while(running_loss/len(train_loader)>0.5):
    running_loss = 0.0
    for minibatch in train_loader:
      data, target = minibatch
      data=data.to(dev)
      target=target.to(dev)
      data = data.flatten(start_dim=1)
      out = classifier(data)
      computed_loss = loss_fn(out, target)
      computed_loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      # Keep track of sum of loss of each minibatch
      running_loss += computed_loss.item()
    loss_lt.append(running_loss/len(train_loader))
    print("Epoch: {} train loss: {}".format(epoch+1, running_loss/len(train_loader)))

  plt.plot([i for i in range(1,epochs+1)], loss_lt)
  plt.xlabel("Epoch")
  plt.ylabel("Training Loss")
  plt.title(
      "MNIST Training Loss: optimizer {}, lr {}".format("SGD", lr))
  plt.show()

  # Save state to file as checkpoint
  torch.save(classifier.state_dict(), 'mnist.pt')