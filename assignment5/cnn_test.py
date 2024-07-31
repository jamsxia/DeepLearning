import torch
import matplotlib.pyplot as plt
import numpy as np


def print_confusion_matrix(confusion):
  print('    0    1    2    3    4    5    6    7    8    9')
  print('    ----------------------------------------------')
  for j in range(10):
    print(j, end=' | ')
    for k in range(10):
        print('%-5d' % confusion[j,k], end='')
    print()

def test(test_loader,
          out_classifier, 
          loss_fn):
  accuracy = 0.0
  computed_loss = 0.0
  out_classifier.eval()

  confusion = np.zeros((10,10)).astype('int')

  with torch.no_grad():
      for data, target in test_loader:
          data = data.flatten(start_dim=1)
          out = out_classifier(data)
          _, preds = out.max(dim=1)

          for (j,k) in zip(target, preds):
              confusion[j,k] += 1

          # Get loss and accuracy
          computed_loss += loss_fn(out, target)
          accuracy += torch.sum(preds==target)
          
      print("Test loss: {}, test accuracy: {}".format(
          computed_loss.item()/(len(test_loader)*64), accuracy*100.0/(len(test_loader)*64)))

  print_confusion_matrix(confusion)
