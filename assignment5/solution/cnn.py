import torch.nn as nn

class MNISTConvNet(nn.Module):

    def __init__(self, nhid1=30, nhid2=60, nhid3=1000,
                 conv_kernel_size=5, maxpool_kernel_size=2, 
                 dropout=0.5):

        super(MNISTConvNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, nhid1, conv_kernel_size, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(maxpool_kernel_size)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(nhid1, nhid2, conv_kernel_size, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(maxpool_kernel_size)
        )

        output_width = 28 // maxpool_kernel_size**2
        output_height = 28 // maxpool_kernel_size**2

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_width*output_height*nhid2, nhid3),
            nn.Dropout(dropout),
            nn.Linear(nhid3, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.fc1(x)
