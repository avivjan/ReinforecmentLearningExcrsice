import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import random
import numpy as np

random.seed(0)

file_name = "mnist_parameter_optimization"

def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    np.save("./mnist_deep.npy", losses)
    plt.show()
    
# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 100
batch_size = 32
learning_rate = 1e-4
hidden_size = 500

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_classes)
        
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


net = Net(input_size, num_classes)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


net.train(True)

losses = []
# Train the Model
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):# batch
        #zero the gradients
        optimizer.zero_grad()
        
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28))# change to vector
        labels = Variable(labels)
        
        #forward
        out = net(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()    
        total_loss += loss.item()
            

        
    print('Epoch_id: %d, loss: %f' % (epoch, loss.item()))
    losses.append(total_loss / len(train_loader))
    
net.eval()
# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    # TODO: implement evaluation code - report accuracy
    outputs = net(images)
    _, max_indicies = torch.max(outputs.data, 1) # max, argmax = torch.max()
    correct += (max_indicies == labels).sum().item()
    total += labels.size(0)

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

plot_loss(losses)

# Save the Model
torch.save(net.state_dict(), file_name+".pkl")
