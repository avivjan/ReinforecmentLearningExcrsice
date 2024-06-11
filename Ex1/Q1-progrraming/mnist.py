import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import random 

random.seed(0)

def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('mnist.png')
    plt.show()
    

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 100
batch_size = 100
learning_rate = 1e-3

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
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out
    


net = Net(input_size, num_classes)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


net.train(True)

losses = []
# Train the Model
for epoch in range(num_epochs):
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
        
        #print loss for each batch of data
        #print('Epoch_id: %d, batch_id: %d, loss: %f' % (epoch,i, loss.item()))
    print('!!!!!!!!!!!Epoch_id: %d, loss: %f!!!!!!!!!!!!!!!!!!' % (epoch, loss.item()))
    losses.append(loss.item())
    
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

# Save the Model
plot_loss(losses)

torch.save(net.state_dict(), 'mnist.pkl')
