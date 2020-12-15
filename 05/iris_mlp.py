import torch
import numpy as np

#--------------------------------------------------------------
# Load the data

from sklearn import datasets
iris = datasets.load_iris() # use the Iris data.

X = iris.data
y = iris.target

N, M = X.shape
o = np.random.choice(N, size = N, replace = False)
Xtrain = X[o[:100],:]
Xtest = X[-o[100:],:]
ytrain = y[o[:100]]
ytest = y[-o[100:]]


#--------------------------------------------------------------
# Define the neural network

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()

#--------------------------------------------------------------
# Set the loss function

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001) #, momentum = 0.9, weight_decay = 0.01)

#--------------------------------------------------------------
# Train the network

# convert numpy arrays to pytorch tensors
inputs = torch.from_numpy(Xtrain).float()
labels = torch.from_numpy(ytrain).long()

for i in range(10000):        
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    print('[%5d] loss: %.3f' % (i + 1, loss), end='\r', flush=True)


print('\nFinished Training')

#--------------------------------------------------------------
# Print the accuracy

testinputs = torch.from_numpy(Xtest).float()
testlabels = torch.from_numpy(ytest).long()
testoutputs = net(testinputs)
    
_, predicted = torch.max(outputs.data, 1)
print(predicted)

_, testpredicted = torch.max(testoutputs.data, 1)

trainacc = ((predicted == labels).float()).mean()
testacc = ((testpredicted == testlabels).float()).mean()
print("Training accuracy: %.3f" % trainacc)
print("Test accuracy: %.3f" % testacc)

