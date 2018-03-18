from __future__ import print_function, division
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd 
import numpy   as np 
import scipy.misc # to visualize only  
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class modified_mnist(Dataset):
    """Modified mnist dataset"""

    def __init__(self, X_in, Y_in, transform=None):
        
        """
        Args:
            X_in: 2D np array from csv file (Nx4096).
            Y_in: 1D np array from csv file (N)
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        
        self.labels = Y_in#np.loadtxt(csv_file_Y, delimiter=",") #tutorial imported with pandas
        self.images = X_in#np.loadtxt(csv_file_X, delimiter=",")
        self.transform = transform #change transform to try various preprocessings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform((image, label)) #Applies transform if needed
            
        else: sample = (image, label)
            
        return sample

from sklearn import preprocessing


def preProcSteps(sample):
    
    
        '''In:
            image - ndarray, 1x4096
            target: ndarray, 1
        Out: pytorch CNN format
            image: doubleTensor [1x64x64]
            target: longTensor [1]
        '''
        
        image, target = sample
    
        scaler = preprocessing.MinMaxScaler()
                
        image = image.reshape(64,64)
        
        image = scaler.fit_transform(image)
        
        #print(image)
        
        #REPORT PREPROCESSING STEP
        image = image.reshape(1,64,64)  
        image = torch.from_numpy(image)
        
        
        #The target must be a long tensor (bug?)
        target = torch.from_numpy(np.asarray([target])).long()    
        return (image, target)
    
class ToTensor(object):
    """Only a wrapper to be able to use pytorch implementation, everything in preProcSteps"""

    def __call__(self, sample):
           return preProcSteps(sample)

class BackgroundFilter(object):
    
    def filter (x):
        if (x < 250.0):
            x = 0.0
        return x
    
    def __call__(self, sample):
        
        image, target = sample
        vectFilter = np.vectorize(filter)
        processed_image = vectFilter(image).astype(float)
        
        return (processed_image, target)
    

url_X_train = "https://doc-08-84-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/3cb5fua6bsfo6fpnvphm3oklmadgo8f4/1521396000000/10970379748800439747/*/1RHRuWeoSGVc0xQQ5Agvt-XkINx37vr5a?e=download"
url_Y_train = "https://doc-0o-84-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/9cut65npeuvp9evcfk0a0nq1al25hnan/1521396000000/10970379748800439747/*/1PuENkRYGxw3bJ-m0HQOLT24tFqOPyCf1?e=download"

import urllib

with urllib.request.urlopen(url_Y_train) as testfile, open('train_y_remote.csv', 'w') as f:
    f.write(testfile.read().decode())
    
with urllib.request.urlopen(url_X_train) as testfile, open('train_x_remote.csv', 'w') as f:
    f.write(testfile.read().decode())




X_train = np.loadtxt('train_x_remote.csv', delimiter = ',')
Y_train = np.loadtxt('train_y_remote.csv', delimiter = ',')
#X_train = np.loadtxt('dev_mini_sample_test_X.csv', delimiter = ',')
#Y_train = np.loadtxt('dev_mini_sample_test_Y.csv', delimiter = ',')

kwargs = {'num_workers': 1, 'pin_memory': True}


mod_mnist_train = modified_mnist(X_in = X_train[:40000],
                                Y_in = Y_train[:40000],
                                           transform=transforms.Compose([
                                               #BackgroundFilter(),
                                               ToTensor(),
                                           ]))


#Loader is an iterator wrapper to facilitate the use of different batches for example
mod_mnist_train_loader = DataLoader(mod_mnist_train, batch_size= 64,
                        shuffle=True, num_workers=1, pin_memory=1)

mod_mnist_test = modified_mnist(X_in=X_train[40000:48000],
                                    Y_in=Y_train[40000:48000],
                                           transform=transforms.Compose([
                                               #BackgroundFilter(),
                                               ToTensor(),
                                           ]))

mod_mnist_test_loader = DataLoader(mod_mnist_test, batch_size=64,
                        shuffle=True, num_workers=1, pin_memory=1)

mod_mnist_classes = np.arange(1.0,10.0,1.0) #numerical labels from 0.0 to 9.0


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


#kwargs = {'num_workers': 1}
class CONVnet(nn.Module):
    def __init__(self):
        super(CONVnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5,padding = 2) #Padding needed to get right input
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #UPDT 3:17 FRIDAY, changed KSIZE TO 5
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        #print('type of input to conv1 - : {}'.format(type(x.data)))
        x = F.relu(F.max_pool2d(self.conv1(x).double(),4))
        #print('type of input to conv2 - : {}'.format(type(x.data)))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 3))
        #print('type of output of conv2 - : {}'.format(type(x.data)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = CONVnet().double()

#CUDA!!
model.cuda()

'''
    MOST RECENT - Training loop:
    Choose a model, so far we have
    
        - Linear_model_1:
              (fc1): Linear(in_features=4096, out_features=2862, bias=True)
              (bc1): BatchNorm1d(2862, eps=1e-05, momentum=0.1, affine=True)
              (fc2): Linear(in_features=2862, out_features=1316, bias=True)
              (bc2): BatchNorm1d(1316, eps=1e-05, momentum=0.1, affine=True)
              (fc3): Linear(in_features=1316, out_features=10, bias=True)
              
        - CONVnet:
              (conv1): Conv2d(1, 10, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
              (conv2): Conv2d(10, 20, kernel_size=(4, 4), stride=(1, 1))
              (conv2_drop): Dropout2d(p=0.5)
              (fc1): Linear(in_features=320, out_features=50, bias=True)
              (fc2): Linear(in_features=50, out_features=10, bias=True)
              
        - Loss: Cross entropy loss
        '''

#MOST RECENT TRAIN
def train(train_loader, model, optimizer,args, epoch ):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
       #if args.cuda:
        data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target.view(target.size()[0]))
        loss.backward()
        optimizer.step()
        
        return loss
        #if batch_idx % args['log_interval'] == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #    100. * batch_idx / len(train_loader), loss.data[0]))


#THIS ONE WORKS
def testing_loop(test_loader, model):  
    
    
    model.eval()
    test_loss = 0
    correct = 0
    
    for data, target in test_loader:

    	#CUDA!!
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        
        target = target.view(target.size()[0])
        
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().sum()

    test_loss /= len(test_loader.dataset)
    
    avg_loss = test_loss
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        accuracy))
    
    return avg_loss, correct, accuracy, output, pred



model = CONVnet().double()

train_arguments = {'momentum' : 0.5,
                  'lr' : 0.01,
                  'log_interval': 3}
optimizer = optim.SGD(model.parameters(), lr=train_arguments['lr'],
                      momentum= train_arguments['momentum'])

losses_train_list = []
losses_test_list = []
accuracy_test_list = []


for epoch in range(1, 300):
    print('Epoch {}'.format(epoch))
    
    train_loss = train(mod_mnist_train_loader, model, optimizer,train_arguments, epoch)
    avg_test_loss, correct, test_accuracy, output, pred = testing_loop(mod_mnist_test_loader, model)
    
    
    #Updating list for plotting
    #losses_train_list.append(train_loss.data.numpy)
    losses_test_list.append(avg_test_loss)
    #print (type(losses_test_list))
    accuracy_test_list.append(test_accuracy)
    
    #Taking a snapshot
    #torch.save(model, "cnn/saves/cnn_17_Filtered_Epoch{}".format(epoch))
    #np.savetxt('cnn/results/17/accuracy_test_list.csv', accuracy_test_list, delimiter = ',')
    #np.savetxt('cnn/results/17/losses_test_list.csv', losses_test_list, delimiter = ',')
    #np.savetxt('cnn/results/17/losses_train_list.csv', losses_train_list, delimiter = ',')

