import argparse as ap
import json
import os
import random
from time import time, sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms

from tqdm import tqdm

import sort

parser = ap.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=64, help="default is 64")
parser.add_argument("--epochs", type=int, default=50, help="default is 50")
parser.add_argument("--learning-rate-type", type=str, default="automatic", help="""write 'manual' or 'automatic'(default). If manual, you need to type all the learning rates {epoch:lr} at '--learning-rate', if automatic, type the epochs that you want the learning rate to decrease at '--learning-rate-decrease-epochs' as a list, the quantity '--learning-rate-decrease-quantity' as an integer, and the starting learning rate at '--learning-rate-default-value'""")
parser.add_argument("--learning-rate",default={1: 0.01, 5: 0.0075, 10: 0.005,20: 0.003,25: 0.0025,30: 0.001,45: 0.0005})
parser.add_argument("--learning-rate-decrease-epochs", type=list, default=[10,25, 40])
parser.add_argument("--learning-rate-decrease-quantity", type=int, default=5)
parser.add_argument("--learning-rate-default-value", type=float, default=0.1)
parser.add_argument("--weight-decay", type=float, default=0.0005)
parser.add_argument("--dropout", type=float, default=0.0005)
parser.add_argument("--model-type", type=str, default="conv2d")

config = parser.parse_args()

print("epochs:" , config.epochs)
COLOURS = ['RED', 'GREEN','YELLOW', 'BLUE', 'MAGENTA','CYAN', 'WHITE']


# if type(config.dropout ) == float:
#     dropout = config.dropout

# elif type(config.dropout) == str: 
#     if config.dropout.lower() == "r":
#         dropout = random.randint(0, 75)/100

#     else:
#         assert "ERROR OF DROPOUT!"

# else:
#     assert "ERROR OF DROPOUT!"

# #######################

# if type(config.weight_decay ) == float:
#     weight_decay = config.weight_decay

# elif type(config.weight_decay) == str: 
#     if config.weight_decay.lower() == "r":
#         weight_decay = random.randint(0, 75)/100

#     else:
#         assert "ERROR OF weight decay!".upper()

# else:
#     assert "ERROR OF weight decay!".upper()

# #########################

if config.learning_rate_type == 'automatic':
    lr = {}
    cr = config.learning_rate_default_value
    for x in config.learning_rate_decrease_epochs:
        lr[x] = cr/config.learning_rate_decrease_quantity
        cr /=config.learning_rate_decrease_quantity

else:
    lr = config.learning_rate

print(lr)
print(config.learning_rate)

# print(lr, "1!")

# # try:LR = int(config.learning_rate) 
# # except: LR = config.learning_rate

# # if type(LR) == float:
# #     lr = {}
# #     lr[1] = config.learning_rate

# # elif type(LR) == str: 
# #     lr = {}
# #     if config.learning_rate.lower() == "r":
# #         lr[1] = random.randint(0, 75)/100

# #     else:
# #         assert "ERROR OF LR!".upper()

# # elif type(LR) == int : 
    
# #     lr = {}

# #     for x in range(1, config.epochs+1, LR):
# #         lr[x] = random.randint(0, 75)/100

    

# #     else:
# #         assert "ERROR OF LR!".upper()
# #     print(lr, LR, "")
# # else:
# #     assert "ERROR OF LR!".upper()

# # print("COMPLETED", type(config.learning_rate))
num=1

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])


trainset = datasets.MNIST('TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=False)

dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

figure = plt.figure()
num_of_images = 10
# if config.model_type == 'linear':
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')


# input_size = 1 ## Square of 28
# hidden_sizes = [16 ,32, 64, 32]
# output_size = 10
input_size = 784 ## Square of 28
hidden_sizes = [512,
                256,
                128,
                64,
                32,
                16]
output_size = 10

if str(config.model_type).lower() == "linear":
    TYPE = nn.Linear
elif str(config.model_type).lower() == "conv2d":
    TYPE = nn.Conv2d
else:
    assert "ERROR! --model-type isn't 'Linear' or 'Conv2D'"

# print(TYPE)

# class Net(nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square, you can specify with a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# net = Net()
# print(net)

random_data = torch.rand((1, 1, 28, 28))

# my_nn = Net()
# result = my_nn(random_data)
# print (result, "NUMBER 1")
# print (result, "NUMBER 2")
# print (result, "NUMBER 3")

# # # # # # model=net

# model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
#                       nn.Dropout(p=dropout),
#                       nn.ReLU(),
#                       nn.Linear(hidden_sizes[0], hidden_sizes[1]),
#                       nn.Dropout(p=dropout),
#                       nn.ReLU(),
#                       nn.Linear(hidden_sizes[1], hidden_sizes[2]),
#                       nn.Dropout(p=dropout),
#                       nn.ReLU(),
#                       nn.Linear(hidden_sizes[2], hidden_sizes[3]),
#                       nn.Dropout(p=dropout),
#                       nn.ReLU(),
#                     #   nn.Linear(hidden_sizes[3], hidden_sizes[4]),
#                     #   nn.Dropout(p=dropout),
#                     #   nn.ReLU(),
#                     #   nn.Linear(hidden_sizes[4], hidden_sizes[5]),
#                     #   nn.Dropout(p=dropout),
#                     #   nn.ReLU(),
#                       nn.Linear(hidden_sizes[-1], output_size),
#                       nn.LogSoftmax(dim=1))



model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.Dropout(p=config.dropout),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.Dropout(p=config.dropout),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                      nn.Dropout(p=config.dropout),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[2], hidden_sizes[3]),
                      nn.Dropout(p=config.dropout),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[3], hidden_sizes[4]),
                      nn.Dropout(p=config.dropout),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[4], hidden_sizes[5]),
                      nn.Dropout(p=config.dropout),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[-1], output_size),
                      nn.LogSoftmax(dim=1))


# model = nn.Sequential(nn.Conv2d(1, 16, 3),
#                     ###   nn.Dropout(p=dropout),
#                       nn.ReLU(),
#                       nn.Conv2d(16, 32, 3),
#                     ###   nn.Dropout(p=dropout),
#                       nn.ReLU(),
#                     #   nn.Conv2d(32, 64, 3),
#                       nn.AvgPool2d(2),
#                     ###   nn.Dropout(p=dropout),
#                       nn.ReLU(),
#                       nn.Conv2d(32, 128, 3),
#                     ##   nn.Dropout(p=dropout),
#                       nn.ReLU(),
#                       nn.Conv2d(128, 256, 3),
#                     ##  nn.Dropout(p=dropout),
#                       nn.ReLU(),
#                       nn.AvgPool2d(2),
#                       nn.ReLU(),
#                       nn.Conv2d(256, 512, 4),
#                     # #  nn.Dropout(p=dropout),
#                       nn.ReLU(),
#                       nn.Conv2d(512, 128, 1),
#                       nn.ReLU(),
#                       nn.Conv2d(128, 32, 1),
#                       nn.ReLU(),
#                       nn.Conv2d(32, 10, 1),
#                     ##   nn.Linear(9216, 128),
#                     ##   nn.Linear(128, 10),
#                       nn.LogSoftmax(dim=1))



# model = nn.Sequential(nn.Conv2d(input_size, hidden_sizes[0], 3),
#                     ###   nn.Dropout(p=dropout),
#                       nn.BatchNorm2d(hidden_sizes[0]),
#                       nn.ReLU(),
#                       nn.Conv2d(hidden_sizes[0], hidden_sizes[1], 3),
#                     ###   nn.Dropout(p=dropout),
#                       nn.BatchNorm2d(hidden_sizes[1]),
#                       nn.ReLU(),
#                     #   nn.Conv2d(32, 64, 3),
#                       nn.AvgPool2d(8),
#                     ###   nn.Dropout(p=dropout),
#                       nn.BatchNorm2d(hidden_sizes[1]),
#                       nn.ReLU(),
#                       nn.Conv2d(hidden_sizes[1], hidden_sizes[2], 3),
#                     ##   nn.Dropout(p=dropout),
#                       nn.BatchNorm2d(hidden_sizes[2]),
#                       nn.ReLU(),
#                       nn.Conv2d(hidden_sizes[2], hidden_sizes[3], 1),
#                     ##  nn.Dropout(p=dropout),
#                       nn.BatchNorm2d(hidden_sizes[3]),
#                       nn.ReLU(),
#                       nn.Conv2d(hidden_sizes[3], output_size, 1),
#                     ##   nn.Linear(9216, 128),
#                     ##   nn.Linear(128, 10),
#                       nn.LogSoftmax(dim=1))
# model = result
print(str(model))
print(type(str(model)))


print(config)
layers=len(hidden_sizes)+1

# print("Logs/MNIST-epochs={}-layers={}-{}.pt\n\n\n\n\n".format(config.epochs, layers, num))
if os.path.exists("Logs/MNIST-epochs={}-layers={}-{}.pt".format(config.epochs,layers,num)):
    f = False
    while not f:
        a = input("Logs/MNIST-epochs={}-layers={}-{}.pt exists, do you want to overwrite it? (y/n/a). \"a\" is for changing the name".format(config.epochs,layers,num)).lower()
        f = True
        if a.lower() == "y":
            pass
        elif a.lower() == "n":
            exit()
        else:
            f = False

# print("EXECUTED")

def train():
    #train
    running_loss = 0
    # print(len(trainloader))
    # print(trainloader)
    tl = iter(trainloader)
    x = tqdm(range(len(trainloader)), colour=random.choice(COLOURS).lower(), leave=False)
    x2 = iter(x)
    pred = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    lenTL  = len(trainloader)
    crTLPos = 1
    for images, labels in trainloader:
        x.set_description("==> {}-{}".format(e, crTLPos))
        ###images, labels = next(tl) ###NEXT STEP
        # tr2 = iter(tr2)
        # images2, labels2 = next(tr2)
        # print(images2, labels2)
        # Flatten MNIST images into a 784 long vector
        if str(config.model_type).lower() == "linear": images = images.view(images.shape[0], -1)
        else:images = images.view(images.size(0), -1) ##comment if conv2d

    
        # Training pass
        optimizer.zero_grad()
        # o = Variable(output)
        # l = Variable(labels)
        
        output = model(images)
        # output = output.squeeze()
        loss = criterion(output, labels)

        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
        next(x2)
        crTLPos+=1
    else:
        x.set_description("==> Epoch {} TRAINING terminated")
        print("==> Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
    


# def val(epoch):
#     a = random.choice(COLOURS).lower()
#     # a = time()
#     correct_count, all_count = 0, 0
#     # print(len(valloader))
#     # a = 0
#     vl = iter(valloader)
#     # x=0
#     for x in tqdm(range(len(valloader)), desc=">>> Validation epoch {}".format(epoch), colour=a, leave=False): ##val
#         # print(x)
#         images, labels = next(vl)
#         # print(valloader)
#         # print(len(labels))
#         for i in range(len(labels)):
#             if config.model_type=="linear": img = images[i].view(1, 784)
#             with torch.no_grad():
#                 if config.model_type=="linear": logps = model(img)
#                 else:logps = model(images)

            
#             ps = torch.exp(logps)
#             probab = list(ps.numpy()[0])
#             pred_label = probab.index(max(probab))
#             true_label = labels.numpy()[i]
#             if(true_label == pred_label):
#                 correct_count += 1
#             all_count += 1
#             ##print(all_count,", ",i,end='\r')
#         # a +=1
#         # print(a, end="\r")

#     # b = time()-a
#     # print("{}:{} time eclapsed for val".format(int(b//60), int(b%60//1)))
    
#     return correct_count, all_count

def val():
    a = time()
    correct_count, all_count = 0, 0
    # print(len(valloader))
    a = 0
    pred = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    x = tqdm(range(len(valloader)), colour=random.choice(COLOURS).lower(), leave=False)
    x2 = iter(x)
    crVLPos = 1
    b = 0
    for images,labels in valloader: ##val

        x.set_description("==> {}-{}".format(e, crVLPos))
        # print(valloader)
        # print(len(labels))
        next(x2)
        for i in range(len(labels)):
            if str(config.model_type).lower() == "linear": img = images[i].view(1, 784)
            else:images = images.view(images.size(0), -1) ##comment if conv2d

            with torch.no_grad():
                if config.model_type=="linear": logps = model(img)
                else:logps = model(images)

            
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            pred[pred_label] += 1
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
            # print( "Predicted: {} True: {}".format(pred_label, true_label))
            ##print(all_count,", ",i,end='\r')
        a +=1
        b += 1
        # print(a, end="\r")
        crVLPos +=1

    # b = time()-a
    # print("{}:{} time eclapsed for val".format(int(b//60), int(b%60//1)))
    
    x.set_description("==> Epoch {} VALIDATION terminated")
    sleep(0.25)
    print(pred)
  
    return correct_count, all_count
            

def test():
    correct_count, all_count = 0, 0
    pred = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    x = tqdm(range(len(valloader)), colour=random.choice(COLOURS).lower(), leave=False)
    x2 = iter(x)
    for images,labels in valloader: ##test
        for i in range(len(labels)):
            next(x2)
            if config.model_type=="linear":img = images[i].view(1, 784)
            with torch.no_grad():
                if config.model_type=="linear": logps = model(img)
                else:logps = model(images)

            
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
        
    sleep(0.25)
    print(pred)
    return correct_count, all_count

        
            

                      
# # lst = [input_size, a for a in hidden_sizes, output_size]
# lst = [input_size, output_size]
# for a in hidden_sizes: lst.append(a)
# lst.sort(reverse=True)
# print(lst)

# # model = nn.Sequential([nn.Linear(lst[a], lst[a+1]),
# #                       nn.ReLU()]for a in range(len(lst)-1) ,
# #                       nn.LogSoftmax(dim=1))

# model = nn.ModuleList([
#     nn.Linear(lst[a], lst[a+1]) for a in range(len(lst)-1)
#     ])
# model.append(nn.LogSoftmax(dim=1))
# model = nn.Sequential(model)

print(model)

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
if str(config.model_type).lower() == "linear":images = images.view(images.shape[0], -1) ##comment if conv2d
else:images = images.view(images.size(0), -1) ##comment if conv2d

# print(images)

logps = model(images) #log probabilities
## print(logps.detach().numpy().shape)
# logps = logps.squeeze()
## print(logps.detach().numpy().shape)
loss = criterion(logps, labels) #calculate the NLL loss

print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)

optimizer = optim.SGD(model.parameters(), lr=config.learning_rate_default_value, momentum=0.9, weight_decay=config.weight_decay)


time0 = time()
epochs = config.epochs
print('{} epochs'.format(epochs))
validation_results = {}
max_score = 0
max_score_epoch = 0





print(lr)

# correct_count, all_count = val()
# print(correct_count/all_count, "TEST EPOCH")

for e in range(1,epochs+1):
    model.train()
    if e in config.learning_rate:
        # print("LEARNING RATE CHANGED!\n")
        for g in optimizer.param_groups:
            g['lr'] = config.learning_rate[e]

    print("==> Training", end="\r")
    for x in range(3):print('==> Training', '.'*(x+1), end='\r');sleep(0.1)
    print("==> Training ...")
    train()
    model.eval()
    print("==> Validating", end="\r")
    for x in range(3):print('==> Validating', '.'*(x+1), end='\r');sleep(0.1)
    print("==> Validating ...")
    correct_count, all_count = val()
    validation_results[e] = correct_count/all_count
    if max_score < correct_count/all_count:
        max_score = correct_count/all_count
        max_score_epoch = e
        torch.save(model, 'Logs/MNIST-epochs={}-layers={}-{}.pt'.format(epochs, layers, num)) 
    print("==> Epoch {} - Validation - Current Model Accuracy: {}\n\
==> Best accuracy: {}\n".format(e, correct_count/all_count, max_score))


print("\n==> Training + Validation Time (in minutes) =",(time()-time0)/60)

# print("Best val results")
# print(validation_results)
print("==> The best score is {} realized on the {} epoch".format(max_score, max_score_epoch))


correct_count, all_count = test()

print("==> Number Of Images Tested =", all_count)
print("\n==> Model Accuracy =", (correct_count/all_count))

# torch.save(model, 'Logs/MNIST-epochs={}-layers={}-{}.pt'.format( epochs, layers, num)) 

# print(model)


f = open("scores.json")
dictionary = json.load(f)
dictionary['Logs/MNIST-epochs={}-layers={}-{}.pt'.format( epochs, layers, num)] ={"score":correct_count/all_count, "max-score":max_score,"max-score-epoch":max_score_epoch, "epochs":epochs,  "validation-results":validation_results, "learning-rate":config.learning_rate, "batch-size":config.batch_size, "weight-decay":config.weight_decay, "model-class-type":config.model_type, "model":str(model)} ## incomplete
json.dump(dictionary, open("scores.json", "w"), indent=4)

sort.sort_dict()


print(config.learning_rate)