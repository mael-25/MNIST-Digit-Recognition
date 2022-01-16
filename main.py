import argparse as ap
import json
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms

import sort

parser = ap.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=64, help="default is 64")
parser.add_argument("--epochs", type=int, default=50, help="default is 50")
parser.add_argument("--learning-rate-type", type=str, default="manual", help="""write 'manual'(default) or 'automatic', if manual, you need to type all the learning rates {epoch:lr} at '--learning-rate', if automatic, type the epochs that you want the learning rate to decrease at '--learning-rate-decrease-epochs' as a list, the quantity '--learning-rate-decrease-quantity' as an integer, and the starting learning rate at '--learning-rate-default-value'""")
parser.add_argument("--learning-rate", type=dict, default={1: 0.01, 5: 0.0075, 10: 0.005,20: 0.003,25: 0.0025,30: 0.001,45: 0.0005})
parser.add_argument("--learning-rate-decrease-epochs", type=list, default=[10,25, 40])
parser.add_argument("--learning-rate-decrease-quantity", type=int, default=5)
parser.add_argument("--learning-rate-default-value", type=float, default=0.1)
parser.add_argument("--weight-decay", type=float, default=1e-4)
config = parser.parse_args()


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
num_of_images = 30
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')


input_size = 784 ## Square of 28
hidden_sizes = [512, 256, 128, 64, 32,16]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[2], hidden_sizes[3]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[3], hidden_sizes[4]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[4], hidden_sizes[5]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[-1], output_size),
                      nn.LogSoftmax(dim=1))


layers=len(hidden_sizes)+1

# print("Logs/MNIST-epochs={}-layers={}-{}.pt\n\n\n\n\n".format(config.epochs, layers, num))
if os.path.exists("Logs/MNIST-epochs={}-layers={}-{}.pt".format(config.epochs,layers,num)):
    a = input("Logs/MNIST-epochs={}-layers={}-{}.pt exists, do you want to overwrite it? (y/n)".format(config.epochs,layers,num))
    if a.lower() == "y":
        pass
    else:
        exit()

# print("EXECUTED")

def train():
    #train
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
    


def val():
    correct_count, all_count = 0, 0
    for images,labels in valloader: ##val
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)

            
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
    
    return correct_count, all_count
            

def test():
    correct_count, all_count = 0, 0
    for images,labels in valloader: ##test
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)

            
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
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
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=config.weight_decay)


time0 = time()
epochs = config.epochs
print('{} epochs'.format(epochs))
validation_results = {}
max_score = 0
max_score_epoch = 0



if config.learning_rate_type == 'automatic':
    lr = {}
    cr = config.learning_rate_default_value
    for x in config.learning_rate_decrease_epochs:
        lr[x] = cr/config.learning_rate_decrease_quantity
        cr /=config.learning_rate_decrease_quantity

else:
    lr = config.learning_rate

print(lr)


for e in range(1,epochs+1):
    if e in lr:
        print("LEARNING RATE CHANGED!\n")
        for g in optimizer.param_groups:
            g['lr'] = lr[e]

    train()
    correct_count, all_count = val()
    validation_results[e] = correct_count/all_count
    print("Epoch {} - Validation - Current Model Accuracy: {}\n".format(e, correct_count/all_count))
    if max_score < correct_count/all_count:
        max_score = correct_count/all_count
        max_score_epoch = e
        torch.save(model, 'Logs/MNIST-epochs={}-layers={}-{}.pt'.format(epochs, layers, num)) 


print("\nTraining + Validation Time (in minutes) =",(time()-time0)/60)

# print("Best val results")
# print(validation_results)
print("The best score is {} realized on the {} epoch".format(max_score, max_score_epoch))


correct_count, all_count = test()

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

# torch.save(model, 'Logs/MNIST-epochs={}-layers={}-{}.pt'.format( epochs, layers, num)) 

# print(model)


f = open("scores.json")
dictionary = json.load(f)
dictionary['Logs/MNIST-epochs={}-layers={}-{}.pt'.format( epochs, layers, num)] ={"score":correct_count/all_count, "max-score":max_score,"max-score-epoch":max_score_epoch, "epochs":epochs, "model-input-size":input_size, "model-hidden-sizes":hidden_sizes, "model-output-size":output_size, "validation-results":validation_results, "learning-rate":lr, "batch-size":config.batch_size, "weight-decay":config.weight_decay}#, "model":model} ## incomplete
json.dump(dictionary, open("scores.json", "w"), indent=4)

sort.sort_dict()


print(lr)
