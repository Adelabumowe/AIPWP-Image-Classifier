import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F 
from torchvision import datasets,transforms,models
import json
import numpy as np
from PIL import Image
import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description = 'Train a neural network with any CNN model of choice')
    parser.add_argument('--data_directory',default="./flowers/",help = 'Directory needed to apply transformations')
    parser.add_argument('--save_dir',help = 'Directory needed to save the model')
    parser.add_argument('--arch',type = str ,help = 'Download from a list of CNN models e.g vgg or desnet')
    parser.add_argument('--gpu',action = 'store_true',help='option to train with gpu or cpu(gpu preferred for faster speed) ')
    parser.add_argument('--learn_rate',type = float,help = 'determine the best learning rate for the optimizer')
    parser.add_argument('--epoch',type = int,help = 'determine the number of epochs')
    parser.add_argument('--hidden_units',type = int,help = 'No of hidden units')
    args = parser.parse_args()
    return args

                                     
def get_dir():
    print('Generating the necessary data!!!!!!')
    train_dir = args.data_directory + '/train'
    valid_dir = args.data_directory + '/valid'
    test_dir = args.data_directory + '/test'
    data_dir = [train_dir,valid_dir,test_dir]
    return transformation(data_dir)

                                     
                                     
def transformation(data_dir):
    train_dir,valid_dir,test_dir = data_dir
    print("Transfomation of Data!!!")
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64,shuffle=True)
    
    loaders = {'train': trainloader,'test': testloader,'valid': validloader,'train_data':train_datasets}
    return loaders

def build_network(data):
    print("Building model network")
    if (args.arch and args.hidden_units) is None:
        print('Automatic set to use vgg model')
        model = models.vgg16(pretrained=True)
        input_units = 25088
        hidden_units = 4096
    else:
        print("Currently using desnet model")
        model = models.densenet121(pretrained=True)
        input_units = 1024
        hidden_units = args.hidden_units
    
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(input_units, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=0.3),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
   
    return model

def train(model,data):
    print('Training model')
    if args.learn_rate is None:
        learn_rate = 0.001
    else:
        learn_rate = args.learn_rate
    if args.epoch is None:
        epoch = 2
    else:
        epoch = args.epoch
    if args.gpu is None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
    model.to(device)
    
    steps = 0
    running_loss = 0
    print_all = 5
    trainloader = data['train']
    validloader = data['valid']
    for e in range(epoch):
        for images,labels in trainloader:
            steps+=1
            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            log_ps = model.forward(images)
            loss = criterion(log_ps,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
         
        if steps % print_all == 0:
            valid_losses = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images,labels in testloader:
                    images,labels = images.to(device),labels.to(device)
                    log_ps = model.forward(images)
                    losses = criterion(log_ps,labels)
                    valid_losses += losses.item()

                    #test the accuracy
                    ps = torch.exp(log_ps)
                    top_ps,top_class = ps.topk(1,dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
            print("Epoch {}/{}".format(e+1,epochs),
                  "Training loss: {:.3f}".format(running_loss/print_all),
                  "Validation loss: {:.3f}".format(valid_losses/len(validloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
            running_loss = 0
            model.train()
    return model


def main():
    print("Creating a flower classifier deep learning model")
    global args
    args = parse()
    data = get_dir()
    model = build_network(data)
    model = train(model,data)
    train_datasets = data['train_data']
    print('Saving model!!!!!!')
    if args.save_dir is None:
        save_dir = 'checkpoint.pth'
    else:
        save_dir = args.save_dir
    
    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {
                   'features': model.features,
                   'classifier': model.classifier,
                   'class_to_idx': model.class_to_idx,
                   'state_dict': model.state_dict(),
                   'arch' : args.arch
                 }
    torch.save(checkpoint,save_dir)
    print('Saving model was successful')

main()
    
        