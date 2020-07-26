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
    parser = argparse.ArgumentParser(description='PREDICTION!!!!')
    parser.add_argument('--image_dir' ,help='directory to the image of the test flower')
    parser.add_argument('--check_point',default = 'checkpoint.pth' ,help='directory to checkpoint')
    parser.add_argument('--top_k', default = 5 ,help='return top most likely classes')
    parser.add_argument('--category_names' ,help='mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='whether gpu or cpu')
    args = parser.parse_args()
    return args

def load_checkpoint(path = 'checkpoint.pth'):
    
    model_desc = torch.load(path,map_location=lambda storage, loc: storage)
    model = models.vgg16(pretrained=True)
    model.classifier = model_desc['classifier']
    model.class_to_idx = model_desc['class_to_idx']
    model.load_state_dict(model_desc['state_dict'])
    return model

   
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
   '''
           
    pro_image = Image.open(image_path).convert("RGB")
    
  
    pil_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    pro_image = pil_transforms(pro_image)
    return pro_image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # DONE: Implement the code to predict the class from an image file
    
    
    image = process_image(image_path)
    image.unsqueeze_(0)
   
    with torch.no_grad():
        outputs = model.forward(image)
        top_ps, top_labels = torch.topk(outputs, topk)
        top_ps = top_ps.exp()
        
    class_to_idxs = {model.class_to_idx[final]:final for final in model.class_to_idx}
    map_class = []
    
    for label in top_labels.numpy()[0]:
        map_class.append(class_to_idxs[label])
    
    probs = zip(top_ps.numpy()[0], map_class)
    return probs

def read_json():
    print('Reading json files')
    if args.category_names:
        cat_names = args.category_names
    else:
        cat_names = 'cat_to_name.json'
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name

def display(probs):
    print('Display classes of flowers')
    json_file = read_json()
    steps = 0
    for a,b in probs:
        steps+=1
        a = str(round(a,3) * 100)
        b = json_file.get(str(b),'None')
       
        print('{} class {}................. {}%'.format(steps,b,a))
     

def main():
    print('Final Lap')
    global args
    args = parse()
    top_k = args.top_k
    if args.image_dir:
        image_dir = args.image_dir
    else:
        image_dir = 'flowers/valid/1/image_06739.jpg'
    path = args.check_point
    model = load_checkpoint(path)
    results = predict(image_dir,model)
    display(results)
    return results

main()
    
    
