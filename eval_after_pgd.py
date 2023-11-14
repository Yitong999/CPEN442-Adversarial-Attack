
from torch.utils.data import Dataset,DataLoader,random_split
from typing import Any,Tuple,Optional,Callable
import PIL
import csv
import pathlib
import torch
import torch.nn as nn
from torch.optim import Adam,lr_scheduler
from torchvision.transforms.v2 import ToTensor,Resize,Compose,ColorJitter,RandomRotation,AugMix,RandomCrop,GaussianBlur,RandomEqualize,RandomHorizontalFlip,RandomVerticalFlip
import matplotlib.pyplot as plt
import pickle
import tqdm


from data.attr_dataset import GTSRB
from module.util import get_model

from utils.options import args_parser


# Check if CUDA is available
use_cuda = torch.cuda.is_available()

# Check if MPS (Apple's Metal Performance Shaders) is available
use_mps = torch.backends.mps.is_available()

# Choose the device based on availability
if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

device = torch.device("cpu")


def pgd_attack_adv(device, model_b, model_d, images, labels, eps=0.4, alpha=4/255, lmd = 2, iters=40) :
    images = images.to(device)
    labels = labels.to(device)

    loss = nn.CrossEntropyLoss(reduction='none')
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs_b = model_b(images)
        outputs_d = model_d(images)
        # outputs1 = model(images)
        # output2
        model_b.zero_grad()
        model_d.zero_grad()

        cost_b = loss(outputs_b, labels).to(device)
        

        cost_d = loss(outputs_d, labels).to(device)

        cost = (cost_b - lmd * cost_d).mean()
        
        
        cost.backward()
        # cost 1
        # cost 2
        # cost1-lambda*cost2
        

        adv_images = images + alpha*images.grad.sign()
        
        
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    
    mode = 1
    
    if mode == 0:
        print('###################')
        print('label: ', labels)
        print('ori predicted by biased model: ', torch.argmax(model_b(ori_images), dim=1))
        print('ori predicted by debiased model: ', torch.argmax(model_d(ori_images), dim=1))
        print('adv predicted by biased model: ', torch.argmax(model_b(images), dim=1))
        print('adv predicted by debiased model: ', torch.argmax(model_d(images), dim=1))
    else:
        return images
    
        
    return images


if __name__ == "__main__":
    args = args_parser()

    _root = args.root


    transforms = Compose([
        Resize([28,28]),
        ToTensor(),
    
    ])

    testdata = GTSRB(root=_root,split='test',transform=transforms)
    print('testing size :',len(testdata))
    test_dataloader = DataLoader(testdata)

    from sklearn.metrics import accuracy_score

    y_pred_1 = []
    y_true_1 = []

    y_pred_2 = []
    y_true_2 = []

    
    model_1 = torch.load('models/model_CNN.pth')
    model_2 = torch.load('models/model_MLP.pth')


    model_1 = model_1.eval().to(device)
    with tqdm.tqdm(colour='red',total=len(test_dataloader)) as progress:
    
        # with torch.no_grad() : 
        for id,(input,label) in enumerate(iter(test_dataloader)):
            input,label = input.to(device),label.to(device)


            input = pgd_attack_adv(device, model_1, model_2, input, label)

            #evaluate model_1
            y_true_1.append(label.item())
            prediction_1 = model_1.forward(input)
            _,prediction_1 = torch.max(prediction_1,1)
            y_pred_1.append(prediction_1.item())
            

            #evaluate model_2
            y_true_2.append(label.item())
            prediction_2 = model_2.forward(input)
            _,prediction_2 = torch.max(prediction_2,1)
            y_pred_2.append(prediction_2.item())


            progress.desc = f'Test Accuracy for model 1: {"{:.3f}".format(accuracy_score(y_true_1,y_pred_1))}; for model 2:  {"{:.3f}".format(accuracy_score(y_true_2,y_pred_2))}'
            progress.update(1)

            #evaluate model_2
            # y_true.append(label.item())
            # prediction = model_2.forward(input)
            # _,prediction = torch.max(prediction,1)
            # y_pred.append(prediction.item())
            
            # progress.desc = f'Test Accuracy : {accuracy_score(y_true,y_pred)} '
            # progress.update(1)

