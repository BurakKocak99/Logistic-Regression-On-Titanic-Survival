import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import torch
import torch.nn as nn
import copy
from termcolor import colored

def sigmoid(Q,x):
    return 1/(1+np.exp(np.dot(Q,x)))


# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

dftrain['sex'].replace('female',0,inplace = True)#Female 0
dftrain['sex'].replace('male',1,inplace = True)#Male 1
dfeval['sex'].replace('female',0,inplace = True)#Female 0
dfeval['sex'].replace('male',1,inplace = True)#Male 1

dftrain['class'].replace(['First','Second','Third'],[1,2,3],inplace = True)#First Second Third => 1 2 3
dfeval['class'].replace(['First','Second','Third'],[1,2,3],inplace = True)

dftrain['alone'].replace(['n','y'],[-1,1],inplace = True)
dfeval['alone'].replace(['n','y'],[-1,1],inplace = True)
dftrain.pop('deck')
dftrain.pop('embark_town')
dfeval.pop('deck')
dfeval.pop('embark_town')

t_train = torch.from_numpy(dftrain.values.astype(np.float32))
t_y_train = torch.from_numpy(y_train.values.astype(np.float32))
t_eval = torch.from_numpy(dfeval.values.astype(np.float32))
t_y_eval = torch.from_numpy(y_eval.values.astype(np.float32))
training = t_eval.numpy().astype(np.float32)








t_y_eval = torch.reshape(t_y_eval,(t_y_eval.shape[0],1))
t_y_train = torch.reshape(t_y_train,(t_y_train.shape[0],1))

out = t_y_eval.numpy().astype(np.float32)
for i in range(7):
    training = (training-training.mean(axis=0))/training.std(axis=0)

training = np.append(np.ones([training.shape[0],1]).astype(np.float32),training,axis = 1)

model = np.load('model.npy')
print(model)

Avr = 0
for input in range(training.shape[0]):
    est = sigmoid(model,training[input])

    print("est is ", est)
    if est.__round__() == out[input]:
        print(colored("Succesfull","green"))
        Avr += 1
    else:
        print(colored("Failed","red"))



print("Accuarcy is: ",(Avr/training.shape[0])*100,"%")