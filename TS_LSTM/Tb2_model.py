#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:36:46 2023
SCRIPT: G_B121_LSTM.py

Machine Learning Model with LSTM bidirectional - 241: hiddendim = 24 layer = 1

LSTM bidirectional + linear

History:
    A2: Accuracy calculation

    
@author: Giusepe Buono
"""
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.data import DataLoader
# from torch.utils.data import random_split
# from torch.utils.data import TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class SoftmaxModel():
    """
    Solver
    """
    def __init__(self, hidden_size, hidden_layers = 1, output_size = 1, lr = 0.001):
        # Define Model
        self.input_size = 1
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.bidirectional = True
        self.model = self.Model(self.input_size, self.hidden_size, self.output_size, self.hidden_layers, self.bidirectional)
        self.model = self.model.to(device)
        self.lr1 = lr
        self.lr2 = lr
        # Define loss function
        self.loss_r = nn.MSELoss()
        self.loss_c = nn.CrossEntropyLoss()
        # Define Optimization Function
        self.optimizer1 = optim.Adam(self.model.parameters(), self.lr1, weight_decay=0.0001)
        self.optimizer2 = optim.Adam(self.model.parameters(), self.lr2, weight_decay=0.0001)
        
        # Define the scheduler of learning rate
        self.scheduler1 = ReduceLROnPlateau(self.optimizer1, mode='min', patience=4, factor=0.5)
        self.scheduler2 = ReduceLROnPlateau(self.optimizer2, mode='min', patience=4, factor=0.5)
        
    class Model(nn.Module):
        """
        input_size - will be 1 in this example since we have only 1 predictor (a sequence of previous values)
        hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
        output_size - This will be equal to the prediciton_periods input to get_x_y_pairs
        """  
        def __init__(self, input_size, hidden_size, output_size, hidden_layers, bidirectional = True):
            super().__init__()
            # print(seq_len, n_features, embedding_dim, hidden_layers, bidirectional)
            self.regression = self.Regression(input_size, hidden_size, 1, hidden_layers, bidir = bidirectional).to(device)
            # input size = 2 if input = w + kpr
            self.classification = self.Classification(input_size*2, hidden_size, output_size, hidden_layers, bidir = bidirectional).to(device)
            # self.classification = self.Classification(input_size, hidden_size, output_size, hidden_layers, bidir = bidirectional).to(device)
        
        def forward(self, w):
            up = self.regression(w)
            # print(up.size(), w.size())
            kpr = w - up
            # print(kpr.size())
            t_input2 = torch.cat((w, up), dim=2)
            kdp = self.classification(t_input2)
            return up, kdp
        
        class Regression(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, num_layers, bidir = True):
                super().__init__()
                self.hidden_layers = num_layers
                self.hidden_dim = hidden_size
                self.bidirectional = bidir
                self.bid_factor = 1 # to manage layer and hidden number if bidirectional = True
                if self.bidirectional == True:
                    self.bid_factor = 2
                self.reset_hidden_lay = self.bid_factor * self.hidden_layers
                    
                # batch_first=True causes input/output tensors to be of shape
                # (batch_dim, seq_dim, feature_dim) 
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional = bidir, batch_first=True)          
                self.linear = nn.Linear(self.bid_factor * self.hidden_dim, output_size)

                
            def forward(self, x): 
                # Initialize hidden state with zeros
                h0 = torch.zeros(self.reset_hidden_lay, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
                # # Initialize cell state
                c0 = torch.zeros(self.reset_hidden_lay, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
                out = x
                out, (hidden, cell) = self.lstm(out, (h0.detach(), c0.detach()))
                out = self.linear(out)
                return out

        class Classification(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, num_layers, bidir = True):
                super().__init__()
                self.hidden_layers = num_layers
                self.hidden_dim = hidden_size
                self.bidirectional = bidir
                self.bid_factor = 1 # to manage layer and hidden number if bidirectional = True
                if self.bidirectional == True:
                    self.bid_factor = 2
                self.reset_hidden_lay = self.bid_factor * self.hidden_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional = bidir, batch_first=True)          
                self.linear = nn.Linear(self.bid_factor * self.hidden_dim, output_size)
                
            def forward(self, x):               
                # Initialize hidden state with zeros
                h0 = torch.zeros(self.reset_hidden_lay, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
                # # Initialize cell state
                c0 = torch.zeros(self.reset_hidden_lay, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
                # We need to detach as we are doing truncated backpropagation through time (BPTT)
                # If we don't, we'll backprop all the way to the start even after going through another batch
                
                out = x
                out, (hidden, cell) = self.lstm(out, (h0.detach(), c0.detach()))
                out = self.linear(out)
                return out
 
    # Change model parameter
    def set_model(self, input_size, hidden_size, output_size, num_layers, bidirectional):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layers = num_layers
        self.bidirectional = True
        self.model = self.Model(input_size, hidden_size, output_size, num_layers, bidirectional)
        self.model = self.model.to(device)
        return self.model

    # Calculate accuracy (a classification metric)
    def accuracy_fn(self, y_true, y_pred):
        # print("y_true ********",y_true)
        # print("y_pred ********",y_pred)
        correct = torch.eq(y_true, y_pred).float().sum().item() # torch.eq() calculates where two tensors are equal
        acc = (correct /(y_pred.size()[1]) )*100
        # print(acc)
        return acc 
    
   
    ### Training
    def train(self, train_dl):
      
        self.model.train()
        epoch_loss_u = 0
        epoch_loss_kd = 0
        epoch_acc = 0
        
        for wrapped_batch, unwrapped_batch, label_batch, lengths in train_dl:
            # if list(timeseries_batch.size()) != [8, 300, 1]:
            #     print("timeseries", timeseries_batch.size())
                
            # 1. Forward pass (model outputs raw logits)
            up, kdp = self.model(wrapped_batch) # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
            
            # 2. Calculate loss/accuracy
            
            # CrossEntropy assumes the feature dim is always the second dimension 
            # permutation is needed
            # print(y_logits.permute(0,2,1).size())
            # print(label_batch.size())
            # print(label_batch.permute(0,2,1).squeeze(dim = 2).long().size())
            loss_u = self.loss_r(up, unwrapped_batch.float()) 
            loss_kd = self.loss_c(kdp.permute(0,2,1).double(), label_batch.permute(0,2,1).squeeze(dim = 1).long())
            
            y_pred = kdp.detach().double()
            y_true = label_batch.squeeze(dim=2).detach().long()

            acc = self.accuracy_fn(y_true = y_true, y_pred = y_pred.argmax(dim = 2))
            # print(y_pred.argmax(dim = 2))                       
            # 3. Optimizer zero grad
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            
            # 4. Loss backwards
            loss = loss_u + loss_kd
            loss.backward()
    
            # 5. Optimizer step
            self.optimizer1.step()
            self.optimizer2.step()
            
            # a = label_batch.argmax(dim = 2)
            # b = y_logits.argmax(dim = 2)
            
            # # # cc = a*max(timeseries_batch.numpy()[0,:,0])
            # fig1 = plt.plot(a.squeeze(), 'o', label = "kdTrue")
            # plt.show()
            # fig2 = plt.plot(b.squeeze(), 'rx', label = "kdP")
            # # fig5 = plt.plot(y_logits.detach().numpy())
        
            # # # fig2 = plt.plot(label_batch.numpy()[0,:,0])
            # # fig4 = plt.plot(timeseries_batch.numpy()[0,:,0])
            # plt.show()

            epoch_loss_u += loss_u.item()
            epoch_loss_kd += loss_kd.item()
            epoch_acc += acc

        return epoch_loss_u/len(train_dl.dataset), epoch_loss_kd/len(train_dl.dataset), epoch_acc/len(train_dl.dataset), self.optimizer1.state_dict 

    def evaluate(self, valid_dl): 
        self.model.eval()
        epoch_val_loss_u = 0
        epoch_val_loss_kd = 0
        epoch_val_acc = 0
        epoch_val_low_acc = 0
        worst_acc = 100
        k = 0
        with torch.inference_mode():
            for wrapped_batch, unwrapped_batch, label_batch, lengths in valid_dl:
                
                up, kdp = self.model(wrapped_batch) # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
                
                y_pred = kdp.detach().double()
                y_true = label_batch.squeeze().detach().long()
                loss_v_u = self.loss_r(up, unwrapped_batch.float()) 
                loss_v_kd = self.loss_c(kdp.permute(0,2,1).double(), label_batch.permute(0,2,1).squeeze(dim = 1).long())

                acc_v = self.accuracy_fn(y_true = y_true, y_pred = y_pred.argmax(dim = 2))

                if (acc_v/y_pred.size()[0]) < worst_acc:
                    worst_acc = acc_v/y_pred.size()[0]
                    low_acc = k
                    print(worst_acc, low_acc)
                    
                    
                if (acc_v/y_pred.size()[0]) < 96:
                    epoch_val_low_acc += 1
                    

                # if worst_acc/y_pred.size()[0] == 12.5:
                #     # # cc = a*max(timeseries_batch.numpy()[0,:,0])
                #     fig1 = plt.plot(timeseries_batch.squeeze()[0].cpu(), 'o', label = "kdTrue")
                #     plt.show()
                #     # fig2 = plt.plot(b.squeeze(), 'rx', label = "kdP")
                #     # fig5 = plt.plot(y_logits.detach().numpy())
               
                #     # # fig2 = plt.plot(label_batch.numpy()[0,:,0])
                #     # fig4 = plt.plot(timeseries_batch.numpy()[0,:,0])
                #     # plt.show() 
                k += 1
                epoch_val_loss_u  += loss_v_u.item()
                epoch_val_loss_kd += loss_v_kd.item()
                epoch_val_acc += acc_v 
                
            # # Aggiorna lo scheduler sulla base della loss
            self.scheduler1.step(epoch_val_loss_u/len(valid_dl.dataset))   
            self.scheduler2.step(epoch_val_loss_kd/len(valid_dl.dataset)) 
            print("lr = ", self.scheduler1.get_last_lr())
            # self.scheduler1.print_lr(is_verbose, group, lr, epoch=None)
            print(epoch_val_low_acc)
        return epoch_val_loss_u/len(valid_dl.dataset), epoch_val_loss_kd/len(valid_dl.dataset), epoch_val_acc/len(valid_dl.dataset), worst_acc

