#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:36:46 2023
SCRIPT: G_B121_WU_train.py
    
direct unwrapping function (from wrapped to unwrapped) - LSTM with single layer
History:
    B1: hidden = 12 layers = 1
    
@author: Giuseppe Buono
"""
from Tb2_model import SoftmaxModel

import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset

if torch.cuda.is_available():
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
     
    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
           
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"  

# print(torch.distributed.is_available())
# print(torch.distributed.is_gloo_available())
# print(torch.distributed.is_initialized())
# print(torch.multiprocessing.get_all_sharing_strategies())
# # print(torch.distributed.get_rank(group=None))
# torch.multiprocessing.set_sharing_strategy('file_system')

# torch.distributed.init_process_group(backend='gloo', world_size=4, rank=0)
# torch.set_num_interop_threads(8) # Inter-op parallelism
# torch.set_num_threads(8) # Intra-op parallelism    

starttime_global = (datetime.now())

# ********* FILE SETTINGS *********************
job_name = "Mb_train"
output_dir = "output"
file_name = "MIX_limit_2.pkl"
class_number = 32 # usa numero pari
sample_number = 120000 # 80000
# number of epoch is calculated from the following parameter
n_min_iters = 250 # in case of sample_number = 48000 / 80000
BATCHSIZE = 8
differential = True
hidden_dim = 96
lstm_layers = 2
lr = 0.0001
accuracy_save_limit = 99.6
notes = "Softmax, CrosseEntropy loss, Adam optim"

# the following parameter is fixed indipendently from sample size
max_iteration_number = 120000 

# epoch_threshold = 20  # if epoch > epoch_threshold then save model_file

if not os.path.isdir(output_dir):
   os.mkdir(output_dir)

job_dir = os.getcwd()
os.chdir('..')
file_dir = os.getcwd() + "/PS_collections/"
os.chdir(job_dir)


# ********* DATA PREPARATION *********************
lam = 55.466
norm_factor = lam / 2.0


training_parameters_df = pd.DataFrame(columns =["python_file", "data_file",  "sample_number", "differential", 
                                   "hidden_dim", "batchsize", "epochs_number", "notes"] )

epochs_df = pd.DataFrame(columns =["data_file", "epochs_number", "tloss", "taccuracy", 
                                   "vloss", "vaccuracy","epoch time", "file time", "model_file"] )

    
print("\n\n********************************************************")
print("|Python file: \t\t", os.path.basename(__file__))
print("|Train Data File: \t", file_name)
print("|Samples number: \t", sample_number)
print("|Batchsize: \t\t\t", BATCHSIZE)
print("|Differential: \t\t", differential)
print("|Hidden Dimension: \t", hidden_dim)
print("|LSTM layers: \t\t", lstm_layers)
print("|lr: \t\t\t\t", lr)


starttime_file = (datetime.now())
   
# Open ps_collection
ps_data = pd.read_pickle(file_dir + '/' + file_name)

# x and y arrays for LSTM
# any row is a ts (Univariate Time Series)
w = ps_data['x'][0]
u = ps_data['y'][0]
kd = ps_data['kd'][0]
l = ps_data['l'][0]

# if differential == True:
#     # If I want to use differential input data: difference and insert 0 column
#     w_diff = np.insert(np.diff(w), 0, 0, axis = 1)  

# x = np.exp(x)

# Class definition and label assign
output_dim = class_number
kd_offset = int(output_dim / 2)
# kd_offset = 1
kd = np.int32(kd) + kd_offset
# print(kd_offset)
# print(np.min(kd), np.max(kd))

# It's essential to have the input vector with the shape: 
# number_of_timeseries, timeseries length, number_of_features
w = np.reshape(w, (w.shape[0], w.shape[1], 1)) 
# w_diff = np.reshape(w_diff, (w_diff.shape[0], w_diff.shape[1], 1))
u = np.reshape(u, (u.shape[0], u.shape[1], 1))  
# for y number of timeseries, timeseries length
kd = np.reshape(kd, (kd.shape[0], kd.shape[1], 1)) 


# Concatenate x and x_diff along the last dimension
# input_data = np.concatenate((w, w_diff), axis=2)
 
# Trasform to Tensor
t_w = torch.tensor(w).to(device)
# t_xdiff = torch.from_numpy(x_diff).to(device)
t_u = torch.tensor(u).to(device)
t_kd = torch.from_numpy(kd).to(device)
t_l = torch.from_numpy(l).to(device)

torch.manual_seed(1)
joint_dataset = TensorDataset(t_w, t_u, t_kd, t_l)
sample_size =  sample_number / max_iteration_number
# print("sample_size", sample_size)
# Proportion: training 0.6, validation 0.3
train_idx, val_idx, test_idx = random_split(joint_dataset, [0.8 * sample_size, 0.2   * sample_size, (1-sample_size)], 
                                        generator=torch.Generator().manual_seed(42))
# print(val_idx[2544][1])
# print(len(val_idx))
# to save test data
# test_file = output_dir + "/" + "test_" + file_name + ".test"
# torch.save(test_idx, test_file)

batch_size = BATCHSIZE
train_dl = DataLoader(train_idx, batch_size, shuffle=True)
valid_dl = DataLoader(val_idx, batch_size, shuffle=True)
# test_dl = DataLoader(test_idx, batch_size, shuffle=True)

# Prepare data Train and Test
# Set the number of epochs
# num_epochs = int(n_iters / (len(t_x) / batch_size))
num_epochs = int(n_min_iters * max_iteration_number / (len(t_l) * sample_size ))
print("|Epochs number: \t\t", num_epochs)
print("|\n|Notes: ", notes)
print("|\n********************************************************\n\n")

training_parameters_df.loc[0] = [os.path.basename(__file__), file_name,  sample_number, differential,
                                                    hidden_dim, batch_size, num_epochs, notes]
training_parameters_df.to_pickle(output_dir + '/' + "training_parameters.df")
# num_epochs = 0

# ********    MODEL DEFINITION ********************************************

solver = SoftmaxModel(hidden_dim, lstm_layers, output_dim, lr)
# print(solver.model)
   
torch.manual_seed(42)
epoch_acc_max = 0

# Build training and evaluation loop
for epoch in range(num_epochs):
    
    startpartial1 = (datetime.now())
    model_file = output_dir + '/' + file_name + "{:04.0f}".format(epoch)
    
    ### Training
    epoch_average_loss_u, epoch_average_loss_kd, epoch_average_acc, optimizer_state = solver.train(train_dl)
        
    ### Testing
    epoch_average_val_loss_u, epoch_average_val_loss_kd, epoch_average_val_acc, worst_acc = solver.evaluate(valid_dl)

    if epoch % 1 == 0:
        print(f"|Epoch: {epoch}")
        print("|Total time: ", datetime.now() - starttime_global)
        print("|Epoch time: ", datetime.now() - startpartial1)
        print(f"|Train Loss:  {epoch_average_loss_u:.5f}, {epoch_average_loss_kd:.5f}, Accuracy: {epoch_average_acc:.2f}%")
        print(f"|Valid Loss:  {epoch_average_val_loss_u:.5f}, {epoch_average_val_loss_kd:.5f}, Accuracy: {epoch_average_val_acc:.2f}%")
        print(f"|Worst acc:   {worst_acc:.2f}%")

        print("------------------------------------------")

    endtime_epoch = datetime.now()
    # epochs_df.loc[len(epochs_df.index)] = [file_name, epoch, epoch_average_loss,
    #               epoch_average_acc, epoch_average_val_loss, epoch_average_val_acc, 
    #               endtime_epoch - startpartial1, endtime_epoch - starttime_file, model_file]
    
    ### Save model parameters
    # if ((epoch % (num_epochs/n_min_iters) == 0) and (epoch / (num_epochs/n_min_iters) > epoch_threshold)):
    #     torch.save(solver.model, model_file)
    #     print(model_file)
    
    # Save training parameters
    epochs_df.to_pickle(output_dir + '/' + file_name + ".df")
    ## Save model parameters
    if epoch_average_acc > accuracy_save_limit:
        model_file = model_file + "_" + "{:02.0f}".format(int(epoch_average_acc)) + ".pt"
        torch.save(solver.model, model_file)
        print("Accuracy threshold exeeded! ", "{:02.0f}".format(int(epoch_average_acc)))
    if epoch_average_val_acc > epoch_acc_max:
        epoch_acc_max = epoch_average_val_acc
        # model_file = model_file + "_" + "{:02.0f}".format(int(epoch_average_acc)) + ".pt"

        torch.save({
            'epoch': epoch,
            'model' : solver.model,
            'model_state_dict': solver.model.state_dict(),
            'optimizer_state_dict': optimizer_state,
            # Altri elementi da salvare se necessario
        }, output_dir + "/Max_acc_model_" + "{:04.0f}".format(epoch) + "_" + "{:04.2f}".format(epoch_average_val_acc) + ".pt")
        print("***********************************\nBest average accuracy! \t\t",  "{:04.2f}".format(epoch_average_val_acc) + '\n')


torch.save(solver.model, model_file)
print(model_file)
    
endtime = (datetime.now())
print("TOTAL TIME: ", endtime - starttime_global)



