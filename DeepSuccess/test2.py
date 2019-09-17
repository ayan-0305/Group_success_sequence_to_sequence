from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import random
import math
import time
import pickle

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import re
import itertools
import math
import os

from data import e_get_sample, loadfiles, getlist
from model import EncoderRNN, LuongAttnDecoderRNN, DecoderRNN
from utils import trainItersCombined, eval_data, pAccuracy, pAccuracy_bin

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
MAX_INP_LEN = 5
MAX_LENGTH = 10


def data_loader( gs, create_new, save_new ):
    data_dir = "./saved_dataset/"
    data_name = "data_gs_"+str(gs)
    metadata_name = "metadata_gs_"+str(gs)

    file_dir = os.path.join(data_dir, data_name)
    mfile_dir = os.path.join(data_dir, metadata_name)

    if create_new:
        print("Creating new dataset using data files....")
        lab, metadata = getlist(gs)
        if save_new:
            print("Saving the new dataset...")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            with open(str(file_dir)+'.pkl','wb') as f1:
                pickle.dump(lab,f1)
            with open(str(mfile_dir)+'.pkl','wb') as f2:
                pickle.dump(metadata,f2)
    else:
        print("Using previously saved datasets...")
        lab = pickle.load(open(str(file_dir)+'.pkl','rb'))
        metadata = pickle.load(open(str(mfile_dir)+'.pkl','rb'))

    grp,feat,tag,tar,grw,siz,cit,grw5=e_get_sample(lab, len(lab),0)
    trr = int(0.8*float(len(lab)))
    data = []
    for i in range(len(lab)):
        data.append([feat[i], grw5[i], tar[i], grp[i], cit[i], tag[i]])

    return data, trr, metadata


gs = 1 # 1 : group_size ,  0 : event_attendance
create_new = 0
save_new = 0

data, trr, metadata = data_loader(gs, create_new, save_new)

train_list = data[:trr]
test_list = data[trr:]
print("size of train data : ", len(train_list))
# print("some sample ======================")
# print(train_list[:2])


gs = 0 # 1 : group_size ,  0 : event_attendance
create_new = 0
save_new = 0

data, trr, metadata2 = data_loader(gs, create_new, save_new)
train_list2 = data[:trr]
test_list2 = data[trr:]
print("size of train data 2 : ", len(train_list2))
# print("some sample ======================")
# print("train_list --- ",train_list[:2])
# print("train_list --- ",train_list2[:2])

#normalizing data sizes
minsize = min(len(train_list),len(train_list2))
if(len(train_list)>minsize):
    train_list = train_list[:minsize]
if(len(train_list2)>minsize):
    train_list2 = train_list2[:minsize]

print("new size of train data 1 : ", len(train_list))
print("new size of train data 2 : ", len(train_list2))

for i, d in enumerate(train_list):
    if(train_list[i][2] != train_list2[i][2]):
        print("loaded")

# # Configure models
# model_name = 's2s_model_pytorch'
# attn_model = 'dot'
# #attn_model = 'general'
# #attn_model = 'concat'
# input_dim = 33
# output_dim = 1
# hidden_size = 50
# encoder_n_layers = 2
# decoder_n_layers = 2
# dropout = 0.1
# batch_size = 64
# max_target_len = 5

# # Set checkpoint to load from; set to None if starting from scratch
# loadFilename = None
# checkpoint_iter = 200

# # Configure training/optimization
# clip = 50.0
# teacher_forcing_ratio = 1
# learning_rate = 0.0001
# decoder_learning_ratio = 5.0
# n_iteration = 100
# print_every = 1
# save_every = 500
# save_dir = "./saved_models/"

# # loadFilename = os.path.join(save_dir, model_name,
# #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
# #                            '{}_checkpoint.tar'.format(checkpoint_iter))

# # Load model if a loadFilename is provided
# if loadFilename:
#     # If loading on same machine the model was trained on
#     checkpoint = torch.load(loadFilename)
#     # If loading a model trained on GPU to CPU
#     #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
#     encoder_sd = checkpoint['en']
#     decoder_sd = checkpoint['de']
#     encoder_optimizer_sd = checkpoint['en_opt']
#     decoder_optimizer_sd = checkpoint['de_opt']


# print('Building encoder and decoder ...')
# # Initialize encoder & decoder models
# encoder = EncoderRNN(input_dim, hidden_size, encoder_n_layers, dropout, bidirection=0)
# decoder = DecoderRNN(attn_model, hidden_size, output_dim, decoder_n_layers, dropout)

# encoder2 = EncoderRNN(input_dim, hidden_size, encoder_n_layers, dropout, bidirection=0)
# decoder2 = DecoderRNN(attn_model, hidden_size, output_dim, decoder_n_layers, dropout)


# if loadFilename:
#     encoder.load_state_dict(encoder_sd)     
#     decoder.load_state_dict(decoder_sd)
# # Use appropriate device
# try:
#     encoder = encoder.to(device)
# except RuntimeError as e:
#     print(e)
#     device = "cpu"
#     print("---------Device switched to CPU----------")

# encoder = encoder.to(device)
# decoder = decoder.to(device)
# encoder2 = encoder2.to(device)
# decoder2 = decoder2.to(device)
# print('Models built and ready to go!')

# # Ensure dropout layers are in train mode
# encoder.train()
# decoder.train()
# encoder2.train()
# decoder2.train()

# # Initialize optimizers
# print('Building optimizers ...')
# encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
# decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

# encoder_optimizer2 = optim.Adam(encoder2.parameters(), lr=learning_rate)
# decoder_optimizer2 = optim.Adam(decoder2.parameters(), lr=learning_rate * decoder_learning_ratio)

# if loadFilename:
#     encoder_optimizer.load_state_dict(encoder_optimizer_sd)
#     decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# # Run training iterations
# print("Starting Training!")
# trainItersCombined(train_list, train_list2, metadata, metadata2, model_name, hidden_size, 
#             encoder, decoder, encoder2, decoder2, encoder_optimizer,
#             decoder_optimizer, encoder_optimizer2, decoder_optimizer2,
#            encoder_n_layers, decoder_n_layers, max_target_len, save_dir, n_iteration, batch_size,
#            print_every, save_every, clip, teacher_forcing_ratio, loadFilename, device)



# print("Starting Testing!....")
# # Ensure dropout layers are in test mode
# encoder.eval()
# decoder.eval()
# encoder2.eval()
# decoder2.eval()
# others, f_out, f_tar = eval_data(test_list, metadata, encoder, decoder, encoder_n_layers, decoder_n_layers, max_target_len, batch_size, device)
# others2, f_out2, f_tar2 = eval_data(test_list2, metadata2, encoder2, decoder2, encoder_n_layers, decoder_n_layers, max_target_len, batch_size, device)

# print(f_out[:2])
# print(" --- ")
# print(f_out2[:2])
# print(others[3][:2])

# bint = np.linspace(0,1,6)
# real_out = []
# for i, d in enumerate(f_out):
#     tfa = []
#     tfb = []
#     for j in range(5):
#         onea = [0,0,0,0,0]
#         oneb = [0 for _ in range(5)]

#         for n in range(0, len(bint)-1):
#             b_start = bint[n]
#             b_end = bint[n+1]
#             if float(f_out[i][j])>=b_start and float(f_out[i][j])<b_end:
#                 onea[n]=1
#             if float(f_out2[i][j])>=b_start and float(f_out2[i][j])<b_end:
#                 oneb[n]=1

#         tfa.extend(onea)
#         tfb.extend(oneb)
#     tfa.extend(tfb)
#     tfa.extend(int(d) for d in others[3][i])
#     real_out.append(tfa)

# print(real_out[:2])
        
