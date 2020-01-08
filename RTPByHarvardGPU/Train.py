import numpy as np
import torch
import torch.nn as nn
import math, copy, time
from torch.autograd import Variable
import seaborn
import datetime
seaborn.set_context(context="talk")
from RTPByHarvardGPU.MultiHeadedAttention import MultiHeadedAttention
from RTPByHarvardGPU.PositionwiseFeedForward import PositionwiseFeedForward
from RTPByHarvardGPU.PositionalEncoding import PositionalEncoding
from RTPByHarvardGPU.EncoderDecoder import EncoderDecoder
from RTPByHarvardGPU.Encoder import Encoder
from RTPByHarvardGPU.EncoderLayer import EncoderLayer
from RTPByHarvardGPU.Decoder import Decoder
from RTPByHarvardGPU.DecoderLayer import DecoderLayer
from RTPByHarvardGPU.Embeddings import Embeddings
from RTPByHarvardGPU.Generator import Generator
from RTPByHarvardGPU.NoamOpt import NoamOpt
from RTPByHarvardGPU.LabelSmoothing import LabelSmoothing
from RTPByHarvardGPU.Batch import Batch
from RTPByHarvardGPU.SimpleLossCompute import SimpleLossCompute
from RTPByHarvardGPU.TempLossCompute import TempLossCompute
from RTPByHarvardGPU.Tool import subsequent_mask
from RTPByHarvardGPU.Data import Data
from sklearn.model_selection import train_test_split
strptime = datetime.datetime.strptime
import os
from RTPByHarvardGPU.TrainTool import make_model, run_epoch, greedy_decode,eval_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
global max_src_in_batch, max_tgt_in_batch
# Train the simple copy task.
#Build Data
Data_test = Data(data_address = '../Data/helpdesk.csv')
Data_test.build_trace(0,[1,2])
Data_test.build_vocab(1)
trace_list = [Data_test.trace_data[t] for t in Data_test.trace_data]
train_list, test_list = train_test_split(trace_list,train_size=0.7)
#Build Model
V = Data_test.vocab_size
# criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
criterion = nn.L1Loss().cuda()
model = make_model(V, V, N=2)
model = model.cuda()
criterion = criterion.cuda()
# model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
#         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
model_opt = torch.optim.Adam(model.parameters(), lr=0.1)
#Train Model
def data_gen(trace_list,event2id, padding_dix,batch_size):
    src_batch_temp = list()
    tgt_batch_temp = list()
    time = len(trace_list)
    #d = datetime.datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S')
    for trace in trace_list:
        #print(trace)
        src_temp = [event2id[record[0]] for record in trace]
        tgt_temp = [float((strptime(trace[-1][1], '%Y-%m-%d %H:%M:%S') - strptime(record[1], '%Y-%m-%d %H:%M:%S')).seconds/60/60/24)  for record in trace]
        #print(tgt_temp)
        src_batch_temp.append(src_temp.copy())
        tgt_batch_temp.append(tgt_temp.copy())
        if len(src_batch_temp) == batch_size:
            time = time - len(src_batch_temp)
            src_batch_length = [len(record) for record in src_batch_temp]
            src_batch_length = list(set(src_batch_length))
            src_batch_maxLength = max(src_batch_length)
            src_batch = list()
            for line in src_batch_temp:
                while len(line) < src_batch_maxLength:
                    line.append(padding_dix)
                src_batch.append(line.copy())
            for line in tgt_batch_temp:
                while len(line) < src_batch_maxLength:
                    line.append(padding_dix)
            src = torch.Tensor(src_batch_temp).cuda()
            tgt = torch.Tensor(tgt_batch_temp).cuda()
            src = Variable(src, requires_grad=False)
            tgt = Variable(tgt, requires_grad=False)
            #print(src.size())
            yield Batch(src, tgt, padding_dix)
            src_batch_temp = list()
            tgt_batch_temp = list()

for epoch in range(10):
    print(epoch)
    model.train()
    # run_epoch(data_gen(train_list, Data_test.event2id, batch_size=200, padding_dix=0), model,
    #           SimpleLossCompute(model.generator, criterion, model_opt))
    run_epoch(data_gen(train_list, Data_test.event2id, batch_size=20, padding_dix=0), model,
              TempLossCompute(criterion, model_opt))

    model.eval()

    print(run_epoch(data_gen(test_list, Data_test.event2id, batch_size=100, padding_dix=0), model,
                    TempLossCompute(criterion, None)))

model.eval()
print(eval_model(data_gen(test_list, Data_test.event2id, batch_size=500, padding_dix=0), model,
                    TempLossCompute(criterion, None)))
# src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]).cuda() )
# src_mask = Variable(torch.ones(1, 1, 10) )
# print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))









