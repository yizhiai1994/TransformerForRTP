import numpy as np
import torch
import torch.nn as nn
import math, copy, time
from torch.autograd import Variable
import seaborn
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
from RTPByHarvardGPU.Tool import subsequent_mask
from RTPByHarvardGPU.Data import Data
import os
from RTPByHarvardGPU.TrainTool import make_model, run_epoch, greedy_decode
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
global max_src_in_batch, max_tgt_in_batch
# Train the simple copy task.
#Build Data
Data_test = Data(data_address = '../Data/helpdesk.csv')
Data_test.build_trace(0,[1,2])
Data_test.build_vocab(1)
#Build Model
V = Data_test.vocab_size
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model = model.cuda()
criterion = criterion.cuda()
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

#Train Model
def data_gen(trace_list,event2id, batch):
#def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for trace in trace_list:
    #for i in range(nbatches):
        #data = [event2id[record[0]]for record in trace_list[trace]]
        data = [[event2id[record[0]] for record in trace_list[trace]]]
        #data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10))).cuda()
        #data[:, 0] = 1
        #print(data)
        data = torch.Tensor(data).cuda()
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        #print(batch,nbatches,tgt.size())
        yield Batch(src, tgt, 0)

for epoch in range(10):
    model.train()
    print()
    run_epoch(data_gen(Data_test.trace_data, Data_test.event2id, 1), model,
              SimpleLossCompute(model.generator, criterion, model_opt))

    model.eval()

    print(run_epoch(data_gen(V, 30, 5), model,
                    SimpleLossCompute(model.generator, criterion, None)))

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]).cuda() )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))









