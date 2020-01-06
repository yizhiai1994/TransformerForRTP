import numpy as np
import torch
import torch.nn as nn
import math, copy, time
from torch.autograd import Variable
import seaborn
seaborn.set_context(context="talk")
from harvard_MultiGPU.MultiHeadedAttention import MultiHeadedAttention
from harvard_MultiGPU.PositionwiseFeedForward import PositionwiseFeedForward
from harvard_MultiGPU.PositionalEncoding import PositionalEncoding
from harvard_MultiGPU.EncoderDecoder import EncoderDecoder
from harvard_MultiGPU.Encoder import Encoder
from harvard_MultiGPU.EncoderLayer import EncoderLayer
from harvard_MultiGPU.Decoder import Decoder
from harvard_MultiGPU.DecoderLayer import DecoderLayer
from harvard_MultiGPU.Embeddings import Embeddings
from harvard_MultiGPU.Generator import Generator
from harvard_MultiGPU.NoamOpt import NoamOpt
from harvard_MultiGPU.LabelSmoothing import LabelSmoothing
from harvard_MultiGPU.Batch import Batch
from harvard_MultiGPU.SimpleLossCompute import SimpleLossCompute
from harvard_MultiGPU.Tool import subsequent_mask
import os
from harvard_MultiGPU.TrainTool import make_model, run_epoch, data_gen, greedy_decode
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
global max_src_in_batch, max_tgt_in_batch
# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model = model.cuda()
criterion = criterion.cuda()
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model,
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model,
                    SimpleLossCompute(model.generator, criterion, None)))

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]).cuda() )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))









