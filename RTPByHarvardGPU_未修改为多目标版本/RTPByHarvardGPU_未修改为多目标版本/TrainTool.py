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
from sklearn.model_selection import KFold
from RTPByHarvardGPU.Tool import subsequent_mask
from RTPByHarvardGPU.PredictLayer import PredictLayer
import os
def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = PredictLayer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def result_eval(out, tgt, src, pad = 0, pos = 'last'):
    src_mask = (src != pad).unsqueeze(-2)
    src_mask = src_mask.contiguous().view(-1, 1)
    out = out.contiguous().view(-1, 1)
    tgt = tgt.contiguous().view(-1, 1)
    pad_error_list = list()
    no_pad_error_list = list()
    if pos == 'all':
        for num in range(len(src_mask)):
            if src_mask[num].item() != False:
                pad_error_list.append(float(out[num].item() - tgt[num].item()))
            no_pad_error_list.append(float(out[num].item() - tgt[num].item()))
    else:
        for num in range(len(src_mask)):
            if src_mask[num].item() != False and src_mask[num + 1].item() == False:
                pad_error_list.append(float(out[num].item() - tgt[num].item()))
            no_pad_error_list.append(float(out[num].item() - tgt[num].item()))
    pad_squaredError_list = [val * val for val in pad_error_list]
    no_pad_squaredError_list = [val * val for val in no_pad_error_list]
    pad_absError_list = [abs(val) for val in pad_error_list]
    no_pad_absError_list = [abs(val) for val in no_pad_error_list]
    pad_mse = sum(pad_squaredError_list) / len(pad_squaredError_list)
    no_pad_mse = sum(no_pad_squaredError_list) / len(no_pad_squaredError_list)
    pad_rmse = math.sqrt(sum(pad_squaredError_list) / len(pad_squaredError_list))
    no_pad_rmse = math.sqrt(sum(no_pad_squaredError_list) / len(no_pad_squaredError_list))
    pad_mae = sum(pad_absError_list) / len(pad_absError_list)
    no_pad_mae = sum(no_pad_absError_list) / len(no_pad_absError_list)
    #print(len(pad_squaredError_list),len(no_pad_squaredError_list))
    #print(pad_mae,no_pad_mae)
    return (pad_mse, no_pad_mse, pad_rmse, no_pad_rmse, pad_mae, no_pad_mae)
def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.src_mask)

        #print(out.contiguous().view(-1, out.size(-1)).size())
        loss = loss_compute(out, batch.trg, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

def build_k_fold_data(k,data):
    k_ford_Data = list()
    kf = KFold(n_splits=k).split(data)

    #print(len(data))
    for train, test in kf:
        train_data = [data[temp] for temp in train]
        test_data = [data[temp] for temp in test]
        k_ford_Data.append((train_data.copy(),test_data.copy()))
        #print('train',len(train),train[])
        #print('test',len(test),test)
    #print(k_ford_Data)
    return k_ford_Data
def eval_model(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    flag = 0
    for i, batch in enumerate(data_iter):

        out = model.forward(batch.src, batch.src_mask)
        loss = loss_compute(out, batch.trg, batch.ntokens)
        result_eval(out, batch.trg, batch.src, pad=0)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if flag == 0:
            src_list = batch.src.contiguous().view(-1, 1)
            tgt_list = batch.trg.contiguous().view(-1, 1)
            out_list = out.contiguous().view(-1, 1)
            flag = flag + 1
        else:
            src_list = torch.cat((src_list,batch.src.contiguous().view(-1, 1)),0)
            tgt_list = torch.cat((tgt_list,batch.trg.contiguous().view(-1, 1)),0)
            out_list = torch.cat((out_list,out.contiguous().view(-1, 1)),0)
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    eval_data = result_eval(out_list, tgt_list, src_list, pad=0)
    return total_loss, total_tokens, eval_data

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
# # Three settings of the lrate hyperparameters.
# opts = [NoamOpt(512, 1, 4000, None),
#         NoamOpt(512, 1, 8000, None),
#         NoamOpt(256, 1, 4000, None)]
# plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
# plt.legend(["512:4000", "512:8000", "256:4000"])
# None

def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10))).cuda()
        #data[:, 0] = 1
        print(data.size())
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        #print(batch,nbatches,tgt.size())
        yield Batch(src, tgt, 0)

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1).cuda()
    return ys