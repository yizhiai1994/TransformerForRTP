from DataOperate import DataOperate
import numpy as np
import torch
import torch.nn as nn
import math,copy, time
from torch.autograd import Variable
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import os
# ToolMethods
def data_gen(trace_list, padding_dix,batch_size):
    src_batch_temp = list()
    tgt_batch_temp = list()
    time = len(trace_list)
    #print(len(trace_list))
    for src,tgt in trace_list:
        src_temp = list()
        #src_temp = [src_temp + record for record in src]
        #src_temp = [(lambda x,y:x+y)(src_temp,record)for record in src]
        mult = lambda x,y:x.append(y)
        [[mult(src_temp,attribute) for attribute in record] for record in src]
        src_batch_temp.append(src_temp.copy())
        tgt_batch_temp.append(tgt.copy())
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
            #print(len(src_batch_temp[0]), len(tgt_batch_temp[0]))
            src = torch.Tensor(src_batch_temp).cuda()
            tgt = torch.Tensor(tgt_batch_temp).cuda()
            src = Variable(src, requires_grad=False)
            tgt = Variable(tgt, requires_grad=False)
            #print(src.size())

            yield Batch(src, tgt, padding_dix)
            src_batch_temp = list()
            tgt_batch_temp = list()
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    #print('query',query.size(),'key',key.size(),'value',value.size())
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    #print(mask.size())
    #print(scores.size())
    if mask is not None:
        mask = mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
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
# ToolClasses
class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        self.trg = trg
        self.ntokens = (self.src != pad).data.sum()
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        #print(size)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        #print(x.size(),self.dropout(sublayer(self.norm(x))).size(),sublayer(self.norm(x)).size(),self.norm(x).size())
        return x + self.dropout(sublayer(self.norm(x)))
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))
class TempLossCompute:
    "A simple loss compute and train function."

    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        # x = self.generator(x)
        # print(x.contiguous().view(-1, x.size(-1)).size(),y.contiguous().view(-1, x.size(-1)).size(),norm)
        #print(x.size(),y.size())
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1, x.size(-1))) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss.item() * norm
# Model
#    GlobalAttention
class GlobalMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(GlobalMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, global_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.global_attn = global_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, memory, src_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.global_attn(x, m, m, src_mask))
        return self.sublayer[1](x, self.feed_forward)
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask)
        return self.norm(x)
#    EmbeddingLayer
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000, attribute_length = 1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.attribute_length = attribute_length
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2 * attribute_length) *
                             -(math.log(10000.0) / d_model))
        #pe[:, 0::2] = torch.sin(position * div_term)
        #pe[:, 1::2] = torch.cos(position * div_term)
        for num in range(attribute_length):
            pe[:, num::attribute_length * 2] = torch.sin(position * div_term)
            pe[:, attribute_length + num::attribute_length * 2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = x.long()
        return self.lut(x) * math.sqrt(self.d_model)
#    EncoderLayer
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #print('2',self.w_2(self.dropout(F.relu(self.w_1(x)))).size(),self.dropout(F.relu(self.w_1(x))).size(),F.relu(self.w_1(x)).size(),self.w_1(x).size())
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
class AttributeMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(AttributeMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, d_atrbt, mask=None ):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(2)
            mask = mask.unsqueeze(2)
        nbatches = query.size(0)
        #print(query.size())
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1,d_atrbt, self.h, self.d_k).transpose(2, 3)
             for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        #print(query.size())
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        #print(x.size())
        # 3) "Concat" using a view and apply a final linear.
        #print(x.size())
        x = x.transpose(2, 3).contiguous() \
            .view(nbatches, -1, d_atrbt, self.h * self.d_k)
        #raise 'x'
        return self.linears[-1](x)
class SelfAttentionNetLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(SelfAttentionNetLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.d_model = size
    def forward(self, x, mask, d_atrbt):
        "Follow Figure 1 (left) for connections."
        #print(x.size())
        nbatches = x.size(0)
        x = x.view(nbatches, -1, d_atrbt, self.d_model)
        mask = mask.view(nbatches, -1, d_atrbt)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x,d_atrbt, mask))
        return self.sublayer[1](x, self.feed_forward)
class SelfAttentionNet(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(SelfAttentionNet, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, d_atrbt):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask, d_atrbt)
        return self.norm(x)
    # FincalLayer
class PredictLayer(nn.Module):
    """
      A standard Encoder-Decoder architecture. Base for this and many
      other models.
      """
    def __init__(self, global_attn, attribute_attn, atrbt_embed, pos_embd, dropout, d_atrbt, d_model):
        super(PredictLayer, self).__init__()
        self.global_attn = global_attn
        self.attribute_attn = attribute_attn
        self.dropout = nn.Dropout(p=dropout)
        self.atrbt_embed = atrbt_embed
        self.pos_embed = pos_embd
        self.d_atrbt = d_atrbt
        self.d_model = d_model
        self.src_embed = nn.Sequential(self.atrbt_embed, pos_embd)
        self.liner = nn.Linear(512, 1)

    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        nbatches = src.size(0)
        memory = self.attribute_attn(self.dropout(self.atrbt_embed(src)), src_mask, self.d_atrbt)
        memory = memory.view(nbatches, -1, self.d_model)
        return self.liner(self.global_attn(self.src_embed(src), self.pos_embed(memory), src_mask))
        #print(memory.size())
        #self.global_attn(self.src_embed(src),, src_mask, self.d_atrbt)

        # print(self.encode(src, src_mask).size())
        #print(self.attribute_attn(self.dropout(self.atrbt_embed(src)), src_mask, self.d_atrbt).size())
        #return self.attribute_attn(self.dropout(self.atrbt_embed(src)), src_mask, self.d_atrbt)
        # return self.attribute_attn(self.atrbt_embed(src), src_mask)

        # return self.liner(self.attribute_attn(self.src_embed(src), src_mask))
        #return self.liner(self.attribute_attn(self.src_embed(src), src_mask))
# TrainTool
def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1,attribute_length = 1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attribute_attn = AttributeMultiHeadedAttention(h, d_model)
    global_attn = GlobalMultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout,attribute_length=attribute_length)
    model = PredictLayer(Decoder(DecoderLayer(d_model, c(global_attn), c(ff), dropout), N),
        SelfAttentionNet(SelfAttentionNetLayer(d_model, c(attribute_attn), c(ff), dropout), N),
        Embeddings(d_model, src_vocab), c(position),dropout,attribute_length,d_model)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
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

# RecordTool

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

# Run
#    ParameterSetting
datafile = './Data/helpdesk.csv'
#time_type='%Y/%m/%d %H:%M:%S.000'
time_type='%Y-%m-%d %H:%M:%S'
trace_column = 0
event_column = 1
time_column = 2
attribute_columns = [1]
train_size=0.7
start_pos = 0
lr = 0.1
#   RecordSetting
result_folder = './result/'
record_folder = './result/record/'
model_save_folder = ''
dataset_name = 'helpdesk'
result_file = result_folder + dataset_name + '.csv'
k = 5
verify_mode = str(k) + '-fold' #k-fold or trani:test
train_to_test = str(k-1) + ':' + str(1)
criterion_mode = 'MSE'
optim_mode = 'Adam'
learning_rate = 0.1
target_mode='single'
#   Train
def train(datafile, time_type, trace_column, attribute_columns,
           criterion_mode, optim_mode, result_folder, target_mode, dataset_name,
          start_pos=0, train_size=0.3, epochs=1, lr=0.1, N=2, padding_dix=0,batch_size=500, k_fold=1 ):
    #   RecordSetting
    result_file = result_folder + dataset_name + '.csv'
    verify_mode = str(k) + '-fold'  # k-fold or trani:test
    train_to_test = str(k - 1) + ':' + str(1)
    result_file = result_folder + dataset_name + '.csv'
    if not os.path.isfile(result_file):
        result_file_open = open(result_file, 'w', encoding='utf-8')
        result_file_open.writelines(
            '数据集,验证方式,训练集:测试集,验证次数,单目标/多目标,损失函数,优化方法,学习率,训练轮数,MAE,MSE,RMSE,total_loss,total_tokens,total_loss/total_tokens\n')
    else:
        result_file_open = open(result_file, 'a', encoding='utf-8')

    # DataSetting
    if k_fold == 1:
        dataimpt = DataOperate(data_address=datafile, time_type=time_type)
        dataimpt.build_trace(trace_column, attribute_columns, time_column)
        dataimpt.build_vocab(attribute_columns)
        dataimpt.build_orginal_trace_list()
        dataimpt.build_processed_trace(start_pos=start_pos)
        trace_list = dataimpt.trace_list
        train_list, test_list = train_test_split(trace_list, train_size=train_size)
        train_list, test_list = dataimpt.build_train_test_trace(train_list, test_list, start_pos=start_pos)
    else:
        dataimpt = DataOperate(data_address=datafile, time_type=time_type)
        dataimpt.build_trace(trace_column, attribute_columns, time_column)
        dataimpt.build_vocab(attribute_columns)
        dataimpt.build_orginal_trace_list()
        dataimpt.build_processed_trace(start_pos=start_pos)
        trace_list = dataimpt.trace_list
        k_fold_data = build_k_fold_data(k, trace_list)

    #  ModelSetting
    V = dataimpt.vocab_size
    model = make_model(V, V, N=N, attribute_length=len(attribute_columns))
    model = model.cuda()

    # TrainSetting
    if optim_mode == 'Adam':
        model_opt = torch.optim.Adam(model.parameters(), lr=lr)
    if criterion_mode == 'MSE':
        criterion = nn.MSELoss().cuda()  # criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
        criterion = criterion.cuda()
    elif criterion_mode == 'L1loss':
        criterion = nn.L1Loss().cuda()
        criterion = criterion.cuda()
    # model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    if k_fold == 1:
        for epoch in range(epochs):
            print("epoch",epoch)
            model.train()
            run_epoch(data_gen(train_list, batch_size=batch_size, padding_dix=padding_dix), model,
                      TempLossCompute(criterion, model_opt))
            model.eval()
            print(run_epoch(data_gen(test_list, batch_size=batch_size, padding_dix=padding_dix), model,
                            TempLossCompute(criterion, None)))

        model.eval()
        print(padding_dix)
        total_loss, total_tokens, (pad_mse, no_pad_mse, pad_rmse, no_pad_rmse, pad_mae, no_pad_mae) = \
            eval_model(data_gen(test_list, dataimpt.event2id, padding_dix=padding_dix, batch_size=batch_size),
                       model, TempLossCompute(criterion, None))
        result_file_open.writelines(
            dataset_name + ',' + verify_mode + ',' + str(len(train_list)) + ':' + str(len(test_list)) + ',' + str(
                1) + ',' + target_mode + ',' +
            criterion_mode + ',' + optim_mode + ',' + str(learning_rate) + ',' + str(epoch) + ',' +
            str(pad_mae) + ',' + str(pad_mse) + ',' + str(pad_rmse) + ',' + str(total_loss.item()) + ',' + str(
                total_tokens.item()) + ',' +
            str(float(total_loss.item() / total_tokens.item())) + '\n')
    else:
        for train_data, test_data in k_fold_data:
            # print(train_data)
            print('fold', fold)
            train_list, test_list = dataimpt.build_train_test_trace(train_data, test_data, start_pos=start_pos)
            for epoch in range(10):
                print(epoch)
                model.train()
                run_epoch(data_gen(train_list, batch_size=batch_size, padding_dix=padding_dix), model,
                          TempLossCompute(criterion, model_opt))
                # model.eval()
                # print(run_epoch(data_gen(test_list, batch_size=batch_size, padding_dix=padding_dix), model,
                #                 TempLossCompute(criterion, None)))
            #
            model.eval()
            # print(eval_model(data_gen(test_list, Data_test.event2id, batch_size=500, padding_dix=0), model,
            #                     TempLossCompute(criterion, None)))
            total_loss, total_tokens, (pad_mse, no_pad_mse, pad_rmse, no_pad_rmse, pad_mae, no_pad_mae) = \
                eval_model(data_gen(test_list, dataimpt.event2id, batch_size=batch_size, padding_dix=padding_dix),
                           model, TempLossCompute(criterion, None))
            '数据集,验证方式,训练集:测试集,验证次数,单目标/多目标,损失函数,优化方法,学习率,训练轮数,MAE,MSE,RMSE,,,\n'
            result_file_open.writelines(
                dataset_name + ',' + verify_mode + ',' + str(len(train_data)) + ':' + str(len(test_data)) + ',' + str(
                    fold) + ',' + target_mode + ',' +
                criterion_mode + ',' + optim_mode + ',' + str(learning_rate) + ',' + str(epoch) + ',' +
                str(pad_mae) + ',' + str(pad_mse) + ',' + str(pad_rmse) + ',' + str(total_loss.item()) + ',' + str(
                    total_tokens.item()) + ',' +
                str(float(total_loss.item() / total_tokens.item())) + '\n')
            fold = fold + 1
        result_file_open.close()

# train(datafile, time_type, trace_column, attribute_columns,
#            criterion_mode, optim_mode, result_folder, target_mode, dataset_name,
#           start_pos=0, train_size=0.3, epochs=1, lr=0.1, N=2, padding_dix=0,batch_size=500, k_fold=1 )

start_pos=0
train_size=0.3
epochs=1
lr=0.1
N=2
padding_dix=0
batch_size=500
k_fold=1

result_file = result_folder + dataset_name + '.csv'
verify_mode = str(k) + '-fold'  # k-fold or trani:test
train_to_test = str(k - 1) + ':' + str(1)
result_file = result_folder + dataset_name + '.csv'
if not os.path.isfile(result_file):
    result_file_open = open(result_file, 'w', encoding='utf-8')
    result_file_open.writelines(
        '数据集,验证方式,训练集:测试集,验证次数,单目标/多目标,损失函数,优化方法,学习率,训练轮数,MAE,MSE,RMSE,total_loss,total_tokens,total_loss/total_tokens\n')
else:
    result_file_open = open(result_file, 'a', encoding='utf-8')

# DataSetting
if k_fold == 1:
    dataimpt = DataOperate(data_address=datafile, time_type=time_type)
    dataimpt.build_trace(trace_column, attribute_columns, time_column)
    dataimpt.build_vocab(attribute_columns)
    dataimpt.build_orginal_trace_list()
    dataimpt.build_processed_trace(start_pos=start_pos)
    trace_list = dataimpt.trace_list
    train_list, test_list = train_test_split(trace_list, train_size=train_size)
    train_list, test_list = dataimpt.build_train_test_trace(train_list, test_list, start_pos=start_pos)
else:
    dataimpt = DataOperate(data_address=datafile, time_type=time_type)
    dataimpt.build_trace(trace_column, attribute_columns, time_column)
    dataimpt.build_vocab(attribute_columns)
    dataimpt.build_orginal_trace_list()
    dataimpt.build_processed_trace(start_pos=start_pos)
    trace_list = dataimpt.trace_list
    k_fold_data = build_k_fold_data(k, trace_list)

#  ModelSetting
V = dataimpt.vocab_size
model = make_model(V, V, N=N, attribute_length=len(attribute_columns))
model = model.cuda()

# TrainSetting
if optim_mode == 'Adam':
    model_opt = torch.optim.Adam(model.parameters(), lr=lr)
if criterion_mode == 'MSE':
    criterion = nn.MSELoss().cuda()  # criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    criterion = criterion.cuda()
elif criterion_mode == 'L1loss':
    criterion = nn.L1Loss().cuda()
    criterion = criterion.cuda()
# model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
if k_fold == 1:
    for epoch in range(epochs):
        print("epoch",epoch)
        model.train()
        run_epoch(data_gen(train_list, batch_size=batch_size, padding_dix=padding_dix), model,
                  TempLossCompute(criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(test_list, batch_size=batch_size, padding_dix=padding_dix), model,
                        TempLossCompute(criterion, None)))

    model.eval()
    print(padding_dix)
    total_loss, total_tokens, (pad_mse, no_pad_mse, pad_rmse, no_pad_rmse, pad_mae, no_pad_mae) = \
        eval_model(data_gen(test_list, dataimpt.event2id, padding_dix=padding_dix, batch_size=batch_size),
                   model, TempLossCompute(criterion, None))
    result_file_open.writelines(
        dataset_name + ',' + verify_mode + ',' + str(len(train_list)) + ':' + str(len(test_list)) + ',' + str(
            1) + ',' + target_mode + ',' +
        criterion_mode + ',' + optim_mode + ',' + str(learning_rate) + ',' + str(epoch) + ',' +
        str(pad_mae) + ',' + str(pad_mse) + ',' + str(pad_rmse) + ',' + str(total_loss.item()) + ',' + str(
            total_tokens.item()) + ',' +
        str(float(total_loss.item() / total_tokens.item())) + '\n')
else:
    for train_data, test_data in k_fold_data:
        # print(train_data)
        print('fold', fold)
        train_list, test_list = dataimpt.build_train_test_trace(train_data, test_data, start_pos=start_pos)
        for epoch in range(10):
            print(epoch)
            model.train()
            run_epoch(data_gen(train_list, batch_size=batch_size, padding_dix=padding_dix), model,
                      TempLossCompute(criterion, model_opt))
            # model.eval()
            # print(run_epoch(data_gen(test_list, batch_size=batch_size, padding_dix=padding_dix), model,
            #                 TempLossCompute(criterion, None)))
        #
        model.eval()
        # print(eval_model(data_gen(test_list, Data_test.event2id, batch_size=500, padding_dix=0), model,
        #                     TempLossCompute(criterion, None)))
        total_loss, total_tokens, (pad_mse, no_pad_mse, pad_rmse, no_pad_rmse, pad_mae, no_pad_mae) = \
            eval_model(data_gen(test_list, dataimpt.event2id, batch_size=batch_size, padding_dix=padding_dix),
                       model, TempLossCompute(criterion, None))
        '数据集,验证方式,训练集:测试集,验证次数,单目标/多目标,损失函数,优化方法,学习率,训练轮数,MAE,MSE,RMSE,,,\n'
        result_file_open.writelines(
            dataset_name + ',' + verify_mode + ',' + str(len(train_data)) + ':' + str(len(test_data)) + ',' + str(
                fold) + ',' + target_mode + ',' +
            criterion_mode + ',' + optim_mode + ',' + str(learning_rate) + ',' + str(epoch) + ',' +
            str(pad_mae) + ',' + str(pad_mse) + ',' + str(pad_rmse) + ',' + str(total_loss.item()) + ',' + str(
                total_tokens.item()) + ',' +
            str(float(total_loss.item() / total_tokens.item())) + '\n')
        fold = fold + 1
    result_file_open.close()