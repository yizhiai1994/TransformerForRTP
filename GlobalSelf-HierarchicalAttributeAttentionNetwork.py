from DataOperate import DataOperate
import numpy as np
import torch
import torch.nn as nn
import math,copy, time
from torch.autograd import Variable
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
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
    print('query',query.size(),'key',key.size(),'value',value.size())
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
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

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
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
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1, x.size(-1))) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss.item() * norm
# Model
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
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
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
        #print(x.size())
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        print(x.size())
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    # FincalLayer
class PredictLayer(nn.Module):
    """
      A standard Encoder-Decoder architecture. Base for this and many
      other models.
      """
    def __init__(self, selfAN, atrbt_embed, pos_embd):
        super(PredictLayer, self).__init__()
        self.selfAN = selfAN
        self.atrbt_embed = atrbt_embed
        self.pos_embed = pos_embd
        self.src_embed = nn.Sequential(self.atrbt_embed, pos_embd)
        self.liner = nn.Linear(512, 1)

    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        # print(self.encode(src, src_mask).size())
        return self.liner(self.selfAN(self.src_embed(src), src_mask))

# TrainTool
def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1,attribute_length = 1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout,attribute_length=attribute_length)
    model = PredictLayer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Embeddings(d_model, src_vocab), c(position))

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
datafile = './Data/HospitalBilling.csv'
time_type='%Y/%m/%d %H:%M:%S.000'
trace_column = 0
event_column = 1
time_column = 2
attribute_columns = [1,7]
#    InitData
dataimpt = DataOperate(data_address=datafile,time_type = time_type)
dataimpt.build_trace(trace_column,attribute_columns,time_column)
dataimpt.build_vocab(attribute_columns)
dataimpt.build_orginal_trace_list()
dataimpt.build_processed_trace(start_pos=0)
trace_list = dataimpt.trace_list
train_list, test_list = train_test_split(trace_list,train_size=0.7)
train_list, test_list = dataimpt.build_train_test_trace(train_list, test_list, start_pos = 0)
#   InitModel
V = dataimpt.vocab_size
criterion = nn.MSELoss().cuda()# criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2,attribute_length=len(attribute_columns))
model = model.cuda()
criterion = criterion.cuda()
model_opt = torch.optim.Adam(model.parameters(), lr=0.1)# model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
#   Train
for epoch in range(10):
    print("epoch",epoch)
    model.train()
    # run_epoch(data_gen(train_list, Data_test.event2id, batch_size=200, padding_dix=0), model,
    #           SimpleLossCompute(model.generator, criterion, model_opt))
    run_epoch(data_gen(train_list, batch_size=500, padding_dix=0), model,
              TempLossCompute(criterion, model_opt))

    model.eval()

    print(run_epoch(data_gen(test_list, batch_size=500, padding_dix=0), model,
                    TempLossCompute(criterion, None)))

model.eval()
print(eval_model(data_gen(test_list, batch_size=500, padding_dix=0), model,
                    TempLossCompute(criterion, None)))
