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
from sklearn.model_selection import KFold
from RTPByHarvardGPU.TrainTool import build_k_fold_data
strptime = datetime.datetime.strptime
import os
import os.path
from RTPByHarvardGPU.TrainTool import make_model, run_epoch, greedy_decode,eval_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
global max_src_in_batch, max_tgt_in_batch
# Train the simple copy task.
#Record Information
result_folder = './result/'
record_folder = './result/record/'
model_save_folder = ''
dataset_name = 'helpdesk'
result_file = result_folder + dataset_name + '.csv'
k = 5
verify_mode = str(k) + '-fold' #k-fold or trani:test
train_to_test = str(k-1) + ':' + str(1)
target_mode = 'simple'
criterion_mode = 'MSE'
optim_mode = 'Adam'
learning_rate = 0.1
if not os.path.isfile(result_file):
    result_file_open = open(result_file,'w',encoding='utf-8')
    result_file_open.writelines('数据集,验证方式,训练集:测试集,验证次数,单目标/多目标,损失函数,优化方法,学习率,训练轮数,MAE,MSE,RMSE,total_loss,total_tokens,total_loss/total_tokens\n')
else:
    result_file_open = open(result_file,'a',encoding='utf-8')
#Build Data

Data_test = Data(data_address = '../Data/helpdesk.csv')
Data_test.build_trace(0,[1,2])
Data_test.build_vocab(1)
Data_test.build_orginal_trace_list()
Data_test.build_processed_trace(start_pos=0)
trace_list = Data_test.trace_list
k_fold_data = build_k_fold_data(k,trace_list)
#print(len(k_fold_data))
def data_gen(trace_list,event2id, padding_dix,batch_size):
    src_batch_temp = list()
    tgt_batch_temp = list()
    time = len(trace_list)
    #d = datetime.datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S')
    for src, tgt in trace_list:
        #print(trace)
        #print(tgt_temp)
        src_batch_temp.append(src.copy())
        tgt_batch_temp.append(src.copy())
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
# Train
fold = 1
for train_data, test_data in k_fold_data:
    #print(train_data)
    print('fold',fold)
    V = Data_test.vocab_size
    criterion = nn.L1Loss().cuda()
    model = make_model(V, V, N=2)
    model = model.cuda()
    criterion = criterion.cuda()
    model_opt = torch.optim.Adam(model.parameters(), lr=0.1)
    train_list, test_list = Data_test.build_train_test_trace(train_data, test_data, start_pos=0)
    for epoch in range(10):
        print(epoch)
        model.train()
        run_epoch(data_gen(train_list, Data_test.event2id, batch_size=500, padding_dix=0), model,
                  TempLossCompute(criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(test_list, Data_test.event2id, batch_size=500, padding_dix=0), model,
                        TempLossCompute(criterion, None)))
    #
    model.eval()
    # print(eval_model(data_gen(test_list, Data_test.event2id, batch_size=500, padding_dix=0), model,
    #                     TempLossCompute(criterion, None)))
    total_loss, total_tokens, (pad_mse, no_pad_mse, pad_rmse, no_pad_rmse, pad_mae, no_pad_mae) = \
        eval_model(data_gen(test_list, Data_test.event2id, batch_size=500, padding_dix=0), model,
                   TempLossCompute(criterion, None))
    '数据集,验证方式,训练集:测试集,验证次数,单目标/多目标,损失函数,优化方法,学习率,训练轮数,MAE,MSE,RMSE,,,\n'
    result_file_open.writelines(dataset_name +','+ verify_mode +','+ str(len(train_data)) + ':' + str(len(test_data)) +','+ str(fold) +','+ target_mode +','+
                                 criterion_mode +','+ optim_mode +','+ str(learning_rate) +','+ str(epoch) +','+
                                str(pad_mae) +','+ str(pad_mse) +','+ str(pad_rmse) +','+ str(total_loss.item()) +','+ str(total_tokens.item()) +','+
                                 str(float(total_loss.item()/total_tokens.item())) +'\n')
    fold = fold + 1
result_file_open.close()
# src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]).cuda() )
# src_mask = Variable(torch.ones(1, 1, 10) )
# print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))









