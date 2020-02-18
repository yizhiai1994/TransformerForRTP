import torch
from sklearn.model_selection import train_test_split
import datetime
class DataOperate():
    def __init__(self,data_address = '',first_line_skip = True, split_kw = ',',encoding = 'utf-8',time_type = '%Y-%m-%d %H:%M:%S'):
        self.data_address = data_address
        self.first_line_skip = first_line_skip
        self.split_kw = split_kw
        self.encoding = encoding
        self.file_content = self.readFile()
        self.time_type = time_type
    def readFile(self):
        content = list()
        with open(self.data_address,'r',encoding=self.encoding) as file:
            if self.first_line_skip == True:
                next(file)
            file_read = file.readlines()
            for line in file_read:
                line = line.replace('\r','').replace('\n','').split(self.split_kw)
                content.append(line.copy())
        return content
    def build_trace(self,main_colum,minor_colums,time_colums):
        trace_data = dict()
        for line in self.file_content:
            if line[main_colum] not in trace_data:
                trace_data[line[main_colum]] = list()
                temp_content = list()
                for colum in minor_colums:
                    temp_content.append(line[colum])
                trace_data[line[main_colum]].append([temp_content.copy(),line[time_colums]])
            else:
                temp_content = list()
                for colum in minor_colums:
                    temp_content.append(line[colum])
                trace_data[line[main_colum]].append([temp_content.copy(),line[time_colums]])
        self.trace_data = trace_data
    def build_orginal_trace_list(self):
        self.trace_list = list()
        self.orginal_list = [self.trace_data[t] for t in self.trace_data]
        for trace in self.orginal_list:

            src_temp = [[self.event2id[attribute] for attribute in event[0]] for event in trace]
            tgt_temp = [float((datetime.datetime.strptime(trace[-1][1], self.time_type) -
                               datetime.datetime.strptime(record[1], self.time_type)).seconds / 60 / 60)
                        for record in trace]
            self.trace_list.append((src_temp.copy(),tgt_temp.copy()))
    def build_processed_trace(self, start_pos):
        processed_trace = list()
        for src, tgt in self.trace_list:
            for pos_temp in range(len(src)):
                #print(src)
                if pos_temp < start_pos:
                    continue
                processed_trace.append((src[0:pos_temp + 1].copy(),tgt[0:pos_temp + 1].copy()))
        print(len(processed_trace),len(self.trace_list))
        self.processed_trace = processed_trace
    def build_train_test_trace(self,train,test,start_pos):
        train_trace = list()
        test_trace = list()
        for src, tgt in train:
            for pos_temp in range(len(src)):
                if pos_temp < start_pos:
                    continue
                train_trace.append((src[0:pos_temp + 1].copy(), tgt[0:pos_temp + 1].copy()))
        for src, tgt in test:
            for pos_temp in range(len(src)):
                if pos_temp < start_pos:
                    continue
                test_trace.append((src[0:pos_temp + 1].copy(), tgt[0:pos_temp + 1].copy()))
        return train_trace, test_trace
    def build_vocab(self,key_row):
        vocab_list = list()
        event2id = dict()
        id2event = dict()
        event2id['<block>'] = 0
        id2event[0] = '<block>'
        for key_temp in key_row:
            for line in self.file_content:
                vocab_list.append(line[key_temp])
        for key in list(set(vocab_list)):
            event2id[key] = len(event2id)
            id2event[len(id2event)] = key
        self.event2id = event2id
        self.id2event = id2event
        self.vocab_size = len(event2id)

    # def split_train_test(self,train_size):
    #     trace_list = [self.trace_data[k] for k in self.trace_data]
    #     self.train_trace, self.test_trace = train_test_split(self.trace_data,train_size=train_size, random_state=42)
    # def gen_batch_predicEvent(self):
    #     for trace in self.train_trace:
    #         train_src_temp = [self.event2id[record[0]] for record in trace_list[trace][:-1]]
    #         tgt_temp = [event2id[record[0]] for record in trace_list[trace][1:]]



# Example
# Data_test = Data(data_address = '../Data/helpdesk.csv')
# Data_test.build_trace(0,[1,2])
# print(Data_test.trace_data)
# Data_test.build_vocab(1)
# print(Data_test.event2id['2'])
# for line in Data_test.trace_data:
#
#     temp = [Data_test.event2id[record[0]] for record in Data_test.trace_data[line]]
#     print(temp)
#     temp = torch.Tensor(temp)
#     print(temp)
