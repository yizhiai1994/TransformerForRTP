import torch
class Data():
    def __init__(self,data_address = '',first_line_skip = True, split_kw = ',',encoding = 'utf-8'):
        self.data_address = data_address
        self.first_line_skip = first_line_skip
        self.split_kw = split_kw
        self.encoding = encoding
        self.file_content = self.readFile()
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
    def build_trace(self,main_colum,minor_colums):
        trace_data = dict()
        for line in self.file_content:
            if line[main_colum] not in trace_data:
                trace_data[line[main_colum]] = list()
                temp_content = list()
                for colum in minor_colums:
                    temp_content.append(line[colum])
                trace_data[line[main_colum]].append(temp_content.copy())
            else:
                temp_content = list()
                for colum in minor_colums:
                    temp_content.append(line[colum])
                trace_data[line[main_colum]].append(temp_content.copy())
        self.trace_data = trace_data
    def build_vocab(self,key_row):
        vocab_list = list()
        event2id = dict()
        id2event = dict()
        event2id['<block>'] = 0
        id2event[0] = '<block>'
        for line in self.file_content:
            vocab_list.append(line[key_row])
        for key in list(set(vocab_list)):
            event2id[key] = len(event2id)
            id2event[len(id2event)] = key
        self.event2id = event2id
        self.id2event = id2event
        self.vocab_size = len(event2id)


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
