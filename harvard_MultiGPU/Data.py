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


# Example
# Data_test = Data(data_address = '../Data/helpdesk.csv')
# Data_test.build_trace(0,[1])
# print(Data_test.trace_data)
