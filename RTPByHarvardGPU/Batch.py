from torch.autograd import Variable
from RTPByHarvardGPU.Tool import subsequent_mask
class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        self.trg = trg
        self.ntokens = (self.src != pad).data.sum()
        # if trg is not None:
        #     self.trg = trg[:, :-1]
        #     self.trg_y = trg[:, 1:]
        #     self.trg_mask = \
        #         self.make_std_mask(self.trg, pad)
        #     self.ntokens = (self.trg_y != pad).data.sum()
        #print(1,self.src_mask)

    # @staticmethod
    # def make_std_mask(tgt, pad):
    #     "Create a mask to hide padding and future words."
    #     tgt_mask = (tgt != pad).unsqueeze(-2)
    #     tgt_mask = tgt_mask & Variable(
    #         subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    #     return tgt_mask