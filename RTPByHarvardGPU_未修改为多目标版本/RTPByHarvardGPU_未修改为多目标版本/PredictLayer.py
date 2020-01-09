import torch.nn as nn
class PredictLayer(nn.Module):
    """
      A standard Encoder-Decoder architecture. Base for this and many
      other models.
      """

    def __init__(self, encoder, src_embed):
        super(PredictLayer, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.liner = nn.Linear(512, 1)

    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        # print(self.encode(src, src_mask).size())
        return self.liner(self.encoder(self.src_embed(src), src_mask))

    def encode(self, src, src_mask):
        # print(src_mask)

        return self.encoder(self.src_embed(src), src_mask)
