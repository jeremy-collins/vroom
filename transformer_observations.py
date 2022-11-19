import torch
import torch.nn as nn
import torch.optim as optim
from positional_encoding import PositionalEncoding
import math
import numpy as np

class TransformerObs(nn.Module):
    # Constructor
    def __init__(
        self,
        dim_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout_p=0.1,
    ):
        super().__init__()

        self.dim_model = dim_model
        self.input_dim = 26
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.SOS_token = torch.ones((1, 1, self.dim_model), dtype=torch.float32, device=self.device) * -100
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.dim_model))

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=64
        )
        # self.embedding = nn.Embedding(num_tokens, dim_model)
        self.embedding = nn.Linear(self.input_dim, dim_model)
        # self.transformer = nn.Transformer(
        #     d_model=dim_model,
        #     nhead=num_heads,
        #     num_encoder_layers=num_encoder_layers,
        #     num_decoder_layers=num_decoder_layers,
        #     dropout=dropout_p,
        # )

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=8)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.out = nn.Linear(dim_model, 4) # linear projection

    # def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
    def forward(self, X):
        # src = self.embedding(src)
        # tgt = self.embedding(tgt)

        # self.SOS_token = self.SOS_token.repeat(src.shape[0], 1, 1)

        # print("src", src.shape)
        # print("SOS", self.SOS_token.shape)

        # # append SOS token to the beginning of the target sequence
        # src = torch.cat((self.SOS_token, src), dim=1)
        # tgt = torch.cat((self.SOS_token, tgt), dim=1)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)

        # UNCOMMENT this if you want to use the embedding layer
        X_emb = self.embedding(X) # projecting from resnet output dim to transformer input dim: (batch_size, num_frames, dim_model)

        # SOS_token = self.SOS_token.repeat(X_emb.shape[0], 1, 1) # repeat SOS token for batch

        # X_emb = torch.cat((SOS_token, X_emb), dim=1)
        # concatenate class token to the beginning of the sequence for the batch
        X_emb = torch.cat((self.class_token.repeat(X_emb.shape[0], 1, 1), X_emb), dim=1)

        y = X_emb # because the target needs to be in the same vector space as the input.
                # we will predict a linear projection of the next embedding (see self.out in transformer.py)

        y = torch.tensor(y).to(self.device)

        # y_input = y
        # y_expected = y

        # shift the tgt by one so we always predict the next embedding
        y_input = y[:,:-1] # all but last

        # Get mask to mask out the future frames
        sequence_length = y_input.size(1)
        tgt_mask = self.get_tgt_mask(sequence_length).to(self.device)

        src = X_emb
        tgt = y_input

        src = src * math.sqrt(self.dim_model)
        tgt = tgt * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        # transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=None, tgt_key_padding_mask=None)
        transformer_out = self.transformer(src)
        # out = self.out(transformer_out) # output size: (sequence length, batch_size, 8)
        out = self.out(transformer_out[0]) # applying mlp to only the class token
        # out = transformer_out

        # return out
        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a square matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        # mask = torch.zeros(size, size)
        # mask[-1] = 1
        # mask[]
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        # mask = self.transformer.generate_square_subsequent_mask(1)

        # EX for size=5:
        # [[0.,   0.,   0.,   0.,   -inf.],
        #  [0.,   0.,   0.,   0.,   0.],
        #  [0.,   0.,   0.,   0.,   0.],
        #  [0.,   0.,   0.,   0.,   0.],
        #  [0.,   0.,   0.,   0.,   0.]]

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)