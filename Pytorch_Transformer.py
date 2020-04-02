import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import AudioDataset


class TransformerModel(nn.Module):

    def __init__(self, num_embed=1, embed_dim=4410, nhead=3, nhid=200, nlayers=2, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(num_embed, embed_dim)
        self.embed_dim = embed_dim
        self.decoder = nn.Linear(embed_dim, embed_dim)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = src * math.sqrt(self.embed_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def train_transformer(self, optimiser, criterion, device):

        epochs = 15
        size_batch = 10
        batch = 9 * 1000 // size_batch

        print("Training classifier with {} epochs, {} batches of size {}".format(epochs, batch, size_batch))

        self.train()
        for epoch in range(epochs):
            for num_batch in range(batch):

                real_noise = np.empty((size_batch, 4410))
                music_generator = np.empty((size_batch, 1, 4410))

                self.iterator = AudioDataset.NoisyMusicDataset(musicFolder="Processed", duration=0.1, samplerate=44100,
                                                               convertToInt=True)

                # if epoch == 0:
                #     self.optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
                # elif epoch == 5 or epoch == 10 or epoch == 15:
                #     self.optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate / 10)

                for i in range(size_batch):
                    noise, music, noise_name, music_name = next(self.iterator)

                    try:
                        real_noise[i] = noise
                        music_generator[i] = music
                    except ValueError as e:
                        print(e)
                        print(music_name, noise_name)
                        i -= 1

                optimiser.zero_grad()

                input_network_tensor = torch.as_tensor(music_generator, dtype=torch.int64).to(device)

                output = self(input_network_tensor)

                labels_tensor = torch.as_tensor(real_noise, dtype=torch.int64).to(device)
                loss = criterion(output, labels_tensor)
                loss.backward()
                optimiser.step()

                print("Epoch {}, batch {}, Generator loss: {}".format(epoch + 1, num_batch + 1, loss))
                print()

                # if num_batch % 500 == 0:
                #     self.generate(iterator=AudioDataset.NoisyMusicDataset(folderIndex=1), folder="GeneratorOutput")

            torch.save(self.state_dict(), "generatorModel" + str(epoch) + ".pt")


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
