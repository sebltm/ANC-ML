import torch
from torch import nn

import GAN
import Pytorch_Generator

if __name__ == "__main__":
    learning_rate = 0.0001

    # generator = Pytorch_Generator.Net()
    #
    # criterion = nn.MSELoss()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # generator.to(device)
    # criterion.to(device)
    #
    # optimiser = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    #
    # generator.trainGenerator(optimiser, criterion, device)

    gan = GAN.GAN(learning_rate=learning_rate)
    gan.train()

