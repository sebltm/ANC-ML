import sys

import torch
from torch import nn

import AudioDataset
import GAN
import Pytorch_Generator

if __name__ == "__main__":

    args = list(sys.argv)[1]
    learning_rate = 0.0001

    if args == "generator-train":
        # Train a generator in a supervised manner
        generator = Pytorch_Generator.Net()

        criterion = nn.MSELoss()
        device = torch.device("cuda")

        generator.to(device)
        criterion.to(device)

        optimiser = torch.optim.Adam(generator.parameters(), lr=learning_rate)

        generator.trainGenerator(optimiser, criterion, device)

    elif args == "gan-train":
        gan = GAN.GAN(learning_rate=learning_rate)
        gan.train()

    elif args == "generator-generate":
        print("Generating 50 samples")

        generator = Pytorch_Generator.Net()

        device = torch.device("cuda")
        generator.to(device)

        generator.load_state_dict(torch.load("generatorModel.pt"))
        generator.eval()

        generator.generate(AudioDataset.NoisyMusicDataset(folderIndex=9), "GeneratorOutput")
