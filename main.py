import sys

import torch
from torch import nn

import AudioDataset
import GAN
import Pytorch_Generator
import Pytorch_Transformer

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

        generator.train_generator(optimiser, criterion, device)

    elif args == "gan-train":
        gan = GAN.GAN(learning_rate=0.000001)
        gan.train()

    elif args == "gan-generate":
        generator = Pytorch_Generator.Net()

        device = torch.device("cuda")
        generator.to(device)

        generator.load_state_dict(torch.load("generatorGANModel0.pt"))
        generator.eval()

        generator.generate(AudioDataset.NoisyMusicDataset(musicFolder="ProcessedTest", folderIndex=0), "GANOutput")

    elif args == "generator-generate":
        generator = Pytorch_Generator.Net()

        device = torch.device("cuda")
        generator.to(device)

        generator.load_state_dict(torch.load("generatorModel9.pt"))
        generator.eval()

        generator.generate(AudioDataset.NoisyMusicDataset(musicFolder="ProcessedTest", folderIndex=0),
                           "GeneratorOutput")

    elif args == "transformer-train":
        transformer = Pytorch_Transformer.TransformerModel()

        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        transformer.to(device)
        criterion.to(device)

        optimiser = torch.optim.SGD(transformer.parameters(), lr=5.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, 1, gamma=0.95)

        transformer.train_transformer(optimiser, criterion, device)
