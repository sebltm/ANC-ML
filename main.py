import sys

import torch
from torch import nn

import AudioDataset
import GAN
import Pytorch_Classifier
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

        generator.train_generator(optimiser, criterion, device)

    elif args == "gan-train":
        gan = GAN.GAN(learning_rate=0.000001)
        gan.train()

    elif args == "gan-generate":
        generator = Pytorch_Generator.Net()

        device = torch.device("cuda")
        generator.to(device)

        generator.load_state_dict(torch.load("generatorGANModel14.pt"))
        generator.eval()

        generator.generate(AudioDataset.NoisyMusicDataset(musicFolder="ProcessedTest", folderIndex=0), "GANOutput")

    elif args == "generator-generate":
        generator = Pytorch_Generator.Net()

        device = torch.device("cuda")
        generator.to(device)

        generator.load_state_dict(torch.load("generatorModelNew1.pt"))
        generator.eval()

        generator.generate(AudioDataset.NoisyMusicDataset(musicFolder="ProcessedTest", folderIndex=0),
                           "GeneratorNewOutput")

    elif args == "visualise-generator":
        generator = Pytorch_Generator.Net()

        generator.load_state_dict(torch.load("generatorModelNew1.pt"))
        generator.eval()

        x = torch.zeros(1, 1, 57330, dtype=torch.float, requires_grad=False)
        out = generator(x)
        make_dot(out).render("GeneratorModelNoMFCC", format="png")

    elif args == "visualise-classifier":
        classifier = Pytorch_Classifier.Net()

        classifier.load_state_dict(torch.load("classifierGANModel.pt"))
        classifier.eval()

        x = torch.zeros(1, 2, 128, dtype=torch.float, requires_grad=False)
        out = classifier(x)
        make_dot(out).render("ClassifierModelNoMFCC", format="png")
