import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

import AudioDataset


class Net(nn.Module):

    def __init__(self):
        self.device = torch.device("cuda")

        super(Net, self).__init__()

        # Encoder
        self.encodingLayer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=2, stride=1, padding=0),
            # nn.BatchNorm2d(64),
        )

        self.encodingLayer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=2, stride=1, padding=0),
            # nn.BatchNorm2d(128),
        )

        self.encodingLayer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=2, stride=1, padding=0),
            # nn.BatchNorm2d(256),
        )

        self.encodingLayer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=2, stride=1, padding=0),
            # nn.BatchNorm2d(512),
        )

        # Decoder
        self.decodingLayer1 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=2, stride=1, padding=0),
            # nn.BatchNorm2d(256),
        )

        self.decodingLayer2 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=2, stride=1, padding=0),
            # nn.BatchNorm2d(128),
        )

        self.decodingLayer3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=1, padding=0),
            # nn.BatchNorm2d(64),
        )

        self.decodingLayer4 = nn.Sequential(
            nn.ConvTranspose1d(64, 1, kernel_size=2, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.encodingLayer1(x)
        x = self.encodingLayer2(x)
        x = self.encodingLayer3(x)
        x = self.encodingLayer4(x)
        x = self.decodingLayer1(x)
        x = self.decodingLayer2(x)
        x = self.decodingLayer3(x)
        x = self.decodingLayer4(x)
        return x

    def trainGenerator(self, optimiser, criterion, device):
        self.optimiser = optimiser
        self.criterion = criterion
        self.device = device

        epochs = 5
        size_batch = 10
        batch = 9*1000//size_batch

        print("Training classifier with {} epochs, {} batches of size {}".format(epochs, batch, size_batch))

        self.train()
        for epoch in range(epochs):
            self.iterator = AudioDataset.NoisyMusicDataset()
            for num_batch in range(batch):

                realNoise = np.empty((size_batch, 1, 57330))
                musicGenerator = np.empty((size_batch, 1, 57330))

                for i in range(size_batch):
                    noise, music, noise_name, music_name = next(self.iterator)

                    try:
                        realNoise[i] = [noise]
                        musicGenerator[i] = [music]
                    except ValueError as e:
                        print(e)
                        print(music_name, noise_name)
                        i -= 1

                optimiser.zero_grad()

                input_network_tensor = torch.as_tensor(musicGenerator, dtype=torch.float32).to(device)

                output = self(input_network_tensor)

                labels_tensor = torch.as_tensor(realNoise, dtype=torch.float32).to(device)
                loss = criterion(output, labels_tensor)
                loss.backward()
                optimiser.step()

                print("Epoch {}, batch {}, Generator loss: {}".format(epoch + 1, num_batch + 1, loss))
                print()

            self.generate(iterator=AudioDataset.NoisyMusicDataset(folderIndex=1), folder="GeneratorOutput")

            if epoch == epochs-1:
                answer = input("Do you want to save the current network?")
                if answer == "y":
                    torch.save(self.state_dict(), "generatorModel.pt")

                answer = input("Do you want to train for 5 more epochs?")
                if answer == "y":
                    epochs += 5

    def testGenerator(self, device):
        # test
        generator_accuracy = []
        size_batch = 10

        realNoise = np.empty((size_batch, 1, 64, 112))
        musicGenerator = np.empty((size_batch, 1, 64, 112))

        for i in range(size_batch):
            noise, music, noise_name, music_name = next(self.iterator)

            realNoise[i] = [noise]
            musicGenerator[i] = [music]

        input_network_tensor = torch.as_tensor(musicGenerator, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = self(input_network_tensor)

        predictions = torch.exp(output).cpu()

        print("Music predictions: ", predictions)

        # accuracy on training set
        # accuracy = mean_squared_error(realNoise, predictions)
        # generator_accuracy.append(accuracy)
        #
        # print("Average accuracy of noise: {}".format(np.average(generator_accuracy)))

    def generate(self, iterator: AudioDataset, folder):
        if iterator is not None:
            self.iterator = iterator

        size_batch = 10

        for i in range(0, 20 * size_batch):
            _, music, noise_name, music_name = next(self.iterator)
            generatorInput = np.empty((1, 1, 57330))

            if (i % 50) == 0:
                generatorInput[0] = music
                print(noise_name, music_name)

                input_network = torch.as_tensor(generatorInput, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    output = self(input_network)

                for _, audio in enumerate(output.cpu().detach().numpy()):
                    file = open(folder + "/" + str(i // 50) + ".RAW", "wb")
                    file.write(audio)
                    file.close()
