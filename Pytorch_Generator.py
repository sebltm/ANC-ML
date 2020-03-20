import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import soundfile as sf

import FileProcessing


class Net(nn.Module):

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(Net, self).__init__()

        # Encoder
        self.encodingLayer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=0),
        )

        self.encodingLayer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
        )

        self.encodingLayer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0),
        )

        self.encodingLayer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0),
        )

        # Decoder
        self.decodingLayer1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1, padding=0),
        )

        self.decodingLayer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1, padding=0),
        )

        self.decodingLayer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=1, padding=0),
        )

        self.decodingLayer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=0),
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

        epochs = 3
        batch = 100
        size_batch = 20

        print("Training classifier with {} epochs, {} batches of size {}".format(epochs, batch, size_batch))

        self.train()
        for epoch in range(epochs):
            self.iterator = FileProcessing.FileIterator()
            for num_batch in range(batch):

                realNoise = np.empty((size_batch, 1, 64, 112))
                musicGenerator = np.empty((size_batch, 1, 64, 112))

                for i in range(size_batch):
                    music, noise = next(self.iterator)

                    realNoise[i] = [noise]
                    musicGenerator[i] = [music]

                # Train fake
                optimiser.zero_grad()

                input_network_tensor = torch.as_tensor(musicGenerator, dtype=torch.float32).to(device)

                output = self(input_network_tensor)

                labels_tensor = torch.as_tensor(musicGenerator, dtype=torch.float32).to(device)

                loss = criterion(output, labels_tensor)
                loss.backward()
                optimiser.step()

                print("Epoch {}, batch {}, Generator loss: {}".format(epoch + 1, num_batch + 1, loss))
                print()

            self.testGenerator(device)

    def testGenerator(self, device):
        # test
        generator_accuracy = []
        size_batch = 10

        self.iterator = FileProcessing.FileIterator()

        realNoise = np.empty((size_batch, 1, 64, 112))
        musicGenerator = np.empty((size_batch, 1, 64, 112))

        for i in range(size_batch):
            music, noise = next(self.iterator)

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

        self.generate()

    def generate(self):
        size_batch = 2

        self.iterator = FileProcessing.FileIterator()

        generatorInput = np.empty((size_batch, 1, 64, 112))

        for i in range(size_batch):
            music, noise = next(self.iterator)

            generatorInput[i] = [music]

        input_network = torch.as_tensor(generatorInput, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self(input_network)

        for i, mfcc in enumerate(output.cpu().detach().numpy()):
            tds = librosa.feature.inverse.mfcc_to_audio(mfcc[0], 64)
            samplerate = 44100
            sf.write("GANOutput/" + str(i) + ".wav", tds, samplerate, subtype='DOUBLE')
