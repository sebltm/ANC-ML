import numpy as np
import torch
import torch.nn as nn

import AudioDataset


class Net(nn.Module):

    def __init__(self, learning_rate=0.0001):
        super(Net, self).__init__()

        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda")
        self.criterion.to(self.device)
        self.iterator = AudioDataset.NoisyMusicDataset(musicFolder="Processed")

        # Encoder
        self.encodingLayer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=1, padding=0),
            nn.BatchNorm1d(64),
        )

        self.encodingLayer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=50, stride=1, padding=0),
            nn.BatchNorm1d(128),
        )

        self.encodingLayer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=50, stride=1, padding=0),
            nn.BatchNorm1d(256),
        )

        self.encodingLayer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=50, stride=1, padding=0),
            nn.BatchNorm1d(512),
        )

        # Decoder
        self.decodingLayer1 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=50, stride=1, padding=0),
            nn.BatchNorm1d(256),
        )

        self.decodingLayer2 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=50, stride=1, padding=0),
            nn.BatchNorm1d(128),
        )

        self.decodingLayer3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=50, stride=1, padding=0),
            nn.BatchNorm1d(64),
        )

        self.decodingLayer4 = nn.Sequential(
            nn.ConvTranspose1d(64, 1, kernel_size=50, stride=1, padding=0),
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)

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

    def train_generator(self, optimiser, criterion, device):

        epochs = 15
        size_batch = 10
        batch = 9 * 1000 // size_batch

        print("Training classifier with {} epochs, {} batches of size {}".format(epochs, batch, size_batch))

        self.train()
        for epoch in range(epochs):
            for num_batch in range(batch):

                real_noise = np.empty((size_batch, 1, 57330))
                music_generator = np.empty((size_batch, 1, 57330))

                self.iterator = AudioDataset.NoisyMusicDataset(musicFolder="Processed")

                if epoch == 0:
                    self.optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
                elif epoch == 5 or epoch == 10 or epoch == 15:
                    self.optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate / 10)

                for i in range(size_batch):
                    noise, music, noise_name, music_name = next(self.iterator)

                    try:
                        real_noise[i] = [noise]
                        music_generator[i] = [music]
                    except ValueError as e:
                        print(e)
                        print(music_name, noise_name)
                        i -= 1

                optimiser.zero_grad()

                input_network_tensor = torch.as_tensor(music_generator, dtype=torch.float32).to(device)

                output = self(input_network_tensor)

                for n in range(len(real_noise)):
                    inverseNoise = np.negative(output.cpu().detach().numpy()[n])
                    real_noise[n] += inverseNoise

                labels_tensor = torch.as_tensor(real_noise, dtype=torch.float32).to(device)
                loss = criterion(output, labels_tensor)
                loss.backward()
                optimiser.step()

                print("Epoch {}, batch {}, Generator loss: {}".format(epoch + 1, num_batch + 1, loss))
                print()

            torch.save(self.state_dict(), "generatorModelNew" + str(epoch) + ".pt")

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

        for i in range(0, 1000):
            _, music, noise_name, music_name = next(self.iterator)
            generatorInput = np.empty((1, 1, 57330))

            if (i % 20) == 2:
                generatorInput[0] = music
                print(noise_name)
                print(music_name)
                print()

                input_network = torch.as_tensor(generatorInput, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    output = self(input_network)

                for _, audio in enumerate(output.cpu().detach().numpy()):
                    file = open(folder + "/" + str(i // 20) + ".RAW", "wb")
                    file.write(audio)
                    file.close()
