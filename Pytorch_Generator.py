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
        self.iterator = AudioDataset.NoisyMusicDataset(noisy_music_folder="Processed")

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

        # calculate number of mini-batches depending on mini-batch size
        epochs = 15
        size_batch = 2
        batch = 9 * 1000 // size_batch

        print("Training classifier with {} epochs, {} batches of size {}".format(epochs, batch, size_batch))

        self.SDR = SignalDistortionRatio().to(device)
        self.SNR = SignalNoiseRatio().to(device)

        self.train()
        for epoch in range(epochs):

            if epoch == 0:
                self.optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            elif epoch == 5 or epoch == 10 or epoch == 15:
                self.optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate / 10)

            self.iterator = AudioDataset.NoisyMusicDataset(noisy_music_folder="ProcessedNew")

            for num_batch in range(batch):

                real_noise = np.empty((size_batch, 1, 57330))
                real_music = np.empty((size_batch, 1, 57330))
                music_generator = np.empty((size_batch, 1, 57330))

                for i in range(size_batch):
                    noise, noisy_music, music, noise_name, noisy_music_name, music_name = next(self.iterator)

                    try:
                        real_noise[i] = [noise]
                        real_music[i] = [music]
                        music_generator[i] = [noisy_music]
                    except ValueError as e:
                        print(e)
                        print(noise_name, noisy_music_name, music_name)
                        i -= 1

                optimiser.zero_grad()

                input_network_tensor = torch.as_tensor(music_generator, dtype=torch.float32).to(device)

                output = self(input_network_tensor)

                for n in range(len(real_noise)):
                    inverseNoise = np.negative(output.cpu().detach().numpy()[n])
                    music_generator[n] += inverseNoise

                predictions_tensor = torch.tensor(music_generator, requires_grad=True).to(device)
                real_tensor = torch.tensor(real_music, requires_grad=True).to(device)
                noise_tensor = torch.tensor(real_noise, requires_grad=True).to(device)

                loss = 0.5 * self.SDR(real_tensor, predictions_tensor) + 0.5 * self.SNR(real_tensor, predictions_tensor,
                                                                                        noise_tensor)
                loss.backward()
                optimiser.step()

                print("Epoch {}, batch {}, Generator loss: {}".format(epoch + 1, num_batch + 1, loss))

            torch.save(self.state_dict(), "generatorModel" + str(epoch) + ".pt")

    def generate(self, iterator: AudioDataset, folder):
        if iterator is not None:
            self.iterator = iterator

        for i in range(0, 1000):
            noise, noisy_music, music, noise_name, noisy_music_name, music_name = next(self.iterator)
            generatorInput = np.empty((1, 1, 57330))

            if (i % 20) == 0:
                generatorInput[0] = noisy_music
                print(noise_name)
                print(music_name)
                print(noisy_music_name)
                print()

                input_network = torch.as_tensor(generatorInput, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    output = self(input_network)

                for _, audio in enumerate(output.cpu().detach().numpy()):
                    file = open(folder + "/" + str(i // 20) + ".RAW", "wb")
                    file.write(audio)
                    file.close()


class SignalDistortionRatio(torch.nn.Module):
    def __init__(self, l1_penalty=0, epsilon=2e-7):
        super(SignalDistortionRatio, self).__init__()
        self.epsilon = epsilon

        self.cuda()

    def forward(self, target, prediction, interference=None):
        sdr = torch.zeros((len(prediction)))
        for n in range(len(prediction)):
            sdr[n] = 10 * torch.log10((torch.dot(prediction[n][0], prediction[n][0]) / (
                        torch.pow(torch.dot(prediction[n][0], target[n][0]), 2) + self.epsilon)))
        return torch.mean(sdr)


class SignalNoiseRatio(torch.nn.Module):
    def __init__(self):
        super(SignalNoiseRatio, self).__init__()
        self.epsilon = 1e-8

    def forward(self, target, prediction, noise):
        res = torch.zeros((len(prediction)))
        for n in range(len(prediction)):
            starget = torch.dot(prediction[n][0], target[n][0]) * target[n][0] / (
                        torch.dot(target[n][0], target[n][0]) + self.epsilon)
            snr = torch.dot(noise[n][0], noise[n][0]) / (torch.dot(starget, starget) + self.epsilon)

            res[n] = 10 * torch.log10(snr)

        return torch.mean(res)
