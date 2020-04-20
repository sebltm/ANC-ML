import librosa
import numpy as np
import soundfile as sf
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
            nn.Conv1d(1, 64, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(64),
        )

        self.encodingLayer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(128),
        )

        self.encodingLayer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(256),
        )

        self.encodingLayer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(512),
        )

        # Decoder
        self.decodingLayer1 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(256),
        )

        self.decodingLayer2 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(128),
        )

        self.decodingLayer3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(64),
        )

        self.decodingLayer4 = nn.Sequential(
            nn.ConvTranspose1d(64, 1, kernel_size=2, stride=1, padding=0),
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
        self.optimiser = optimiser
        self.criterion = criterion
        self.device = device

        # calculate number of mini-batches depending on mini-batch size
        epochs = 15
        size_batch = 2
        batch = 9 * 1000 // size_batch

        print("Training classifier with {} epochs, {} batches of size {}".format(epochs, batch, size_batch))

        self.train()
        for epoch in range(epochs):

            # reduce learning rate every 5 epochs
            if epoch == 0:
                self.optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            elif epoch == 5 or epoch == 10 or epoch == 15:
                self.optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate / 10)

            # go back to beginning of dataset
            self.iterator = AudioDataset.NoisyMusicDataset(noisy_music_folder="ProcessedNew", mode='trainMFCC')

            for num_batch in range(batch):

                # create numpy containers for data
                real_noise = np.empty((size_batch, 1, 128, 112))
                music_generator = np.empty((size_batch, 1, 128, 112))

                # get samples
                for i in range(size_batch):
                    noise, noisy_music, music, noise_name, noisy_music_name, music_name = next(self.iterator)

                    try:
                        real_noise[i] = [noise]
                        music_generator[i] = [noisy_music]
                    except ValueError as e:
                        print(e)
                        print(noise_name, noisy_music_name, music_name)
                        i -= 1

                # train
                optimiser.zero_grad()

                input_network_tensor = torch.as_tensor(music_generator, dtype=torch.float32).to(device)

                output = self(input_network_tensor)

                labels_tensor = torch.as_tensor(real_noise, dtype=torch.float32).to(device)

                loss = criterion(output, labels_tensor)
                loss.backward()
                optimiser.step()

                print("Epoch {}, batch {}, Generator loss: {}".format(epoch + 1, num_batch + 1, loss))

            torch.save(self.state_dict(), "generatorModelMFCC" + str(epoch) + ".pt")

    def generate(self, iterator: AudioDataset, folder):

        if iterator is not None:
            self.iterator = iterator

        for i in range(0, 1000):
            noise, noisy_music, music, noise_name, noisy_music_name, music_name = next(self.iterator)
            generator_input = np.empty((1, 1, 128, 112))

            # generate every 20 samples - same music, different noises
            if (i % 20) == 0:
                generator_input[0] = noisy_music
                print(noise_name)
                print(music_name)
                print(noisy_music_name)
                print()

                input_network = torch.as_tensor(generator_input, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    output = self(input_network)

                # invert mfcc and write results
                for _, mfcc in enumerate(output.cpu().detach().numpy()):
                    tds = librosa.feature.inverse.mfcc_to_audio(mfcc[0], 128)
                    sample_rate = 44100
                    sf.write(folder + "/" + str(i // 50) + ".wav", tds, sample_rate, subtype='FLOAT')
