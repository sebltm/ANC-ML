import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

import AudioDataset

np.set_printoptions(threshold=sys.maxsize)


class Net(nn.Module):

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(Net, self).__init__()

        self.iterator = AudioDataset.NoisyMusicDataset()

        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=0),
            nn.Tanhshrink(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.25)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.Tanhshrink(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.5)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.Tanhshrink(),
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.5)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512, 10),
            nn.Tanhshrink(),
            nn.Dropout(0.5),
            nn.Linear(10, 2),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, x.size(1))
        x = self.fc1(x)
        return x

    def trainClassifier(self, optimiser, criterion, device):
        self.optimiser = optimiser
        self.criterion = criterion
        self.device = device

        epochs = 3
        batch = 15
        size_batch = 10

        print("Training classifier with {} epochs, {} batches of size {}".format(epochs, batch, size_batch))

        self.train()
        for epoch in range(epochs):
            self.iterator = AudioDataset.NoisyMusicDataset()
            for num_batch in range(batch):

                fake = np.empty((size_batch, 2, 128, 112))
                real = np.empty((size_batch, 2, 128, 112))

                for i in range(size_batch):
                    noise, music, noise_file, music_file = next(self.iterator)

                    fake[i] = np.vstack(([noise], [music]))
                    real[i] = np.vstack(([noise], [noise]))

                self.zero_grad()

                # Train real
                self.optimiser.zero_grad()

                input_network_tensor = torch.as_tensor(real, dtype=torch.float32).to(self.device)
                output = self(input_network_tensor)

                labels = torch.ones(output.size())
                labels_tensor = torch.as_tensor(labels, dtype=torch.float32).to(self.device)

                loss_real = self.criterion(output, labels_tensor)
                loss_real.backward()
                self.optimiser.step()

                self.zero_grad()

                # Train fake
                # self.optimiser.zero_grad()
                #
                # input_network_tensor = torch.as_tensor(fake, dtype=torch.float32).to(self.device)
                #
                # output = self(input_network_tensor)
                #
                # labels = torch.zeros(output.size())
                # labels_tensor = torch.as_tensor(labels, dtype=torch.float32).to(self.device)
                #
                # loss_fake = self.criterion(output, labels_tensor)
                # loss_fake.backward()
                # self.optimiser.step()

                print("Epoch {}, batch {}, Classifier loss: {}".format(epoch + 1, num_batch + 1, loss_real))
                print()

    def testClassifier(self):
        # test
        music_accuracy = []
        noise_accuracy = []
        size_batch = 2

        self.iterator = AudioDataset.NoisyMusicDataset()

        fake = np.empty((size_batch, 2, 128, 112))
        real = np.empty((size_batch, 2, 128, 112))

        for i in range(size_batch):
            music, noise = next(self.iterator)

            fake[i] = np.vstack(([noise], [music]))
            real[i] = np.vstack(([noise], [noise]))

        # # Test music files:
        # labels = []
        # input_network = torch.as_tensor(fake, dtype=torch.float32).to(self.device)
        #
        # with torch.no_grad():
        #     output = self(input_network)
        #
        # labels = torch.zeros(output.size())
        #
        # softmax = torch.exp(output).cpu()
        # prob = list(softmax.numpy())
        # predictions = np.argmax(prob, axis=1)
        #
        # print("Music predictions: ", predictions)

        # # accuracy on training set
        # accuracy = accuracy_score(np.argmax(labels, axis=1), predictions)
        # music_accuracy.append(accuracy)

        # Test noise files
        input_network = torch.as_tensor(real, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self(input_network)

        labels = torch.ones(output.size()[0])

        softmax = output.detach()

        prob = list(softmax.cpu().numpy())
        predictions = np.argmax(prob, axis=1)

        # accuracy on training set
        accuracy = accuracy_score(labels.data, predictions)
        noise_accuracy.append(accuracy)

        print("Average accuracy of noise: {}".format(np.average(noise_accuracy)))
        # print("Average accuracy of music: {}".format(np.average(music_accuracy)))
