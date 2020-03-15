import torch
import numpy as np
from sklearn.metrics import accuracy_score
import Pytorch_Generator as Generator
import Pytorch_Classifier as Classifier


class Trainer:
    factor = 1

    def __init__(self, model, criterion, optimiser):
        self.model = model
        self.criterion = criterion
        self.optimiser = optimiser

    def k_fold_validation(self, music, noise, device):
        noise_accuracy = []
        music_accuracy = []

        for i in range(0, len(music)):
            music_mfcc = music.copy()
            test_music = music_mfcc.pop(i)

            noise_mfcc = noise.copy()
            test_noise = noise_mfcc.pop(i)

            music_mfcc = np.array(music)
            noise_mfcc = np.array(noise)
            trained_model = self.train(music_mfcc, noise_mfcc, device)
            results = self.test(test_music, test_noise, trained_model)
            noise_accuracy.append(results[0])
            music_accuracy.append(results[1])

        print("Accuracy of {} folds for music is {}, for noise is {}, for both is {}"
              .format(len(music),
                      np.average(noise_accuracy),
                      np.average(music_accuracy),
                      np.average([np.average(noise_accuracy), np.average(music_accuracy)])))

    def train(self, music, noise, device):
        # noise is real
        real_label = 0

        # noisy music is fake
        fake_label = 1

        self.model.train()

        print(music.shape, noise.shape)

        labels = []
        if isinstance(self.model, Classifier.Net):
            input_network = np.vstack((music, noise))
        else:
            input_network = music

        input_network_tensor = torch.as_tensor(input_network, dtype=torch.float32).to(device)
        self.optimiser.zero_grad()
        output = self.model(input_network_tensor)

        if isinstance(self.model, Classifier.Net):
            for _ in range(len(output) // 2):
                labels.append(1)

            for _ in range(len(output) // 2):
                labels.append(0)
        else:
            labels = noise

        labels_tensor = torch.as_tensor(labels, dtype=torch.long).to(device)
        print(output.shape)
        print(labels_tensor.shape)

        loss = self.criterion(output, labels_tensor)
        loss.backward()
        self.optimiser.step()

        return loss

    def test(self, music, noise, model):
        music_accuracy = []
        noise_accuracy = []

        music = np.array(music)
        print(music.shape)

        noise = np.array(noise)
        print(noise.shape)

        # Test music files:
        labels = []
        input_network = torch.as_tensor(music, dtype=torch.float32).cuda()

        with torch.no_grad():
            output = model(input_network)

        for n in range(len(output)):
            labels.append(1.0)

        labels = torch.as_tensor(labels, dtype=torch.long)

        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)

        print("Music predictions: ", predictions)

        # accuracy on training set
        accuracy = accuracy_score(labels, predictions)
        music_accuracy.append(accuracy)

        # Test noise files
        labels = []
        input_network = torch.as_tensor(noise, dtype=torch.float32).cuda()

        with torch.no_grad():
            output = model(input_network)

        for n in range(len(output)):
            labels.append(0.0)

        labels = torch.as_tensor(labels, dtype=torch.long)

        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)

        print("Noise predictions: ", predictions)

        # accuracy on training set
        accuracy = accuracy_score(labels, predictions)
        noise_accuracy.append(accuracy)

        print("Average accuracy of noise: {}".format(np.average(noise_accuracy)))
        print("Average accuracy of music: {}".format(np.average(music_accuracy)))

        return np.average(noise_accuracy), np.average(music_accuracy)
