import Pytorch_Generator as Generator
import Pytorch_Classifier as Classifier
import TrainTest
import FileProcessing
import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
from sklearn.metrics import accuracy_score, mean_squared_error
import librosa.feature
import soundfile as sf


class GAN:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

        self.ClassifierModel = Classifier.Net()
        self.GeneratorModel = Generator.Net()
        self.criterion = nn.BCELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ClassifierModel.to(self.device)
        self.GeneratorModel.to(self.device)
        self.criterion.to(self.device)

        self.optimiserGenerator = torch.optim.Adam(self.GeneratorModel.parameters(), lr=self.learning_rate)
        self.optimiserClassifier = torch.optim.Adam(self.ClassifierModel.parameters(), lr=self.learning_rate)

        self.iterator = FileProcessing.FileIterator()

    def train(self):
        epochs = 45
        batch = 250
        size_batch = 1

        # train classifier on small batch
        self.ClassifierModel.trainClassifier(self.optimiserClassifier, self.criterion, self.device)

        print("Training GAN with {} epochs, {} batches of size {}".format(epochs, batch, size_batch))

        self.ClassifierModel.train()
        for epoch in range(epochs):
            self.GeneratorModel.train()
            self.iterator = FileProcessing.FileIterator(noise_samples=10)
            for num_batch in range(batch):

                realClassifier = np.empty((size_batch, 2, 57330))
                musicGenerator = np.empty((size_batch, 1, 57330))

                for i in range(size_batch):
                    music, noise = next(self.iterator)

                    realClassifier[i] = np.vstack(([noise], [noise]))
                    musicGenerator[i] = [music]

                #######################################################################################
                ################################# TRAIN DISCRIMINATOR #################################
                #######################################################################################
                realClassifierTensor = torch.as_tensor(realClassifier, dtype=torch.float32).to(self.device)
                generatorInput = torch.as_tensor(musicGenerator, dtype=torch.float32).to(self.device)

                # real
                self.ClassifierModel.zero_grad()

                classifierReal = self.ClassifierModel(realClassifierTensor)
                labelsReal = torch.ones(classifierReal.size()).to(self.device)
                classifierReal_loss = self.criterion(classifierReal, labelsReal)
                classifierReal_loss.backward()

                # fake

                # don't update the generator
                generatorOutput = self.GeneratorModel(generatorInput)

                # rebundle the generator's output into a batch for the classifier
                fakeClassifier = np.empty((size_batch, 2, 57330))
                for i in range(size_batch):
                    generatorOutputCPU = generatorOutput.cpu().detach().numpy()
                    fakeClassifier[i] = np.vstack(([realClassifier[i][0]], generatorOutputCPU[i]))

                fakeClassifier = torch.as_tensor(fakeClassifier, dtype=torch.float32).to(self.device)
                classifierFake = self.ClassifierModel(fakeClassifier)
                labelsFake = torch.zeros(classifierFake.size()).to(self.device)
                classifierFake_loss = self.criterion(classifierFake, labelsFake)

                classifierFake_loss.backward()
                self.optimiserClassifier.step()

                #######################################################################################
                ################################### TRAIN GENERATOR ###################################
                #######################################################################################
                self.GeneratorModel.zero_grad()

                # rebundle the generator's output into a batch for the classifier
                fakeClassifier = np.empty((size_batch, 2, 57330))
                for i in range(size_batch):
                    generatorOutputCPU = generatorOutput.cpu().detach().numpy()
                    fakeClassifier[i] = np.vstack(([realClassifier[i][0]], generatorOutputCPU[i]))

                fakeClassifier = torch.as_tensor(fakeClassifier, dtype=torch.float32).to(self.device)
                classifierFake = self.ClassifierModel(fakeClassifier)
                generatorLoss = self.criterion(classifierFake, labelsReal)

                generatorLoss.backward()
                self.optimiserGenerator.step()

                print("Epoch {}, batch {}, Classifier loss: {}, Generator loss: {}".format(epoch + 1, num_batch + 1,
                                                                                           classifierFake_loss +
                                                                                           classifierReal_loss / 2,
                                                                                           generatorLoss.data))

            self.ClassifierModel.testClassifier()
            self.GeneratorModel.generate()
