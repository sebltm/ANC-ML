import numpy as np
import torch
import torch.nn as nn

import AudioDataset
import Pytorch_Classifier as Classifier
import Pytorch_Generator as Generator


class GAN:

    def __init__(self, learning_rate=0.0001):
        self.learning_rate = learning_rate

        self.ClassifierModel = Classifier.Net()
        self.GeneratorModel = Generator.Net()
        self.criterion = nn.BCELoss()

        self.device = torch.device("cuda")

        self.ClassifierModel.to(self.device)
        self.GeneratorModel.to(self.device)
        self.criterion.to(self.device)

        self.optimiserGenerator = torch.optim.Adam(self.GeneratorModel.parameters(), lr=self.learning_rate)
        self.optimiserClassifier = torch.optim.Adam(self.ClassifierModel.parameters(), lr=self.learning_rate)

        self.iterator = AudioDataset.NoisyMusicDataset(noisy_music_folder="ProcessedNew")

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self):
        epochs = 15
        size_batch = 2
        batch = 9 * 1000 // size_batch

        self.ClassifierModel.apply(self.weights_init)
        self.ClassifierScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimiserClassifier)

        print("Training GAN with {} epochs, {} batches of size {}".format(epochs, batch, size_batch))

        # self.ClassifierModel.train()
        for epoch in range(epochs):

            if epoch == 0:
                self.optimiserGenerator = torch.optim.Adam(self.GeneratorModel.parameters(), lr=self.learning_rate)
                self.optimiserClassifier = torch.optim.Adam(self.ClassifierModel.parameters(), lr=self.learning_rate)
            elif epoch == 3 or epoch == 6 or epoch == 9 or epoch == 12:
                self.optimiserGenerator = torch.optim.Adam(self.GeneratorModel.parameters(),
                                                           lr=self.learning_rate / 100)
                self.optimiserClassifier = torch.optim.Adam(self.ClassifierModel.parameters(),
                                                            lr=self.learning_rate / 100)

            self.GeneratorModel.train()
            self.iterator = AudioDataset.NoisyMusicDataset(noisy_music_folder="ProcessedNew")

            for num_batch in range(batch):

                real_classifier = np.empty((size_batch, 2, 57330))
                real_noise = np.empty((size_batch, 1, 57330))
                music_generator = np.empty((size_batch, 1, 57330))

                for i in range(size_batch):
                    noise, noisy_music, music, noise_name, noisy_music_name, music_name = next(self.iterator)

                    real_classifier[i] = np.vstack(([music], [music]))
                    real_noise[i] = [noise]
                    music_generator[i] = [noisy_music]

                #######################################################################################
                ################################# TRAIN DISCRIMINATOR #################################
                #######################################################################################
                real_data = torch.as_tensor(real_classifier, dtype=torch.float32).to(self.device)
                generator_input = torch.as_tensor(music_generator, dtype=torch.float32).to(self.device)

                # real
                self.ClassifierModel.zero_grad()

                classifier_real = self.ClassifierModel(real_data)
                labels_real = torch.ones(classifier_real.size()).to(self.device)
                classifier_real_loss = self.criterion(classifier_real, labels_real)
                classifier_real_loss.backward()

                # fake
                generator_output = self.GeneratorModel(generator_input).detach()


                # rebundle the generator's output into a batch for the classifier
                fake_data = np.empty((size_batch, 2, 57330))
                for i in range(size_batch):
                    inverseNoise = np.negative(generator_output.cpu().detach().numpy()[i])
                    music_generator[i] += inverseNoise

                    fake_data[i] = np.vstack(([real_classifier[i][0]], music_generator[i]))

                fake_data = torch.as_tensor(fake_data, dtype=torch.float32).to(self.device)
                classifier_fake = self.ClassifierModel(fake_data)
                labels_fake = torch.zeros(classifier_fake.size()).to(self.device)
                classifier_fake_loss = self.criterion(classifier_fake, labels_fake)

                classifier_fake_loss.backward()
                self.optimiserClassifier.step()

                #######################################################################################
                ################################### TRAIN GENERATOR ###################################
                #######################################################################################
                self.GeneratorModel.zero_grad()

                fake_data = torch.as_tensor(fake_data, dtype=torch.float32).to(self.device)
                classifier_fake = self.ClassifierModel(fake_data)
                generator_loss = self.criterion(classifier_fake, labels_real)

                generator_loss.backward()
                self.optimiserGenerator.step()

                print("Epoch {}, batch {}, Classifier loss: {}, Generator loss: {}".format(epoch + 1, num_batch + 1,
                                                                                           classifier_fake_loss +
                                                                                           classifier_real_loss / 2,
                                                                                           generator_loss.data))

            # self.GeneratorModel.generate(iterator=AudioDataset.NoisyMusicDataset(folderIndex=1), folder="GANOutput")

            torch.save(self.GeneratorModel.state_dict(), "generatorGANModel" + str(epoch) + ".pt")
            torch.save(self.ClassifierModel.state_dict(), "classifierGANModel" + str(epoch) + ".pt")
