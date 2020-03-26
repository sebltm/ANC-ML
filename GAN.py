import numpy as np
import torch
import torch.nn as nn

import AudioDataset
import Pytorch_Classifier as Classifier
import Pytorch_Generator as Generator


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

        self.iterator = AudioDataset.NoisyMusicDataset()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self):
        epochs = 45
        batch = 900
        size_batch = 10

        self.ClassifierModel.apply(self.weights_init)
        self.GeneratorModel.apply(self.weights_init)

        # train classifier on small batch
        self.ClassifierModel.trainClassifier(self.optimiserClassifier, self.criterion, self.device)

        print("Training GAN with {} epochs, {} batches of size {}".format(epochs, batch, size_batch))

        self.ClassifierModel.train()
        for epoch in range(epochs):
            self.GeneratorModel.train()
            self.iterator = AudioDataset.NoisyMusicDataset()
            for num_batch in range(batch):

                real_classifier = np.empty((size_batch, 2, 200, 112))
                music_generator = np.empty((size_batch, 1, 200, 112))

                for i in range(size_batch):
                    noise, music, noise_name, music_name = next(self.iterator)

                    real_classifier[i] = np.vstack(([noise], [noise]))
                    music_generator[i] = [music]

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
                fake_data = np.empty((size_batch, 2, 200, 112))
                for i in range(size_batch):
                    generator_output_cpu = generator_output.cpu().detach().numpy()
                    fake_data[i] = np.vstack(([real_classifier[i][0]], generator_output_cpu[i]))

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

                # rebundle the generator's output into a batch for the classifier
                fake_data = np.empty((size_batch, 2, 200, 112))
                for i in range(size_batch):
                    generator_output_cpu = generator_output.cpu().detach().numpy()
                    fake_data[i] = np.vstack(([real_classifier[i][0]], generator_output_cpu[i]))

                fake_data = torch.as_tensor(fake_data, dtype=torch.float32).to(self.device)
                classifier_fake = self.ClassifierModel(fake_data)
                generator_loss = self.criterion(classifier_fake, labels_real)

                generator_loss.backward()
                self.optimiserGenerator.step()

                print("Epoch {}, batch {}, Classifier loss: {}, Generator loss: {}".format(epoch + 1, num_batch + 1,
                                                                                           classifier_fake_loss +
                                                                                           classifier_real_loss / 2,
                                                                                           generator_loss.data))

            self.GeneratorModel.generate(iterator=self.iterator, folder="GANOutput")

            while False:
                answer = input("Do you want to save the current network?")
                if answer == "y":
                    torch.save(self.GeneratorModel.state_dict(), "generatorGANModel.pt")
                    torch.save(self.ClassifierModel.state_dict(), "classifierGANModel.pt")
                    break
                elif answer == "n":
                    break
