import numpy as np
import librosa
import librosa.feature
import soundfile as sf
from pathlib import Path


class FileIterator:

    def __init__(self, samplerate=44100, noise_samples=50):
        self.samplerate = samplerate
        self.folders = sorted(list(Path("../Sounds/Processed/").glob("*")))

        self.folderIndex = 0
        self.sampleIndex = 0
        self.noiseSamples = noise_samples

    def __iter__(self):
        self.__next__()

    def __next__(self):
        self.noiseMfcc = np.empty((64, 112))
        self.noisyMusicMfcc = np.empty((64, 112))

        folder = self.folders[self.sampleIndex // self.noiseSamples]

        self.noiseFiles = sorted(list(Path("../Sounds/UrbanSound8K/audio/").glob("**/*.wav")))
        self.noisyMusicFiles = sorted(list(Path(folder).glob("[0-9]*.RAW")))

        sample_index = self.sampleIndex % self.noiseSamples
        print(self.sampleIndex // self.noiseSamples * self.noiseSamples + sample_index)

        # SAMPLE A NOISE
        raw_noise = sf.SoundFile(self.noiseFiles[sample_index].as_posix())
        raw_noise_read = raw_noise.read(dtype=np.double, always_2d=False)
        noise_transpose = raw_noise_read.T
        noise_mono = librosa.to_mono(noise_transpose)[0:57330]

        while len(noise_mono) < 57330:
            raw_noise = sf.SoundFile(self.noiseFiles[sample_index].as_posix())
            raw_noise_read = raw_noise.read(dtype=np.double, always_2d=False)
            noise_transpose = raw_noise_read.T
            noise_mono = librosa.to_mono(noise_transpose)[0:57330]

        noise_mfcc = librosa.feature.mfcc(y=noise_mono, sr=self.samplerate, n_mfcc=64)
        self.noiseMfcc = noise_mfcc

        # SAMPLE A MUSIC
        raw_noisy_music = sf.SoundFile(self.noisyMusicFiles[sample_index].as_posix(),
                                       samplerate=self.samplerate,
                                       channels=1,
                                       subtype='DOUBLE')
        raw_noisy_music_read = raw_noisy_music.read(dtype=np.double, always_2d=False)
        noisy_music_transpose = raw_noisy_music_read.T[0:57330]

        noisy_music_mfcc = librosa.feature.mfcc(y=noisy_music_transpose,
                                                sr=self.samplerate,
                                                dtype=np.double,
                                                n_mfcc=64)
        self.noisyMusicMfcc = noisy_music_mfcc
        self.sampleIndex += 1

        return self.noiseMfcc, self.noisyMusicMfcc

    # def next(self):
    #     self.noiseMfcc = np.empty((10, 64, 44))
    #     self.noisyMusicMfcc = []
    #
    #     print(self.folderIndex)
    #
    #     folder = self.folders[self.folderIndex]
    #
    #     self.noiseFiles = list(Path("../Sounds/UrbanSound8K/audio/").glob("**/*.wav"))
    #     self.noisyMusicFiles = list(Path(folder).glob("[0-9]*.RAW"))
    #
    #
    #     # Read the first 10 audio noise filesextMusic again and again
    #     for i, noiseIndex in enumerate(range(0, 10)):
    #         rawNoise = sf.SoundFile(self.noiseFiles[noiseIndex].as_posix())
    #         rawNoiseRead = rawNoise.read(dtype=np.double, always_2d=False)
    #         noiseTranspose = rawNoiseRead.T
    #         noiseMono = librosa.to_mono(noiseTranspose)[0:22050]
    #
    #         while len(noiseMono) < 22050:
    #             noiseIndex += 1
    #             rawNoise = sf.SoundFile(self.noiseFiles[noiseIndex].as_posix())
    #             rawNoiseRead = rawNoise.read(dtype=np.double, always_2d=False)
    #             noiseTranspose = rawNoiseRead.T
    #             noiseMono = librosa.to_mono(noiseTranspose)[0:22050]
    #
    #         noiseMfcc = librosa.feature.mfcc(y=noiseMono, sr=self.samplerate, n_mfcc=64)
    #         self.noiseMfcc[i] = [noiseMono]
    #
    #     # Read the first 10 combinations in each folder
    #     for noisyMusicIndex in range(min(10, len(self.noiseMfcc))):
    #         rawNoisyMusic = sf.SoundFile(self.noisyMusicFiles[noisyMusicIndex].as_posix(),
    #                                      samplerate=self.samplerate,
    #                                      channels=1,
    #                                      subtype='DOUBLE')
    #         rawNoisyMusicRead = rawNoisyMusic.read(dtype=np.double, always_2d=False)
    #         noisyMusicTranspose = rawNoisyMusicRead.T
    #
    #         try:
    #             noisyMusicMfcc = librosa.feature.mfcc(y=noisyMusicTranspose,
    #                                                   sr=self.samplerate,
    #                                                   dtype=np.double,
    #                                                   n_mfcc=64)
    #             self.noisyMusicMfcc.append([noisyMusicTranspose])
    #         except librosa.util.exceptions.ParameterError as exception:
    #             print(exception)
    #             print(type(self.noisyMusicMfcc))
    #             print(noisyMusicTranspose)
    #
    #     self.folderIndex += 1
    #     self.noiseSampleIndex += 0
    #     return self.noiseMfcc[:len(self.noisyMusicMfcc)], np.array(self.noisyMusicMfcc)


if __name__ == "__main__":

    iterator = FileIterator()

    for _ in range(10):
        next(iterator)
