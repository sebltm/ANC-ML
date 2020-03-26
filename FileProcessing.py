import numpy as np
import librosa
import librosa.feature
import soundfile as sf
import torchaudio
from pathlib import Path

from torch.utils.data import Dataset


class NoisyMusicDataset(Dataset):

    def __init__(self, samplerate=44100):
        self.samplerate = samplerate
        self.folders = sorted(list(Path("../Sounds/Processed/").glob("*")))
        self.noiseFiles = sorted(list(Path("../Sounds/UrbanSound8K/audio/").glob("**/*.wav")))

        self.sampleIndex = 0

        self.noiseMfcc = np.empty((64, 112))
        self.noisyMusicMfcc = np.empty((64, 112))

    def __iter__(self):
        self.__next__()

    def __next__(self):
        self.noiseSamples = 30

        folder = self.folders[self.sampleIndex // self.noiseSamples]
        self.noisyMusicFiles = sorted(list(Path(folder).glob("[0-9]*.RAW")))

        sample_index = self.sampleIndex % self.noiseSamples\

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


if __name__ == "__main__":

    iterator = NoisyMusicDataset()

    for _ in range(10):
        next(iterator)
