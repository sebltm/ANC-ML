from pathlib import Path

import librosa
import librosa.feature
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset


class NoisyMusicDataset(Dataset):

    def __init__(self, musicFolder, duration=1.30, samplerate=44100, folderIndex=0):
        self.samplerate = samplerate
        self.baseFolder = Path("/home/sebltm/OneDrive/Documents/Exeter/BSc_Dissertation/Sounds")
        self.baseNoiseFolder = self.baseFolder / "UrbanSound8K/audio"
        self.baseMusicFolder = self.baseFolder / musicFolder

        self.folderIndex = folderIndex
        self.musicIndex = 0

        self.duration = duration
        self.num_samples = int(duration * samplerate)

        self.noiseMfcc = np.empty((1, self.num_samples))
        self.noisyMusicMfcc = np.empty((1, self.num_samples))

    def __iter__(self):
        self.__next__()

    def __next__(self):
        if self.musicIndex == 1000:
            self.musicIndex = 0
            self.folderIndex += 1

        self.currentMusicFolder = self.baseMusicFolder / ("fold" + str(self.folderIndex))
        noiseMetadataFile = open((self.currentMusicFolder / "metadata.txt").as_posix())

        noiseFiles = [Path(line.strip()) for line in noiseMetadataFile]

        noiseFile = noiseFiles[self.musicIndex // 20]
        musicFile = self.currentMusicFolder / (str(self.musicIndex) + ".RAW")

        # SAMPLE A NOISE
        noise, _ = librosa.load(noiseFile.as_posix(), mono=True, sr=self.samplerate, duration=self.duration)
        self.noiseMfcc = noise[:self.num_samples]

        # SAMPLE A NOISY MUSIC
        raw_music = sf.SoundFile(musicFile.as_posix(), channels=1, samplerate=self.samplerate, subtype='FLOAT')
        raw_music_read = raw_music.read(dtype=np.float, always_2d=False)[:self.num_samples]
        music_transpose = raw_music_read.T
        self.noisyMusicMfcc = music_transpose

        self.musicIndex += 1

        return self.noiseMfcc, self.noisyMusicMfcc, noiseFile.as_posix(), musicFile.as_posix()


if __name__ == "__main__":

    iterator = NoisyMusicDataset()

    for _ in range(10):
        next(iterator)
