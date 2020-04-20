import warnings
from pathlib import Path

import librosa
import librosa.feature
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=UserWarning)


class NoisyMusicDataset(Dataset):

    def __init__(self, noisy_music_folder, max_sample=57330, sample_rate=44100, folder_index=0, mode='train'):
        self.sample_rate = sample_rate
        self.mode = mode
        self.baseFolder = Path("../Sounds")
        self.baseNoiseFolder = self.baseFolder / "UrbanSound8K/audio"
        self.baseNoisyMusicFolder = self.baseFolder / noisy_music_folder

        self.musicFolder = self.baseFolder / "fma_small"
        self.musics = list(self.musicFolder.glob("**/*.mp3"))

        self.folderIndex = folder_index
        self.musicIndex = 0

        self.num_samples = max_sample
        self.duration = max_sample / sample_rate

        self.currentMusicFolder = self.baseNoisyMusicFolder / ("fold" + str(self.folderIndex))

    def __iter__(self):
        self.__next__()

    def __next__(self):

        if self.musicIndex == 1000:
            self.musicIndex = 0
            self.folderIndex += 1

            self.currentMusicFolder = self.baseNoisyMusicFolder / ("fold" + str(self.folderIndex))

        # Read the metadata files to find where the noise and music files that were used to
        # generate the folder are located
        noise_metadata_file = open((self.currentMusicFolder / "metadataNOISE.txt").as_posix())
        music_metadata_file = open((self.currentMusicFolder / "metadataMUSIC.txt").as_posix())

        noise_files = [Path(line.strip()) for line in noise_metadata_file]
        music_files = [Path(line.strip()) for line in music_metadata_file]

        # Absolute Path
        noise_file = noise_files[self.musicIndex // 20]
        music_file = music_files[self.musicIndex // 50]
        noisy_music_file = self.currentMusicFolder / (str(self.musicIndex) + ".RAW")

        # Relative Path
        noise_file_parts = list(noise_file.parts[1:])
        noise_file = Path(*noise_file_parts)

        music_file_parts = list(music_file.parts[1:])
        music_file = Path(*music_file_parts)

        noise = []
        music = []

        # when training, we feed a music with no noise every 20 items, so the networks learns
        # to recognise when there is no noise
        # we don't need the music and noise when generating
        if self.musicIndex % 20 == 0 and (self.mode == 'train' or self.mode == 'trainMFCC'):

            # SAMPLE A NOISE
            noise = np.zeros((1, self.num_samples), dtype=np.float)

            # SAMPLE A MUSIC
            music, _ = librosa.load(music_file.as_posix(), mono=True, sr=self.sample_rate, dtype=np.float)

            self.musicIndex += 1

        elif self.mode == 'train' or self.mode == 'trainMFCC':
            # SAMPLE A NOISE
            noise, _ = librosa.load(noise_file.as_posix(), mono=True, sr=self.sample_rate, duration=self.duration)

            # SAMPLE A MUSIC
            music, _ = librosa.load(music_file.as_posix(), mono=True, sr=self.sample_rate, duration=self.duration)

            self.musicIndex += 1

        # we always need the noisy music, whether we are training or generating
        raw_noisy_music = sf.SoundFile(noisy_music_file.as_posix(), channels=1, samplerate=self.sample_rate,
                                       subtype='FLOAT')
        raw_noisy_music_read = raw_noisy_music.read(dtype=np.float, always_2d=False)[:self.num_samples]
        noisy_music_transpose = raw_noisy_music_read.T

        # compute the MFCC if we're training or generating with the MFCC model
        if self.mode == 'trainMFCC' or self.mode == 'generateMFCC':
            self.noise = librosa.feature.mfcc(noise, sr=self.sample_rate, n_mfcc=128)
            self.music = librosa.feature.mfcc(music, sr=self.sample_rate, n_mfcc=128)
            self.noisy_music = librosa.feature.mfcc(noisy_music_transpose, sr=self.sample_rate, n_mfcc=128)
        else:
            self.noise = noise[:self.num_samples]
            self.music = music[:self.num_samples]
            self.noisy_music = noisy_music_transpose

        return self.noise, self.noisy_music, self.music, noise_file.as_posix(), noisy_music_file.as_posix(), music_file.as_posix()


if __name__ == "__main__":

    iterator = NoisyMusicDataset(noisy_music_folder="Processed")

    for _ in range(10):
        next(iterator)
