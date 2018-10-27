import os
import os.path
import math
import threading
import torch
import torch.utils.data
import numpy as np
import librosa as lr
import bisect
import h5py
import scipy
try:
    import soundfile
except:
    print("importing soundfile package is not possible, audio files cannot be converted")
import time
from torch.autograd import Variable
from pathlib import Path


def list_all_audio_files(location):
    types = [".mp3", ".wav", ".aif", "aiff", ".flac", ".m4a"]
    audio_files = []
    for type in types:
        audio_files.extend(sorted(location.glob('**/*' + type)))
    if len(audio_files) == 0:
        print("found no audio files in " + str(location))
    return audio_files


class ParallelWavenetDataset(torch.utils.data.Dataset):
    """
    Dataset with audio files for parallel wavenet
    Args:
        location (string):          Location of the audio files
        item_length (Int):          Length of the examples (in frames)
        target_length (Int):        Length of the targets (in frames)
        sampling_rate (Int):        Sampling rate
        mono (Boolean):             reduce to mono if true
        classes (Int):              Number of possible values each sample can have
        test_stride (Int):          If test_stride > 1, every test_strideth example is placed in the test_set
        create_files (Boolean):     If true, new files with the specified sampling rate and specified mono/multichannel-
                                    setting will be created the first time the data set is created to speed up loading
    """
    def __init__(self,
                 location,
                 item_length,
                 target_length,
                 sampling_rate=16000,
                 mono=True,
                 test_stride=100,
                 create_files=True):

        self.location = Path(location)
        self.dataset_path = self.location / 'dataset'
        self.target_length = target_length
        self.sampling_rate = sampling_rate
        self.mono=mono
        self._item_length = item_length
        self._test_stride = test_stride
        self._length = 0
        self.start_samples = [0]
        self.train = True
        self.create_files = create_files

        try:
            _ = soundfile.available_formats()
        except:
            "print cannot create files, soundfile package not loaded"
            create_files = False

        if create_files:
            if self.dataset_path.exists():
                self.files = list_all_audio_files(self.dataset_path)
            else:
                unprocessed_files = list_all_audio_files(self.location)
                self.dataset_path.mkdir()
                self.create_dataset(unprocessed_files)
                self.files = list_all_audio_files(self.dataset_path)
        else:
            self.files = list_all_audio_files(self.location)

        self.calculate_length()

    def load_file(self, file, frames=-1, start=0):
        if self.create_files:
            data, _ = soundfile.read(file, frames, start, dtype='float32')
        else:
            data, _ = lr.load(file,
                              sr=self.sampling_rate,
                              mono=self.mono,
                              dtype=np.float32)
            if frames == -1:
                frames = data.size
            data = data[start:start+frames]
        return data

    def create_dataset(self, files):
        for i, file in enumerate(files):
            data, _ = lr.load(str(file), sr=self.sampling_rate, mono=self.mono, dtype=np.float32)
            new_name = 'file_' + str(i) + ".wav"
            new_file = self.dataset_path / new_name
            soundfile.write(str(new_file), data, samplerate=self.sampling_rate, subtype='PCM_16')
            #lr.output.write_wav(str(new_file), data, sr=self.sampling_rate)
            print("processed " + str(file))

    def calculate_length(self):
        """
        Calculate the number of items in this data sets.
        Additionally the start positions of each file are calculate in this method.
        """
        start_samples = [0]
        for idx in range(len(self.files)):
            file_data = self.load_file(str(self.files[idx]))
            start_samples.append(start_samples[-1] + file_data.size)
        available_length = start_samples[-1] - (self._item_length - (self.target_length - 1)) - 1
        self._length = math.floor(available_length / self.target_length)
        self.start_samples = start_samples

    def set_item_length(self, l):
        self._item_length = l
        self.calculate_length()

    @property
    def item_length(self):
        return self._item_length

    @item_length.setter
    def item_length(self, value):
        self._item_length = value
        self.calculate_length()

    def load_sample(self, file_index, position_in_file, item_length):
        """
        Load the specified audio sample from the audio files (the sample may span multiple files).
        """
        file_length = self.start_samples[file_index + 1] - self.start_samples[file_index]
        remaining_length = position_in_file + item_length + 1 - file_length
        if remaining_length < 0:
            sample = self.load_file(str(self.files[file_index]),
                                    frames=item_length + 1,
                                    start=position_in_file)
        else:
            # if the specified file is to short for this sample, recursively call this method and concatenate the clips
            this_sample = self.load_file(str(self.files[file_index]),
                                         frames=item_length - remaining_length,
                                         start=position_in_file)
            next_sample = self.load_sample(file_index + 1,
                                           position_in_file=0,
                                           item_length=remaining_length)
            sample = np.concatenate((this_sample, next_sample))
        return sample

    def get_position(self, idx):
        """
        :param idx: global index of the item in the dataset
        :return: file index of the item, position of the item in this file
        """
        if self._test_stride < 2:
            sample_index = idx * self.target_length
        elif self.train:
            sample_index = idx * self.target_length + math.floor(idx / (self._test_stride-1)) * self.target_length
        else:
            sample_index = self.target_length * (self._test_stride * (idx+1) - 1)

        file_index = bisect.bisect_left(self.start_samples, sample_index) - 1
        if file_index < 0:
            file_index = 0
        if file_index + 1 >= len(self.start_samples):
            print("error: sample index " + str(sample_index) + " is to high. Results in file_index " + str(file_index))
        position_in_file = sample_index - self.start_samples[file_index]
        return file_index, position_in_file

    def __getitem__(self, idx):
        file_index, position_in_file = self.get_position(idx)
        sample = self.load_sample(file_index, position_in_file, self._item_length)

        channel_count = 1
        if not self.mono:
            channel_count = sample.shape[-1]
        example = torch.from_numpy(sample[:self._item_length]).type(torch.FloatTensor).view(channel_count, -1)
        target = torch.from_numpy(sample[-self.target_length:]).type(torch.FloatTensor).view(channel_count, -1)
        return example, target

    def get_segment(self, position=0, file_index=0, duration=None):
        """
        Convenience function to get a segment from a file
        :param position: position in the file in seconds
        :param file_index: index of the file
        :param duration: the duration of the segment in seconds (plus the receptive field). If 'None', then only one receptive field is returned.
        :return: the specified segment (without labels)
        """
        position_in_file = (position // self.sampling_rate) - self.start_samples[file_index]
        if duration is None:
            item_length = self._item_length
        else:
            item_length = int(duration * self.sampling_rate)
        segment = self.load_sample(file_index, position_in_file, item_length)
        return segment

    def __len__(self):
        if self._test_stride > 1:
            test_length = math.floor(self._length / self._test_stride)
        else:
            test_length = 0
        if self.train:
            return self._length - test_length
        else:
            return test_length

    @staticmethod
    def process_batch(batch, dtype, ltype):
        example, target = batch
        example = Variable(example.type(dtype))
        target = Variable(target.type(ltype))
        return example, target