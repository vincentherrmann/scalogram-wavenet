from unittest import TestCase

import torch
from parallel_wavenet import *
from wavenet_dataset import *


class TestWaveNetModel(TestCase):
    def test_wavenet_mono(self):
        test_model = WaveNetModel()
        test_input = torch.rand(1, 1, 4096)
        test_output = test_model(test_input)

        assert test_output.shape[0] == test_model.output_length