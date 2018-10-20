from unittest import TestCase

import torch
from parallel_wavenet import *
from wavenet_dataset import *


class TestWaveNetModelWithConditioning(TestCase):
    def test_wavenet_mono(self):
        test_model = WaveNetModelWithConditioning(skip_channels=32, conditioning_period=128, conditioning_channels=[250, 16])
        test_input = torch.rand(1, 1, 4096)
        test_conditioning = torch.rand(1, 250, 16)
        test_output = test_model((test_input, test_conditioning))

        assert test_output.shape[0] == test_model.output_length