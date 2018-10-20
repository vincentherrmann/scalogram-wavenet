from unittest import TestCase

from constant_q_transform import CQT
from parallel_wavenet import *
from parallel_wavenet_trainer import *
from wavenet_dataset import *

class TestParallelWavenetTrainer(TestCase):
    def test_trainer(self):
        wavenet_settings = conditioning_wavenet_default_settings
        wavenet_settings['conditioning_channels'] = [256, 16]
        test_model = WaveNetModelWithConditioning(wavenet_settings)
        dataset = ParallelWavenetDataset('../audio_clips',
                                         item_length=test_model.input_length,
                                         target_length=test_model.output_length,
                                         test_stride=0)
        trainer = ParallelWavenetTrainer(test_model, dataset, CQT())
        trainer.train(4, 1)
