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

    def test_trainer_with_dataset(self):
        # setup model
        model_params = conditioning_wavenet_default_settings
        model_params["bias"] = False
        model_params["dilation_channels"] = 16
        model_params["residual_channels"] = 16
        model_params["skip_channels"] = 16
        model_params["end_channels"] = [32, 32]
        model_params["conditioning_channels"] = [250]
        model_params["conditioning_period"] = 128
        model_params["output_length"] = 1024
        model = WaveNetModelWithConditioning(model_params)

        # setup dataset
        print("load dataset...")
        dataset = ParallelWavenetDataset('/Volumes/Elements/Projekte/scalogram-wavenet/house_dataset',
                                         item_length=model.input_length,
                                         target_length=model.output_length,
                                         test_stride=0)
        print("dataset length", len(dataset))

        cqt_module = CQT(n_bins=250)
        #cqt_module.register_backward_hook(debug_backward_hook)
        #model.register_backward_hook(debug_backward_hook)
        trainer = ParallelWavenetTrainer(model, dataset, cqt_module)
        trainer.train(batch_size=8, epochs=1)

