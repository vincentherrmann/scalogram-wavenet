import torch
import torch.optim
import torch.utils.data
import time
from constant_q_transform import *
from parallel_wavenet import *


class ParallelWavenetTrainer:
    def __init__(self, model: WaveNetModel, dataset, cqt_module, logger):
        self.model = model
        self.dataset = dataset
        self.cqt_module = cqt_module
        self.logger = logger
        self.loss_func = nn.MSELoss()
        self.epsilon = 1e-10

        self.cqt_time = 0
        self.model_time = 0
        self.loading_time = 0

    def train(self,
              batch_size=32,
              epochs=10,
              lr=0.0001,
              continue_training_at_step=0,
              num_workers=1):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers,
                                                 pin_memory=False)

        dev = next(self.model.parameters()).device

        step = continue_training_at_step
        tic = time.time()
        for current_epoch in range(epochs):
            print("epoch", current_epoch)
            for batch in iter(dataloader):
                toc = time.time()
                self.loading_time = toc - tic
                example_signal = torch.cat([batch[0], batch[1]], dim=2).to(dev)
                example_signal.requires_grad = False
                tic = time.time()
                example_cqt = torch.log(abs(self.cqt_module(example_signal))**2 + self.epsilon)
                toc = time.time()
                self.cqt_time = toc-tic

                input_noise = torch.randn(batch_size, 1, self.model.input_length, device=dev)

                tic = time.time()
                output = self.model((input_noise, example_cqt))
                toc = time.time()
                self.model_time = toc - tic

                # create output cqt
                output_signal = torch.cat([batch[0].to(dev), output], dim=2).to(dev)
                output_cqt = self.cqt_module(output_signal)
                output_cqt = abs(output_cqt + self.epsilon)**2
                output_cqt = torch.log(output_cqt + self.epsilon)

                loss = self.loss_func(output_cqt, example_cqt)

                if torch.isnan(loss).item() > 0:
                    print("error, loss is nan.")

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1

                self.logger.log(step, loss.item())

                #if step % 1 == 0:
                #    print("step", step, "- loss:", loss.item(), "- cqt:", self.cqt_time, "s - model:", self.model_time,
                #          "s - loading_time:", self.loading_time, "s")

                tic = time.time()
#
#
# class WavenetTrainer:
#     def __init__(self,
#                  model,
#                  dataset,
#                  optimizer=torch.optim.Adam,
#                  parameters=None,
#                  lr=0.001,
#                  weight_decay=0,
#                  gradient_clipping=None,
#                  logger=Logger(),
#                  snapshot_path=None,
#                  snapshot_name='snapshot',
#                  snapshot_interval=1000,
#                  snapshot_callback=None,
#                  dtype=torch.FloatTensor,
#                  ltype=torch.LongTensor,
#                  num_workers=8,
#                  pin_memory=False,
#                  process_batch=None,
#                  loss_fun=F.cross_entropy,
#                  accuracy_fun=softmax_accuracy):
#         self.model = model
#         self.dataset = dataset
#         self.dataloader = None
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.clip = gradient_clipping
#         self.optimizer_type = optimizer
#         if parameters is None:
#             self.optimizer = self.optimizer_type(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#         else:
#             self.optimizer = self.optimizer_type(params=parameters, lr=self.lr, weight_decay=self.weight_decay)
#         self.logger = logger
#         self.logger.trainer = self
#         self.snapshot_path = snapshot_path
#         self.snapshot_name = snapshot_name
#         self.snapshot_interval = snapshot_interval
#         self.snapshot_callback = snapshot_callback
#         self.dtype = dtype
#         self.ltype = ltype
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.process_batch = process_batch
#         self.loss_fun = loss_fun
#         self.accuracy_fun = accuracy_fun
#
#         self.num_step_track = 10
#         self._tic = 0
#         self._toc = 0
#         self._step_times = np.zeros(self.num_step_track)
#
#     def track_step_times(self, step, begin_tracking_step=0):
#         if step - begin_tracking_step == 0:
#             self._tic = time.time()
#         elif step - begin_tracking_step <= self.num_step_track:
#             self._toc = time.time()
#             self._step_times[step - begin_tracking_step - 1] = self._toc - self._tic
#             self._tic = time.time()
#             if step - begin_tracking_step == self.num_step_track:
#                 mean = np.mean(self._step_times)
#                 std = np.std(self._step_times)
#                 print("one training step does take " + str(mean) + " +/- " + str(std) + " seconds")
#                 self._step_times = np.zeros(self.num_step_track)
#
#     def train(self,
#               batch_size=32,
#               epochs=10,
#               continue_training_at_step=0):
#         self.model.train()
#         self.dataloader = torch.utils.data.DataLoader(self.dataset,
#                                                       batch_size=batch_size,
#                                                       shuffle=True,
#                                                       num_workers=self.num_workers,
#                                                       pin_memory=self.pin_memory)
#         step = continue_training_at_step
#         for current_epoch in range(epochs):
#             print("epoch", current_epoch)
#             for batch in iter(self.dataloader):
#                 if self.process_batch is None:
#                     x, target = batch
#                     x = Variable(x.type(self.dtype))
#                     target = Variable(target.type(self.ltype))
#                 else:
#                     x, target = self.process_batch(batch, self.dtype, self.ltype)
#                 output = self.model(x)
#                 target = target.view(-1)
#                 loss = self.loss_fun(output.squeeze(), target.squeeze())
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 loss = loss.data[0]
#
#                 self.optimizer.step()
#
#                 self.track_step_times(step, begin_tracking_step=continue_training_at_step)
#
#                 step += 1
#
#                 if step % self.snapshot_interval == 0:
#                     if self.snapshot_path is None:
#                         continue
#                     time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
#                     torch.save(self.model, self.snapshot_path + '/' + self.snapshot_name + '_' + time_string)
#                     if self.snapshot_callback is not None:
#                         self.snapshot_callback()
#
#                 self.logger.log(step, loss)
#
#     def validate(self):
#         self.model.eval()
#         dataset_state = self.dataset.train  # remember the current state
#         self.dataset.train = False
#         total_loss = 0
#         accurate_classifications = 0
#         for batch in iter(self.dataloader):
#             if self.process_batch is None:
#                 x, target = batch
#                 x = Variable(x.type(self.dtype))
#                 target = Variable(target.type(self.ltype))
#             else:
#                 x, target = self.process_batch(batch, self.dtype, self.ltype)
#
#             output = self.model(x)
#             target = target.view(-1)
#             loss = self.loss_fun(output.squeeze(), target.squeeze())
#             total_loss += loss.data[0]
#
#             correct_predictions = self.accuracy_fun(output, target)
#             accurate_classifications += correct_predictions
#         # print("validate model with " + str(len(self.dataloader.dataset)) + " samples")
#         # print("average loss: ", total_loss / len(self.dataloader))
#         avg_loss = total_loss / len(self.dataloader)
#         avg_accuracy = accurate_classifications / (len(self.dataset)*self.dataset.target_length)
#         self.dataset.train = dataset_state
#         self.model.train()
#         return avg_loss, avg_accuracy