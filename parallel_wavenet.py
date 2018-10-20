import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable, Function
import numpy as np


def dilate(x, dilation, init_dilation=1, pad_start=True):
    """
    :param x: Tensor of size (N, C, L), where N is the input dilation, C is the number of channels, and L is the input length
    :param dilation: Target dilation. Will be the size of the first dimension of the output tensor.
    :param pad_start: If the input length is not compatible with the specified dilation, zero padding is used. This parameter determines wether the zeros are added at the start or at the end.
    :return: The dilated tensor of size (dilation, C, L*N / dilation). The output might be zero padded at the start
    """

    [n, c, l] = x.size()
    if n == 1 and l == 1:
        return x
    dilation_factor = dilation / init_dilation
    if dilation_factor == 1:
        return x

    # zero padding for reshaping
    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        l = new_l
        if pad_start:
            x = F.pad(x, (new_l - l + 1, 0))
        else:
            x = F.pad(x, (0, new_l - l + 1))

    l = math.ceil(l * init_dilation / dilation)
    n = math.ceil(n * dilation / init_dilation)

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(c, l, n)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x


wavenet_default_settings = {"layers": 10,
                            "blocks": 4,
                            "dilation_channels": 32,
                            "residual_channels": 32,
                            "skip_channels": 512,
                            "end_channels": [256, 128],
                            "output_channels": 1,
                            "output_length": 1024,
                            "kernel_size": 2,
                            "dilation_factor": 2,
                            "bias": True,
                            "dtype": torch.FloatTensor}


class WaveNetModel(nn.Module):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    def __init__(self, args_dict=wavenet_default_settings):

        super(WaveNetModel, self).__init__()

        self.layers = args_dict["layers"]
        self.blocks = args_dict["blocks"]
        self.dilation_channels = args_dict["dilation_channels"]
        self.residual_channels = args_dict["residual_channels"]
        self.skip_channels = args_dict["skip_channels"]
        self.end_channels = args_dict["end_channels"]
        self.output_channels = args_dict["output_channels"]
        self.output_length = args_dict["output_length"]
        self.kernel_size = args_dict["kernel_size"]
        self.dilation_factor = args_dict["dilation_factor"]
        self.dtype = args_dict["dtype"]
        self.use_bias = args_dict["bias"]

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.end_layers = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=1, #self.in_classes,
                                    out_channels=self.residual_channels,
                                    kernel_size=1,
                                    bias=self.use_bias)

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=1,
                                                     bias=self.use_bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=1,
                                                 bias=self.use_bias))

                receptive_field += additional_scope
                additional_scope *= self.dilation_factor
                init_dilation = new_dilation
                new_dilation *= self.dilation_factor

        in_channels = self.skip_channels
        for end_channel in self.end_channels:
            self.end_layers.append(nn.Conv1d(in_channels=in_channels,
                                             out_channels=end_channel,
                                             kernel_size=1,
                                             bias=True))
            in_channels = end_channel

        self.end_layers.append(nn.Conv1d(in_channels=in_channels,
                                         out_channels=self.output_channels,
                                         kernel_size=1,
                                         bias=True))

        # self.output_length = 2 ** (layers - 1)
        self.receptive_field = receptive_field
        self.activation_unit_init()

    @property
    def input_length(self):
        return self.receptive_field + self.output_length - 1

    def wavenet(self, input, dilation_func, activation_input={'x': None}):
        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #			 |----------------------------------------|     *residual*
            #            |                                        |
            # 			 |	  |-- conv -- tanh --|			      |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #				  |-- conv -- sigm --|     |
            #							              1x1
            #							               |
            # ---------------------------------------> + ------------->	*skip*

            (dilation, init_dilation) = self.dilations[i]

            residual = dilation_func(x, dilation, init_dilation)
            activation_input['x'] = residual

            # dilated convolution
            x = self.activation_unit(activation_input, i, dilation_func)

            # parametrized skip connection
            s = x.clone()
            if x.size(2) != 1:
                pass
            if x.size(2) != 0:  # 1: TODO: delete this line !? (why is it there?)
                 s = self.wavenet_dilate(x, 1, init_dilation=dilation)
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            del s

            x = self.residual_convs[i](x)
            x = x + residual[:, :, (self.kernel_size - 1):]

            del residual

        x = skip
        for this_layer in self.end_layers:
            x = this_layer(F.relu(x, inplace=True))

        return x

    def activation_unit_init(self):
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        for _ in range(len(self.skip_convs)):
            # dilated convolutions
            self.filter_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                               out_channels=self.dilation_channels,
                                               kernel_size=self.kernel_size,
                                               bias=self.use_bias))

            self.gate_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                             out_channels=self.dilation_channels,
                                             kernel_size=self.kernel_size,
                                             bias=self.use_bias))

    def activation_unit(self, input, layer_index, dilation_func):
        # gated activation unit
        filter = self.filter_convs[layer_index](input['x'])
        filter = torch.tanh(filter)
        gate = self.gate_convs[layer_index](input['x'])
        gate = torch.sigmoid(gate)
        x = filter * gate
        return x

    def wavenet_dilate(self, input, dilation, init_dilation):
        x = dilate(input, dilation, init_dilation)
        return x

    def forward(self, input):
        input = input[:, :, -(self.receptive_field + self.output_length - 1):]
        x = self.wavenet(input,
                         dilation_func=self.wavenet_dilate)

        # reshape output
        [n, c, l] = x.size()
        l = self.output_length
        x = x[:, :, -l:]
        #x = x.transpose(1, 2).contiguous()
        #x = x.view(n * l, c)
        return x


    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def cpu(self, type=torch.FloatTensor):
        super().cpu()


conditioning_wavenet_default_settings = wavenet_default_settings
conditioning_wavenet_default_settings["conditioning_channels"] = [16, 32, 16]
conditioning_wavenet_default_settings["conditioning_period"] = 128


class WaveNetModelWithConditioning(WaveNetModel):
    def __init__(self, args_dict=conditioning_wavenet_default_settings):
        self.conditioning_channels = args_dict["conditioning_channels"]
        self.conditioning_period = args_dict["conditioning_period"]

        super().__init__(args_dict)

        self.conditioning_layers = nn.ModuleList()
        self.file_conditioning_cross_layers = nn.ModuleList()
        for i in range(len(self.conditioning_channels)-1):
            self.conditioning_layers.append(nn.Conv1d(in_channels=self.conditioning_channels[i],
                                                      out_channels=self.conditioning_channels[i+1],
                                                      kernel_size=1,
                                                      bias=False if i == 0 else self.use_bias))

    def activation_unit_init(self):
        super().activation_unit_init()

        self.filter_conditioning_convs = nn.ModuleList()
        self.gate_conditioning_convs = nn.ModuleList()
        for l in range(len(self.skip_convs)):
            self.filter_conditioning_convs.append(nn.Conv1d(in_channels=self.conditioning_channels[-1],
                                                            out_channels=self.dilation_channels,
                                                            kernel_size=1,
                                                            bias=self.use_bias))

            self.gate_conditioning_convs.append(nn.Conv1d(in_channels=self.conditioning_channels[-1],
                                                          out_channels=self.dilation_channels,
                                                          kernel_size=1,
                                                          bias=self.use_bias))

    def forward(self, input):
        input, conditioning = input
        conditioning = self.conditioning_network(conditioning)

        activation_input = {'x': None, 'conditioning': conditioning}
        input = input[:, :, -(self.receptive_field + self.output_length - 1):]
        x = self.wavenet(input,
                         dilation_func=self.wavenet_dilate,
                         activation_input=activation_input)

        # reshape output
        [n, c, l] = x.size()
        l = self.output_length
        x = x[:, :, -l:]
        #x = x.transpose(1, 2).contiguous()
        #x = x.view(n * l, c)
        return x

    def activation_unit(self, input, layer_index, dilation_func):
        # gated activation unit with conditioning
        filter = self.filter_convs[layer_index](input['x'])
        gate = self.gate_convs[layer_index](input['x'])

        conditioning = input['conditioning']
        dilation = input['x'].size(0) // conditioning.size(0)

        filter_conditioning = self.filter_conditioning_convs[layer_index](conditioning)
        gate_conditioning = self.gate_conditioning_convs[layer_index](conditioning)

        n, c, _ = filter_conditioning.shape
        l = filter.size(2)

        # upsample conditioning by repeating the values (could also be done with a transposed convolution)
        filter_conditioning = filter_conditioning.repeat(1, 1, 1, self.conditioning_period).view(n, c, -1)
        gate_conditioning = gate_conditioning.repeat(1, 1, 1, self.conditioning_period).view(n, c, -1)
        # dilate to current dilation level
        filter_conditioning = dilation_func(filter_conditioning, dilation, init_dilation=1)
        gate_conditioning = dilation_func(gate_conditioning, dilation, init_dilation=1)
        # possibly cut off end make it the same shape as filter/gate
        filter_conditioning = filter_conditioning[:, :, :l]
        gate_conditioning = gate_conditioning[:, :, :l]

        filter = torch.tanh(filter + filter_conditioning)
        gate = torch.sigmoid(gate + gate_conditioning)

        x = filter * gate
        return x

    def conditioning_network(self, conditioning):
        for l in range(len(self.conditioning_layers)):
            conditioning = self.conditioning_layers[l](conditioning)
            conditioning = F.elu(conditioning, inplace=True)
        return conditioning

    def conditioning_parameters(self):
        conditioning_modules = [self.conditioning_layers]
        parameters = []
        for m in conditioning_modules:
            parameters.extend(m.parameters())
        return parameters
