import torch
import torch.nn as nn
import librosa as lr
import numpy as np
import math


def complex_multiply(a, b, complex_dim_a=None, complex_dim_b=None):
    # if a.shape != b.shape:
    #    print('a and b must have the same shape')
    #    print('shape a:', a.shape, 'shape b:', b.shape)

    r = torch.LongTensor([0]).to(a.device)

    if complex_dim_a is None:
        complex_dim_a = len(a.shape) - 1

    if complex_dim_b is None:
        complex_dim_b = len(b.shape) - 1

    real_a = torch.index_select(a, complex_dim_a, r).squeeze(complex_dim_a)
    imag_a = torch.index_select(a, complex_dim_a, r+1).squeeze(complex_dim_a)
    real_b = torch.index_select(b, complex_dim_b, r).squeeze(complex_dim_b)
    imag_b = torch.index_select(b, complex_dim_b, r+1).squeeze(complex_dim_b)

    product_real = real_a * real_b - imag_a * imag_b
    product_imag = real_a * imag_b + imag_a * real_b

    stack_dim = max(complex_dim_a, complex_dim_b)
    return torch.stack([product_real, product_imag], dim=stack_dim)


def abs(z, complex_dim=None):
    r = torch.LongTensor([0]).to(z.device)

    if complex_dim is None:
        complex_dim = len(z.shape) - 1
    real = torch.index_select(z, complex_dim, r).squeeze(dim=complex_dim)
    imag = torch.index_select(z, complex_dim, r+1).squeeze(dim=complex_dim)
    return torch.sqrt(real ** 2 + imag ** 2)


def angle(z, complex_dim=None):
    r = torch.LongTensor([0]).to(z.device)
    if complex_dim is None:
        complex_dim = len(z.shape) - 1
    real = torch.index_select(z, complex_dim, r).squeeze(dim=complex_dim)
    imag = torch.index_select(z, complex_dim, r+1).squeeze(dim=complex_dim)
    return torch.atan2(imag, real)


def to_complex(real, imag, complex_dim=None):
    if complex_dim is None:
        complex_dim = len(real.shape)
    return torch.stack([real, imag], dim=complex_dim)


def ifft_shift(x):
    n = math.ceil(x.shape[1] / 2)
    m = math.floor(x.shape[1] / 2)
    shifted_x = torch.zeros_like(x)
    shifted_x[:, :m, :] = x[:, n:, :]
    shifted_x[:, m:, :] = x[:, :n, :]
    return shifted_x


def torch_cqt(x, filters, norm_factors=1., hop_length=128):
    x_fft = torch.rfft(x, signal_ndim=1, onesided=True, normalized=False)
    product = complex_multiply(x_fft.unsqueeze(1), filters[:, :x_fft.shape[1], :].unsqueeze(0))
    cqt = torch.ifft(product, signal_ndim=1, normalized=False)[:, ::hop_length]
    cqt = ifft_shift(cqt)

    cqt *= norm_factors * 0.5
    return cqt


class CQT(nn.Module):
    def __init__(self, sr=16000, fmin=30, n_bins=256, bins_per_octave=32, filter_scale=1., hop_length=128):
        super().__init__()

        self.hop_length = hop_length

        # load filters
        cqt_filters, cqt_filter_lenghts = lr.filters.constant_q(sr,
                                                                fmin=fmin,
                                                                n_bins=n_bins,
                                                                bins_per_octave=bins_per_octave,
                                                                filter_scale=filter_scale)
        self.cqt_filter_lengths = cqt_filter_lenghts

        # one convolution operation per octave
        self.conv_kernel_sizes = []  # the kernel sizes of the octaves
        self.conv_index_ranges = []  # the indices belonging to each convolution operation
        current_kernel_size = None
        last_change_index = 0
        for i, l in enumerate(cqt_filter_lenghts):
            kernel_size = 2 ** math.ceil(np.log2(l))
            if current_kernel_size is not None and kernel_size >= current_kernel_size:
                # continue if this is in the same octave
                continue
            self.conv_kernel_sizes.append(kernel_size)
            current_kernel_size = kernel_size
            if i != 0:
                self.conv_index_ranges.append(range(last_change_index, i))
            last_change_index = i
        self.conv_index_ranges.append(range(last_change_index, len(self.cqt_filter_lengths)))

        filter_length = cqt_filters.shape[-1]
        self.conv_modules = []
        for i, size in enumerate(self.conv_kernel_sizes):
            this_range = self.conv_index_ranges[i]
            offset = (filter_length - size) // 2
            if offset > 0:
                this_filter = cqt_filters[this_range, offset:-offset]
            else:
                this_filter = cqt_filters[this_range, :]
            this_filter = torch.cat([torch.from_numpy(np.real(this_filter)),
                                     torch.from_numpy(np.imag(this_filter))], dim=0).type(torch.FloatTensor)
            # print(this_filter.shape)
            this_conv = nn.Conv1d(in_channels=1, out_channels=this_filter.shape[0], kernel_size=size, bias=False,
                                  stride=hop_length, padding=size // 2)
            this_conv.weight = torch.nn.Parameter(this_filter.unsqueeze(1), requires_grad=False) # should be False
            self.conv_modules.append(this_conv)

    def forward(self, x):
        real = []
        imag = []
        for conv in self.conv_modules:
            conv_result = conv(x)
            r, i = torch.chunk(conv_result, 2, dim=1)
            real.append(r)
            imag.append(i)
        real = torch.cat(real, dim=1)
        imag = torch.cat(imag, dim=1)
        return torch.stack([real, imag], dim=3)

    def to(self, device):
        super().to(device)
        for conv in self.conv_modules:
            conv.to(device)

def debug_hook(grad):
    print("grad:", grad)
    return grad

def debug_backward_hook(module, grad_input, grad_output):
    print("passed through backward hook")
    diagnostic_hook(module, grad_input, grad_output)
    pass

def diagnostic_hook(module, grad_input, grad_output):
    module += 1
    print("module:", module, "- grad in:", grad_input, "- grad out:", grad_output)