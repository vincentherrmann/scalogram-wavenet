from unittest import TestCase

import torch
from constant_q_transform import *


class TestAbs(TestCase):
    def test_abs_backward(self):
        input = torch.randn([6, 4, 5, 2], requires_grad=True) * 0. + 1e-10
        x = abs(input).sum()
        grad_x, = torch.autograd.grad(x, input)

        print("input:", input)
        print("x:", x)
        print("grad_x:", grad_x)