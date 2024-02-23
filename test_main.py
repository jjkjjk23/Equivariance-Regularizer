import unittest
import torch
import torch.nn as nn
from src.equivariance_regularizer import EquivarianceRegularizer

class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 2)

    def forward(self, tensor):
        return 2*tensor


def scale2(tensor):
    return 2*tensor


def transform1(tensor):
    return shift(tensor, 1, -1)


def transform2(tensor):
    return shift(tensor, 1, -1, .5)


def e(tensor):
    return tensor


def shift(x, shiftnum=1, axis=-1, fill = 0):
    x = torch.transpose(x, axis, -1)
    if shiftnum == 0:
        padded = x
    elif shiftnum > 0:
        paddings = [0 for j in range(2*len(tuple(x.shape)))]
        paddings[1]=shiftnum
        paddings=tuple(paddings)
        padded = nn.functional.pad(x[..., shiftnum:], paddings)
    elif shiftnum < 0:
        #paddings = (-shift, 0, 0, 0, 0, 0)
        paddings = [0 for j in range(2*len(tuple(x.shape)))]
        paddings[0]=-shiftnum
        paddings=tuple(paddings)
        padded = nn.functional.pad(x[..., :shiftnum], paddings, value=fill)
    else:
        raise ValueError
    return torch.transpose(padded, axis,-1)


class TestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize shared_variable here
        cls.device1 = torch.device('cpu')
        cls.model1 =  Model1().to(device = cls.device1, dtype = torch.float32)
        cls.tensor1 = torch.full((1, 2), 1, device=cls.device1, dtype = torch.float32)
        cls.tensor2 = shift(cls.tensor1, 1, -1, .5)

    def test1(self):
        er = EquivarianceRegularizer(self.model1, (1, 2), [[scale2, 0, 1]])
        self.assertEqual(er(), 0)

    def test2(self):
        er = EquivarianceRegularizer(self.model1, (1, 2), [[transform1, e,  0, 1]])

        self.assertEqual(er(self.tensor1), 2)

    def test_random_shift(self):
        er = EquivarianceRegularizer(self.model1, (1, 2), [[transform1,  0, 1]])

        self.assertEqual(er(), 0)

    def test_shift_ce(self):
        er = EquivarianceRegularizer(
                self.model1,
                (1, 2),
                [[transform2, 0, 1]],
                dist="cross_entropy"
             )
        self.assertEqual(er(self.tensor2), 0)


if __name__ == '__main__':
    unittest.main()
