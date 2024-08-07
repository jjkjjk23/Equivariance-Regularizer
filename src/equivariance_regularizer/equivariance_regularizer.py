import torch
import torch.nn as nn
import random
from numpy import prod
import typing


class EquivarianceRegularizer(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        shape: tuple[int],
        transforms,
        dist: int = 2,
        n: int = 1,
        num_funcs: int = 1,
        bounds=[0, 1]
    ):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.dist = dist
        self.n = n
        self.num_funcs = num_funcs
        self.bounds = bounds
        self.shape = shape

    def __call__(self, inputs=None) -> torch.Tensor:
        funcs = random.choices(self.transforms, k=self.num_funcs)
        output = 0
        if inputs is None:
            inputs = self.sampler()
        else:
            self.n = inputs.shape[0]
        for f in funcs:
            # This is a trick that lets f be random as long as f[0]==f[1]
            # in which case the user should omit f[1] from the list
            if len(f) == 3:
                inputs = torch.cat([inputs, self.model(inputs)])
                inputs = f[0](inputs)
                error = self.distance(
                            self.model(inputs[:inputs.shape[0]//2]),
                            inputs[inputs.shape[0]//2:]
                        )
            else:
                assert len(f) == 4, "The transform should be given as a list of four elements: the output transform, the input transform, the threshold value epsilon, and the equivariance weight. If the input transform equals the output transform, the list may be of length three."
                error = self.distance(
                            f[0](self.model(inputs)),
                            self.model(f[1](inputs))
                        )
            output += f[-1]*nn.functional.relu(error - f[-2])
        return output

    def distance(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        out_shape = t1.shape
        if type(self.dist) not in [int, float, str]:
            return torch.mean(self.dist(t1, t2))
        if type(self.dist) in [int, float] or self.dist == 'inf':
            t1 = t1.reshape(self.n, prod(out_shape[1:]))
            t2 = t2.reshape(self.n, prod(out_shape[1:]))
            return torch.mean(
                torch.linalg.norm(t1-t2, dim=-1, ord=float(self.dist))
            )
        if self.dist == "cross_entropy_logits":
            t1 = torch.nn.functional.log_softmax(t1, dim=-1)
            t2 = torch.nn.functional.log_softmax(t2, dim=-1)
            t1 = t1.reshape(self.n, prod(out_shape[1:]))
            t2 = t2.reshape(self.n, prod(out_shape[1:]))
            return torch.mean(torch.linalg.norm(t1-t2, dim=-1))
        if self.dist == "cross_entropy":
            t1 = t1.reshape(self.n, prod(out_shape[1:]))
            t2 = t2.reshape(self.n, prod(out_shape[1:]))
            t1 = torch.nan_to_num(
                     torch.log(t1),
                     nan=0.0,
                     posinf=0.0,
                     neginf=0.0
                 )
            t2 = torch.nan_to_num(
                     torch.log(t2),
                     nan=0.0,
                     posinf=0.0,
                     neginf=0.0
                 )
            return torch.mean(torch.linalg.norm(t1-t2, dim=-1))

    def sampler(self) -> torch.Tensor:
        shape = (self.n,) + self.shape
        if self.bounds == [0, 1]:
            output = torch.rand(shape)
        else:
            if not torch.is_tensor(self.bounds):
                bounds = torch.tensor(self.bounds)
            bounds = torch.broadcast_to(bounds, shape + (2,))
            output = torch.rand(shape)*(bounds[..., 1]-bounds[..., 0])+bounds[..., 0]
        return output.to(next(self.model.parameters()).device)
