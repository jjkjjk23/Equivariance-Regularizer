import torch
import torch.nn as nn
import random
from numpy import prod

class EquivarianceRegularizer(nn.Module):
    def __init__(self, model, shape, transforms, dist=2, n=1, num_funcs=1, bounds=[0,1]):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.dist = dist
        self.n = n
        self.num_funcs = num_funcs
        self.bounds = bounds
        self.shape = shape
        
    def __call__(self):
        funcs = random.choices(self.transforms, k=self.num_funcs)
        output = 0
        inputs = self.sampler()
        for f in funcs:
            error = self.distance(f[0](self.model(inputs)), self.model(f[1](inputs)))
            output += f[3]*nn.functional.relu(error - f[2])
        return output
    def distance(self, t1,t2):
        out_shape = t1.shape
        if type(self.dist) not in [int, float, str]:
            return torch.mean(self.dist(t1,t2))
        if type(self.dist) in [int, float] or self.dist=='inf':
            t1 = t1.reshape(self.n, prod(out_shape[1:]))
            t2 = t2.reshape(self.n, prod(out_shape[1:]))
            return torch.mean(torch.linalg.norm(t1-t2, dim = -1, ord = float(self.dist)))
        if self.dist == "cross_entropy_logits":
            t1 = torch.nn.functional.log_softmax(t1, dim=-1)
            t2 = torch.nn.functional.log_softmax(t2, dim=-1)
            t1 = t1.reshape(self.n, prod(out_shape[1:]))
            t2 = t2.reshape(self.n, prod(out_shape[1:]))
            return torch.mean(torch.linalg.norm(t1-t2, dim = -1))
        if self.dist == "cross_entropy":
            t1 = t1.reshape(self.n, prod(out_shape[1:]))
            t2 = t2.reshape(self.n, prod(out_shape[1:]))
            t1 = torch.log(t1)
            t2 = torch.log(t2)
            return torch.mean(torch.linalg.norm(t1-t2, dim = -1))

    def sampler(self):
        shape=(self.n,)+self.shape
        if self.bounds == [0,1]:
            output = torch.rand(shape)
        else:
            if not torch.is_tensor(self.bounds):
                bounds=torch.tensor(self.bounds)
            bounds=torch.broadcast_to(bounds,shape+(2,))
            output = torch.rand(shape)*(bounds[...,1]-bounds[...,0])+bounds[...,0]
        return output.to(next(self.model.parameters()).device)
    