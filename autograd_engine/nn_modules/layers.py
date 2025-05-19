from ..core import *
from ..utils import *
import torch

class Module:
    ''' General Module superclass'''
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        '''
        Returns all model parameters in a list. Iterates over each item in self.__dict__,
        and returns every Parameter object, or Tensor objects with requires_grad set to True.

        @returns params (list): All parameters in the model.
        '''
        params = []
        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                params += param.parameters()
            elif isinstance(param, Parameter):
                params.append(param)
            elif isinstance(param, Tensor):
                if param.requires_grad:
                    params.append(param)
        return params

    def train(self):
        ''' Sets module's mode to train, which influences layers like Dropout'''
        self.mode = 'train'
        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                param.train()


    def eval(self):
        ''' Sets module's mode to eval, which influences layers like Dropout'''
        self.mode = 'eval'
        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                param.eval()

# Parameter subclass, inherits from Tensor:
class Parameter(Tensor):
    ''' Subclass of Tensor which always tracks gradients. '''
    def __init__(self, data, requires_grad = True, operation = None,op='') -> None:
        super().__init__(data, requires_grad=requires_grad, operation=operation,op=op)

# Base Layers:
class Linear(Module):
    ''' Simple linear layer, with weight matrix and optional bias. Does not contain nonlinearity. '''
    def __init__(self, in_size: int, out_size: int, bias: bool = True):
        '''
        @param in_size (int): size of the last dimention of the input array.
        @param out_size (int): size of the last dimention of the output array.
        @param bias (bool): wether to include a bias term.
        '''
        super().__init__()
        self.W = tensor(torch.randn(in_size, out_size) / torch.sqrt(in_size), requires_grad=True)
        self.b = tensor(torch.zeros(out_size), requires_grad=True)
        self.has_bias = bias

    def forward(self, x):
        z = x @ self.W 
        if self.has_bias:
            z += self.b
        return z
    
class Softmax(Module):
    ''' Softmax non-linearity class. '''
    def __init__(self):
        super().__init__()

    def __call__(self, x, dim=-1):
        '''
        @param dim (int): dimention across which to apply Softmax.
        '''
        return self.forward(x, dim)

    def forward(self, z, dim=-1):
        z = exp(z)
        out = z / sum(z, dim=dim, keepdims=True)
        return out