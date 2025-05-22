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
    
# Embedding Layers
class Embedding(Module):
    ''' Embedding class, turns indexes into vectors. '''
    def __init__(self, in_size, embed_size):
        '''
        @param in_size (int): number of different indexes (vocabulary size).
        @param embed_size (int): size of the embedding vector generated.
        '''
        super().__init__()
        
        self.E = tensor(torch.random.randn(in_size, embed_size) / torch.sqrt(in_size), requires_grad=True)


    def forward(self, idx):
        # Extracts embedding from row "idx":
        x = self.E[idx._data]

        return x


class PositionalEmbedding(Module):
    ''' Embedding class, turns indexes into vectors. '''
    def __init__(self, n_timesteps, embed_size):
        '''
        @param n_timesteps (int): number of timesteps processed in each element in the batch.
        @param embed_size (int): size of the embedding vector generated.
        '''
        super().__init__()
        self.E = tensor(torch.random.randn(n_timesteps, embed_size) / torch.sqrt(n_timesteps), requires_grad=True)


    def forward(self, x):
        B, T = x.shape
        # Adds positional embeddings to input of size (batch_size,n_timesteps,embedding_dim):
        x = self.E[:T]
        return x
    
# Regularization Layers:
class Dropout(Module):
    ''' Dropout class, added usually after other layers, to drop values to zero with given probability. '''
    def __init__(self,drop_prob):
        '''
        @param drop_prob (float): probability to drop each value in input.
        '''
        super().__init__()
        self.p = drop_prob
        self.mode = 'train'
   
    def forward(self,z):
        if self.mode == 'eval':
            return z
        mask = rand(z.shape) > self.p
        a = z.masked_fill(mask, 0) 
        a = a / (1 - self.p)
        return a


class LayerNorm(Module):
    ''' Layer Norm class, added usually after other layers to normalize across all of the output. '''
    def __init__(self, n_embed):
        '''
        @param n_embed (float): size of the last dimention of the imput.
        '''
        super().__init__()
        self.gamma = ones([1, n_embed], requires_grad=True)
        self.beta = zeros([1, n_embed], requires_grad=True)
    

    def forward(self,x):
        var_x = var(x, dim=-1, keepdims=True) # (B, T)
        norm_x = (x - mean(x, dim=-1, keepdims=True)) / sqrt(var_x) # (B, T, D)
        z = norm_x * self.gamma + self.beta # (B, T, D)
        return z

# Non-Linearity Layers:
class ReLU(Module):
    ''' ReLU non-linearity class. '''
    def __init__(self):
        super().__init__()

    def forward(self, z):
        mask = Tensor(torch.where(z._data < 0, 0, 1))
        z = z * mask
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

    def forward(self, z, dim=None):
        z = exp(z)
        out = z / sum(z, dim=dim, keepdims=True)
        return out


class Tanh(Module):
    ''' Tanh non-linearity class. '''
    def __init__(self):
        super().__init__()

    def forward(self, z):
        z = exp(z)
        z_neg = exp(-z)
        out = (z - z_neg) / (z + z_neg)
        return out