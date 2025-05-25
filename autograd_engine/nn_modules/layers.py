from ..core import *
from autograd_engine import utils as engine
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
        mask = engine.rand(z.shape) > self.p
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
        self.gamma = engine.ones([1, n_embed], requires_grad=True)
        self.beta = engine.zeros([1, n_embed], requires_grad=True)
    

    def forward(self,x):
        var_x = engine.var(x, dim=-1, keepdims=True) # (B, T)
        norm_x = (x - engine.mean(x, dim=-1, keepdims=True)) / engine.sqrt(var_x) # (B, T, D)
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
        z = engine.exp(z)
        out = z / engine.sum(z, dim=dim, keepdims=True)
        return out


class Tanh(Module):
    ''' Tanh non-linearity class. '''
    def __init__(self):
        super().__init__()

    def forward(self, z):
        z_pos = engine.exp(z)
        z_neg = engine.exp(-z)
        out = (z_pos - z_neg) / (z_pos + z_neg)
        return out
    
# Loss Layer:
class MSE(Module):

  def __init__(self):
     super().__init__()

  def __call__(self, y, y_pred):
     return self.forward(y, y_pred)

  def forward(self, y:Tensor, y_pred:Tensor):
    diff = y - y_pred
    sq = diff ** 2
    y_shape = y.shape if isinstance(y, Tensor) else Tensor(y).shape
    return sq.sum()/y_shape[0]

class CrossEntropyLoss(Module):
    ''' Cross Entropy Loss class, returns the loss given the output and the expected indexes. '''
    def __init__(self):
        super().__init__()

    def __call__(self, z, y):
        '''
        @param z (Tensor): output from the last dimention of the network. 
        Must have shape like (*Batch dimentions, Number of possible classes).
        @param y (any Array): correct indexes expected from the model.
        Must have shape like (*Batch dimentions), with each value being the
        expected index.

        @returns loss (float): negative-log-likelihood loss of the model output.
        '''
        return self.forward(z, y)

    # def forward(self, z, y):
    #     *B_dims, D = z.shape
    #     B = torch.prod(B_dims)
    #     z = z.reshape(B,D)
        
    #     logits = engine.exp(z)
    #     logits = logits / sum(logits, dim= 1, keepdims=True)

    #     y = array(y).reshape(B)
            
    #     # get cross-entropy loss:
    #     log_losses = log(logits[np.arange(B), y])
    #     loss = -sum(log_losses) / (B)
    #     return loss
    def forward(self, y_pred:Tensor, y_true:Tensor):
        '''
        @param y_pred (Tensor): output from the last dimention of the network. 
        Must have shape like (*Batch dimentions, Number of possible classes).
        @param y_true (any Array): correct indexes expected from the model.
        Must have shape like (*Batch dimentions), with each value being the
        expected index.

        @returns loss (float): negative-log-likelihood loss of the model output.
        '''
        '''
        # Get batch size
        batch_size = y_pred._data.shape[0]

        # Apply softmax
        # exp_pred = torch.exp(y_pred.data - torch.max(y_pred.data,dim=1,keepdims=True).values)
        max_y_pred = engine.max(y_pred,dim=1,keepdims=True)
        sub = y_pred - max_y_pred
        exp_val = engine.exp(sub)
        # print('exp_val',exp_val)
        # print('exp_val', exp_val)
        exp_val_sum = engine.sum(exp_val,dim=1, keepdims=True)
        # print('exp_pred ',exp_pred[1:10])
        # print('exp_Pred', exp_pred.shape)
        # softmax_preds = exp_pred / torch.sum(exp_pred,dim=1, keepdims=True)
        softmax_preds = exp_val/exp_val_sum

        # Get Predicit probability for the correct class
        # print('softmax ',softmax_preds[1:10])
        # print('softmax ',softmax_preds.shape)
        correc_class_prds = softmax_preds[range(batch_size), y_true.data]

        # Calculate negative log likelihood

        loss = -correc_class_prds.log()

        return loss.mean()
        '''
        sftmax = y_pred.exp() / (y_pred.exp().sum(-1)).unsqueeze(-1)
        return -sftmax[range(y_true.shape[0]), y_true._data].log().mean()