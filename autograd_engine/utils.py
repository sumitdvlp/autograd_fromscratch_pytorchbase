
from autograd_engine.core import *

def tensor(data, requires_grad = False,op=''):
    return Tensor(data=data, requires_grad=requires_grad,op=op)

# Element-wise operations:
def exp(a: Tensor):
    ''' Element-wise exponentiation of the "a" Tensor. '''
    op = Exp()
    return op.forward(a)

def log(a: Tensor):
    ''' Element-wise natural logarithm of the "a" Tensor. '''
    op = Log()
    return op.forward(a)

def sqrt(a: Tensor):
    ''' Element-wise square root of the "a" Tensor. '''
    op = Sqrt()
    return op.forward(a)

def mean(a: Tensor, dim: None, keepdims: bool=False):
    """
    Returns the mean of all values across the "dim" dimention.
    Example: (B, T, D), dim = 1 -> (B, D).

    @param a (Tensor): tensor to perform the mean() operation.  
    @param dim (int): dimention to be averaged across.
    @param keepdims (bool): wether to broadcast result to same shape as input.
    """
    return a.mean(dim=dim, keepdims=keepdims)