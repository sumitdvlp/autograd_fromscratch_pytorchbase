
from autograd_engine.core import *

def tensor(data, requires_grad = False,op=''):
    return Tensor(data=data, requires_grad=requires_grad,op=op)