
from autograd_engine.core import *
from autograd_engine.nn_modules.layers import *

def tensor(data, requires_grad = False,op=''):
    '''
    Creates new instance of the Tensor class.

    @param data (Array-like): Iterable containing the data to be stored in the Tensor.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containing "data".
    '''
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


def parameter(data, requires_grad=True):
    '''
    Creates a Parameter for your model (an instance of the Tensor class).

    @param data (Array-like): Iterable containing the data to be stored in the Tensor.

    @returns Tensor (Tensor): Tensor containing "data".
    '''
    return Parameter(data, requires_grad=True)

def zeros(shape, requires_grad = False):
    '''
    Creates new instance of the Tensor class, filled with zeros.

    @param shape (tuple): iterable with the shape of the resulting Tensor.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining zeros with "shape" shape.
    '''
    data = torch.zeros(shape)
    return Tensor(data, requires_grad=requires_grad)

def ones(shape, requires_grad = False):
    '''
    Creates new instance of the Tensor class, filled with ones.

    @param shape (tuple): iterable with the shape of the resulting Tensor.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining ones with "shape" shape.
    '''
    data = torch.ones(shape)
    return Tensor(data, requires_grad=requires_grad)

def randint(low: int = 0, high: int = None, shape: tuple = (1), requires_grad: bool = False):
    '''
    Creates new instance of the Tensor class, filled with random integers.

    @param low (int): lowest integer to be generated. [OPTIONAL]
    @param high (int): one above the highest integer to be generated.
    @param shape (tuple): iterable with the shape of the resulting Tensor.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining random integers with "shape" shape.
    '''
    if type(high).__name__ == 'int':
        data = torch.randint(low, high, size=shape)
    else:
        data = torch.randint(low, size=high)
    return Tensor(data, requires_grad=requires_grad)

def randn(shape, xavier = False, requires_grad = False):
    '''
    Creates new instance of the Tensor class, filled with floating point numbers in a normal distribution.

    @param shape (tuple): iterable with the shape of the resulting Tensor.
    @param xavier (Bool): Whether to use Xavier initialization on tensor (scale by squre root of first dimension).
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining normally distributed floats with "shape" shape.
    '''
    data = torch.randn(*shape)
    if xavier:
        data /= torch.sqrt(shape[0])
    return Tensor(data, requires_grad=requires_grad)

def rand(shape, requires_grad = False):
    '''
    Creates new instance of the Tensor class, filled with floating point numbers in a normal distribution.

    @param shape (tuple): iterable with the shape of the resulting Tensor.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining normally distributed floats with "shape" shape.
    '''
    data = torch.random(shape)
    return Tensor(data, requires_grad=requires_grad)

def zeros_like(other: Tensor, requires_grad = False):
    '''
    Creates new instance of the Tensor class with same shape as given Tensor, and filled with zeros.
    @param other (Tensor): Tensor to copy shape from.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining zeros with other Tensor's shape.
    '''
    shape = other.shape
    return zeros(shape=shape, requires_grad=requires_grad)

def ones_like(other: Tensor, requires_grad = False):
    '''
    Creates new instance of the Tensor class with same shape as given Tensor, and filled with ones.
    @param other (Tensor): Tensor to copy shape from.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining ones with other Tensor's shape.
    '''
    shape = other.shape
    return ones(shape=shape, requires_grad=requires_grad)

def randn_like(other: Tensor, xavier = True, requires_grad = False):
    '''
    Creates new instance of the Tensor class with same shape as given Tensor,
    and filled with random floats in a normal distribution.
    @param other (Tensor): Tensor to copy shape from.
    @param xavier (Bool): Whether to use Xavier initialization on tensor (scale by squre root of first dimension).
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining normally distributed floats with other Tensor's shape.
    '''
    shape = other.shape
    return randn(shape=shape, xavier=xavier, requires_grad=requires_grad)

def randint_like(other: Tensor, low: int, high: int=0, requires_grad = False):
    '''
    Creates new instance of the Tensor class with same shape as given Tensor,
    and filled with random integers in the given distribution.
    @param other (Tensor): Tensor to copy shape from.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining normally distributed floats with other Tensor's shape.
    '''
    shape = other.shape
    if high == 0:
        return randint(low, shape, requires_grad=requires_grad)
    else:
        return randint(low, high, shape, requires_grad=requires_grad)
    
    # Methods to work with Tensors:

# Statistics:
def max(a: Tensor, dim: None, keepdims: bool=False):
    """
    Returns the largest values across the "dim" dimention.
    Example: (B, T, D), dim = 1 -> (B, D).

    @param a (Tensor): tensor to perform the max() operation.  
    @param dim (int): dimention to be reduced (only largest remains).
    @param keepdims (bool): wether to broadcast result to same shape as input.
    """
    return a.max(dim=dim, keepdims=keepdims)

def argmax(a: Tensor, dim: None, keepdims: bool=False):
    """
    Returns the index of the largest values across the "dim" dimention.
    Example: (B, T, D), dim = 1 -> (B, D).

    @param a (Tensor): tensor to perform the argmax() operation.  
    @param dim (int): dimention to be reduced (only largest index remains).
    @param keepdims (bool): wether to broadcast result to same shape as input.
    """
    return Tensor(torch.argmax(a._data,axis=dim,keepdims=keepdims))

def sum(a: Tensor, dim: None, keepdims: bool=False):
    """
    Returns the sum of all values across the "dim" dimention.
    Example: (B, T, D), dim = 1 -> (B, D).
        
    @param a (Tensor): tensor to perform the sum() operation.  
    @param dim (int): dimention to be summed across.
    @param keepdims (bool): wether to broadcast result to same shape as input.
    """
    return a.sum(dim=dim, keepdims=keepdims)

def mean(a: Tensor, dim: None, keepdims: bool=False):
    """
    Returns the mean of all values across the "dim" dimention.
    Example: (B, T, D), dim = 1 -> (B, D).

    @param a (Tensor): tensor to perform the mean() operation.  
    @param dim (int): dimention to be averaged across.
    @param keepdims (bool): wether to broadcast result to same shape as input.
    """
    return a.mean(dim=dim, keepdims=keepdims)

def var(a: Tensor, dim: None, keepdims: bool=False):
    """
    Returns the variance of all values across the "dim" dimention.
    Example: (B, T, D), dim = 1 -> (B, D).

    @param a (Tensor): tensor to perform the var() operation.    
    @param dim (int): dimention the variance will be computed across.
    @param keepdims (bool): wether to broadcast result to same shape as input.
    """
    return a.var(dim=dim, keepdims=keepdims)

def where(condition: any, a: Tensor, value: float):
    """
    Returns the "a" tensor with all values where condition is True set to "value".

    @param condition (Array-like): two dimentions to be transposed.
    @param a (Tensor): tensor to be filled by "value" where condition is True.    
    @param value (float): value to fill Tensor with, where condition is True.
    """
    return a.masked_fill(condition, value)

def cat(tensors: tuple, dim: int):
    """
    Concatenates all tensors across an existing dimention.
    Example: [(B, T, D), (C, T, D)], dim = 0 -> (B+C, T, D).

    @param tensors (list of Tensors): tensors to be concatenated.  
    @param dim (int): dimention to be concatenate across.
    """
    op = Cat()
    return op.forward(tensors, dim)

def stack(tensors: tuple, dim: int):
    """
    Stacks all tensors across a new dimention.
    Example: [(B, T, D), (C, T, D)], dim = 0 -> (2, B, T, D).

    @param tensors (list of Tensors): tensors to be stacked.  
    @param dim (int): position of the new dimention to stack across.
    """
    op = Stack()
    return op.forward(tensors, dim)