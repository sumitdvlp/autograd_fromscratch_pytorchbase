import torch

# Tensor class, with __init__, backward, magic methods, and utils:
class Tensor:
    ''' Tensor class, with __init__, backward, magic methods, and utils '''
    def __init__(self, data, requires_grad = False, operation = None,op = None) -> None:
        '''
        Creates new instance of the Tensor class.

        @param data (Array-like): Iterable containing the data to be stored in the Tensor.
        @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.
        @param operation (Operation Object): When a tensor is created from other tensors, this stores
        the operation that generated the new tensor (e.g. "Add", "Exp", "MatMul").
        '''
        self._data = data
        self.requires_grad = requires_grad
        self.operation = operation
        self.children = []
        self.shape = self._data.shape
        self.op = op
        if self.requires_grad:
            self.grad = torch.zeros_like(self._data)
    
    def __repr__(self):
        return f"""({self._data}, requires_grad = {self.requires_grad})"""

    def data(self):
        ''' Returns the data stored in the tensor as a Numpy Array. '''
        return self._data
    
    def backward(self, grad = None, z = None):
        ''' 
        Performs the backpropagation with gradient descent from current tensor.
        Will fill every tensor's "grad" attribute with gradients relative to "self" (current Tensor).
        '''
        if not self.requires_grad:
            return "this tensor has requires_grad set to False"
        
        if grad is None:
            grad = torch.ones_like(self._data)

        self.grad += grad

        if z is not None:
            self.children.remove(z)
        
        if self.operation:
            if not self.children:
                self.operation.backward(self.grad, self)

    def zero_grad(self):
        ''' Reset the Tensor's gradients to zero. '''
        self.grad = torch.zeros_like(self._data)

    def zero_grad_tree(self):
        ''' Reset the gradients of this Tensor, and of all of the Tensors that led to it. '''
        self.zero_grad()
        if self.operation:
            for parent in self.operation.parents:
                parent.zero_grad_tree()
            self.operation = None

    def __add__(self, other):
        """ New = self + other """
        op = Add()
        return op.forward(self, tensor(other))
    
    def sum(self, dim=None, keepdims=False):

        op = Sum()
        return op.forward(self, dim, keepdims=keepdims)

    def __mul__(self, other):
        """ New = self * other """
        op = Mul()
        return op.forward(self, tensor(other))

    
    def __truediv__(self, other):
        """ New = self / other """
        op = Div()
        return op.forward(self, tensor(other))

    
    def __sub__(self, other):
        """ New = self - other """
        return self + -other

    def __rsub__(self, other):
        """ New = other - self """
        return other + -self

    def __isub__(self, other):
        """ self -= other """
        return self + -other

    def __neg__(self):
        """ self = -self """
        op = Neg()
        return op.forward(self) 
    
    def __pow__(self, other):
        op = Pow()
        return op.forward(self, tensor(other))    

    def __matmul__(self, other):
        """ New = self @ other """
        op = MatMul()
        return op.forward(self, tensor(other))

    def transpose(self, *dims):
        """
        Returns the original tensor with the two given dimentions transposed.
        Example: (16, 8, 4), *dims=(-2,-1) -> (16, 4, 8)

        @param *dims (integers): two dimentions to be transposed.
        """
        op = Transpose()
        return op.forward(self, *dims)

    def mean(self, dim=None, keepdims=False):
        """
        Returns the mean of all values across the "dim" dimention.
        Example: (B, T, D), dim = 1 -> (B, D).

        @param dim (int): dimention to be averaged across.
        @param keepdims (bool): wether to broadcast result to same shape as input.
        """
        op = Mean()
        return op.forward(self, dim, keepdims=keepdims)

    def max(self, dim=None, keepdims=False):
        """
        Returns the largest values across the "dim" dimention.
        Example: (B, T, D), dim = 1 -> (B, D).
        
        @param dim (int): dimention to be reduced (only largest remains).
        @param keepdims (bool): wether to broadcast result to same shape as input.
        """
        op = Max()
        return op.forward(self, dim, keepdims=keepdims)

# Helper functions
def tensor(data):
    if isinstance(data, Tensor):
        return data
    else: 
        return Tensor(data)


# Core Implementations

class Add:
    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad
      
        # Get new Tensor's data:
        data = a._data + b._data
      
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self,op='+') 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z

    def backward(self, dz, z):
        a, b = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            da = dz

            # Rescale gradient to have the same shape as "a":
            '''
            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            '''
            a.backward(da, z)

        # Find gradients relative to "b", and pass it downstream:
        if b.requires_grad:
            db = dz

            # Rescale gradient to have the same shape as "b":
            '''
            grad_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            '''
            b.backward(db, z)

class Sum():
    def forward(self, a, dim, keepdims):
        requires_grad = a.requires_grad

        data = a._data.sum(dim=dim, keepdims=keepdims)

        z = Tensor(data, requires_grad=requires_grad, operation=self,op='sum()')

        self.parents = (a,)
        a.children.append(z)

        self.cache = (a)

        return z

    def backward(self, dz, z):
        a = self.cache

        if a.requires_grad:
            da = torch.ones_like(a._data) * dz
            a.backward(da, z)

class Mul:
    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad

        #  Get new Tensors data
        data = a._data * b._data
        z = Tensor(data, requires_grad=requires_grad, operation=self, op='*')

        self.parents = (a,b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a,b)

        return z

    def backward(self, dz, z):
        a, b = self.cache

        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            da = dz * b._data
        
            a.backward(da, z)
        
        if b.requires_grad:
            db = dz * a._data
            b.backward(db, z)
        
class Neg:

    def forward(self, a):
        requires_grad = a.requires_grad
   
        # Get new Tensor's data:
        data = -a._data 
   
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self,op='-') 
   
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)

        self.cache = a

        return z 
    
    def backward(self, dz, z):
        a = self.cache

        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            da = -dz
            a.backward(da, z)

class Div:

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad
       
        # Get new Tensor's data:
        data = a._data / b._data
       
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self,op='/') 
       
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z 
    
    def backward(self, dz, z):
        a, b = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # d/da(a/b) = (1/b), apply chain rule:
            da = dz * (1 / b._data)
            a.backward(da, z)

        # Find gradients relative to "b", and pass it downstream:
        if b.requires_grad:
            # d/db(a/b) = -(a/b^2), apply chain rule:
            db = - dz * a._data / (b._data ** 2)

            b.backward(db, z)

class Pow():
    def forward(self, tensor_a, tensor_b):
        requires_grad = tensor_a.requires_grad
        data = tensor_a._data ** tensor_b._data
        z = Tensor(data, requires_grad=requires_grad, operation=self,op='^')
        tensor_a.children.append(z)
        self.cache = (tensor_a, tensor_b)
        return z
    
    def backward(self, dz, z):
        tensor_a, tensor_b = self.cache
        if tensor_a.requires_grad:
            da = dz * (tensor_b._data * tensor_a._data ** (tensor_b._data-1))
            '''
            grad_dim = len(da.shape)
            in_dim = len(tensor_a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            for n, dim in enumerate(tensor_a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            '''
            tensor_a.backward(da, z)

class MatMul:

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad
     
        # Get new Tensor's data:
        data = a._data @ b._data
      
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self,op='@') 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z  

    def backward(self, dz, z):
        a, b = self.cache
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            da = dz @ b._data.swapaxes(-1,-2)
            a.backward(da, z)
        
        if b.requires_grad:
            db = a._data.swapaxes(-1,-2) @ dz
            b.backward(db, z)

class Transpose:

    def forward(self, a, *dims):
        requires_grad = a.requires_grad
       
        # Get new Tensor's data:
        data = a._data.swapaxes(*dims)
       
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self,op='.T')
       
        # Add new Tensors to "children" and old Tensors to "parents": 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, dims)

        return z
    
    def backward(self, dz, z):
        a, dims = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Transpose upstream gradients:
            da = dz.swapaxes(*dims)
 
            a.backward(da, z)


class Exp:

    def forward(self, a):
        requires_grad = a.requires_grad
       
        # Get new Tensor's data:
        data = torch.exp(a._data)
       
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self,op='exp') 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, data)

        return z
    
    def backward(self, dz, z):
        a, data = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # d/da(e^a) = e^a, apply the chain rule to the derivative of e^a:
            da = data * dz
            a.backward(da, z)

class Log:

    def forward(self, a):
        requires_grad = a.requires_grad
     
        # Get new Tensor's data:
        data = torch.log(a._data)
     
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self,op=f'log{a}') 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a)

        return z
    
    def backward(self, dz, z):
        a = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # d/da(ln(a)) = (1/a), apply the chain rule to the derivative of the natural log:
            da = (1 / a._data) * dz
            a.backward(da, z)

class Sqrt:

    def forward(self, a):
        requires_grad = a.requires_grad
     
        # Get new Tensor's data:
        data = torch.sqrt(a._data)
     
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self,op=f'sqrt w.r.t {a}') 
     
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, data)

        return z
    
    def backward(self, dz, z):
        a, data = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # d/dx(sqrt(a)) = (1/2) * (1/sqrt(a)), apply the chain rule to the derivative of the square root:
            da = (1 / 2) * (1 / data) * dz
            a.backward(da, z)

class Mean:

    def forward(self, a, dim, keepdims):
        requires_grad = a.requires_grad
    
        # Get new Tensor's data:
        data = a._data.mean(axis=dim, keepdims=keepdims)
      
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
       
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, dim)

        return z
    
    def backward(self, dz, z):
        a, dim =  self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Propagate through the mean(x) operation:
            da = torch.ones(a.shape) * dz
            da /= torch.prod(torch.tensor(a.shape)[dim])
            a.backward(da, z)

# Tensor Operations:
class Reshape:

    def forward(self, a, shape):
        requires_grad = a.requires_grad
      
        # Get new Tensor's data:
        data = a._data.reshape(*shape)
      
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self)
      
        # Add new Tensors to "children" and old Tensors to "parents": 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a)

        return z
    
    def backward(self, dz, z):
        a = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Reshape upstream gradients:
            da = dz.reshape(a.shape)
 
            a.backward(da, z)

class Max:

    def forward(self, a, dim, keepdims=False):
        requires_grad = a.requires_grad
      
        # Get new Tensor's data:
        data = torch.max(a._data, axis=dim, keepdims=keepdims)
        if keepdims:
            data = torch.ones(a.shape) * data

        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
     
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, data, dim)

        return z
    
    def backward(self, dz, z):
        a, data, dim =  self.cache

        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            max = data
            if a.shape != dz.shape: 
                # Brodcast upstream derivative to the size of "a":
                dz = torch.expand_dims(dz, axis=dim)
                dz = dz * torch.ones_like(a._data)
                # Brodcast upstream output (max) to the size of "a":
                max = torch.expand_dims(data, axis=dim)
                max = max * torch.ones_like(a._data)
            # Add upstream gradients to the [max] values:
            da = dz * torch.equal(a._data, max)
            a.backward(da, z)

class MaskedFill:

    def forward(self, a, condition, value):
        requires_grad = a.requires_grad
      
        # Get new Tensor's data:
        data = torch.where(condition, a._data, value)
      
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, condition)

        return z 
    
    def backward(self, dz, z):
        a, condition = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Because some activations are just set to a value, this operation is not differentiable.
            da = torch.where(condition, dz, 0)
 
            a.backward(da, z)

class Slice:

    def forward(self, a, index):
        requires_grad = a.requires_grad
      
        # Get new Tensor's data:
        data = a._data[index]
       
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
       
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, index)

        return z
    
    def backward(self, dz, z):
        a, index =  self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Add upstream gradients to [index] part of da.
            da = torch.zeros_like(a._data)
            da[index] = dz
            a.backward(da, z)

class Stack:
    
    def forward(self, tensors: tuple, dim: int):

        # Verify if any original tensors requires grad:
        requires_grad = False
        for tensor in tensors:
            if tensor.requires_grad == True:
                requires_grad = True
       
        # Get new Tensor's data:
        data = torch.stack([tensor._data for tensor in tensors], axis=dim)
       
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
       
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = tensors
        for tensor in tensors:
            tensor.children.append(z)
        self.cache = (tensors, dim)

        return z
    
    def backward(self, dz, z):
        tensors, dim = self.cache

        dz = torch.split(dz, len(tensors), dim)

        # Find gradients relative to each tensor in "tensor", and pass it downstream:
        for i, tensor in enumerate(tensors):
            if tensor.requires_grad:
                # For every tensor that generated the stack, get gradients relative to that part of "dz": 
                di = dz[i].reshape(tensor._data.shape)
    
                tensor.backward(di, z)

class Cat:

    def forward(self, tensors: tuple, dim: int):

        requires_grad = False
        for tensor in tensors:
            if tensor.requires_grad == True:
                requires_grad = True
    
        # Get new Tensor's data:
        data = torch.concatenate([tensor._data for tensor in tensors], axis=dim)
    
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
    
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = tensors
        for tensor in tensors:
            tensor.children.append(z)
        self.cache = (tensors, dim)

        return z
    
    def backward(self, dz, z):
        tensors, dim = self.cache
        
        dz = torch.split(dz, len(tensors), dim)

        # Find gradients relative to each tensor in "tensor", and pass it downstream:
        for i, tensor in enumerate(tensors):
            if tensor.requires_grad:
                # For every tensor that generated the output, get gradients relative to that part of "dz": 
                di = dz[i]
    
                tensor.backward(di, z)

class Transpose:

    def forward(self, a, *dims):
        requires_grad = a.requires_grad
       
        # Get new Tensor's data:
        data = a._data.swapaxes(*dims)
       
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self)
       
        # Add new Tensors to "children" and old Tensors to "parents": 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, dims)

        return z
    
    def backward(self, dz, z):
        a, dims = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Transpose upstream gradients:
            da = dz.swapaxes(*dims)
 
            a.backward(da, z)