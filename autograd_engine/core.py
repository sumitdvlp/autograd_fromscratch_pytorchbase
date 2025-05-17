import torch

# Tensor class, with __init__, backward, magic methods, and utils:
class Tensor:
    ''' Tensor class, with __init__, backward, magic methods, and utils '''
    def __init__(self, data, requires_grad = False, operation = None,op="") -> None:
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
        self.op = ""
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

    



# Helper functions
def tensor(data):
    if isinstance(data, Tensor):
        return data
    else: 
        return Tensor(data)
    









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
            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da, z)

        # Find gradients relative to "b", and pass it downstream:
        if b.requires_grad:
            db = dz

            # Rescale gradient to have the same shape as "b":
            grad_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
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