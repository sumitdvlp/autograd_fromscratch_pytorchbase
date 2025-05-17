import torch

# Engine Class with 
class Engine:
    ''' stores a single scalar value and its gradient '''
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = torch.zeros_like(data,dtype=float) if torch.is_tensor(data) else 0.0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Engine) else Engine(other)
        out = Engine(torch.add(self.data,other.data), (self, other), '+')
        # both matrix have either same dimensions OR other is broadcasted
        def _backward():
          self.grad = out.grad
          other.grad = out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Engine) else Engine(other)

        mul_res = self.data * other.data
        out = Engine(mul_res, (self, other), '*')
        def _backward():

            self.grad = out.grad * other.data
        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Engine) else Engine(other)
        mul_res = torch.mm(self.data, other.data)
        out = Engine(mul_res, (self, other), '@')
        # self - m x n
        # other -  n x k
        # out -  m x k
        ## m x k = m x n @ n x k

        def _backward():
            self.grad = torch.mm(out.grad,other.data.T) # (out) m x k @ k x n (other.T) > m x n
            other.grad = torch.mm(self.data.T, out.grad) # (self.T) n x m @ m x k (out) > n x k
        out._backward = _backward

        return out

    def __pow__(self, other):
        # assert isinstance(other, (torch.tensor., float)), "only supporting int/float powers for now"
        out = Engine(torch.pow(self.data, other), (self,), f'^{other}')

        def _backward():
            # x^3 = 3 * x ^ (3-1)
            internal_pow = torch.pow(self.data,torch.add(other,-1))
            mm_pow = other * internal_pow
            self.grad = mm_pow * out.grad
            ## self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out


    def relu(self):
        # out = Engine(torch.relu(self.data), (self,), 'ReLU')
        out = Engine(torch.where(self.data < 0.0, 0.0, self.data), (self,), 'ReLU')
        '''
          input shape = m x n
          output shape = m x n

        '''
        def _backward():
            self.grad = torch.where(self.data < 0.0,0.0,out.grad)
        out._backward = _backward

        return out


    def shape(self):
      return self.data.shape

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = torch.ones(self.data.shape,dtype=float)
        # print('topo',len(topo))
        for v in reversed(topo):
            # print('v op',v._op)
            # print('v __repr__',v.__repr__())
            v._backward()


    def __neg__(self): # -self
      out = Engine(torch.negative(self.data), (self,), f'-{self}')

      def _backward():
        self.grad = torch.negative(out.grad)
      out._backward = _backward

      return out

    def sum(self, dim=None,keepdim=False):
      out = Engine(torch.sum(self.data, dim=dim,keepdim=keepdim),(self,), f'sum')

      def _backward():
        self.grad = torch.ones_like(self.data) * out.grad
      out._backward = _backward

      return out

    def log(self):
      out = Engine(torch.log(self.data))

      def _backward():
        self.grad = (1/self.data) * out.grad
      out._backward = _backward
      return out

    def __exp__(self):
      out = Engine(torch.exp(self.data),(self,),f'exp')

      def _backward():
        self.grad = self.data * out.grad
      out._backward = _backward
      return out

    def max(self, dim, keepdim=False):
      out = Engine(torch.max(self.data,dim=dim,keepdim=keepdim).values,(self,),f'max')

      def _backward():
        if self.data.shape != self.grad.shape:
          # Brodcast upstream derivative to the size of "a":
          out.grad = torch.unsqueeze(out.grad, dim=dim)
          out.gad = out.grad * torch.ones_like(self.data)
          # Brodcast upstream output (max) to the size of "a":
          max_v = torch.unsqueeze(out, dim=dim)
          max_v = max_v * torch.ones_like(self.data)
        # Add upstream gradients to the [max] values
        self.grad = out.grad * torch.equal(self.data, max_v)

      out._backward = _backward

      return out

    def __getitem__(self, index):
      out = Engine(self.data[index],(self,),f'getitem')

      def _backward():
        self.grad = torch.zeros_like(self.data)
        self.grad[index] = out.grad

      out._backward = _backward
      return out

    def mean(self,dim=-1,keepdim=False):
      out = Engine(torch.mean(self.data,dim= None if dim == -1 else dim, keepdim=keepdim),(self,), f'mean({self.data})')

      def _backward():
        self.grad = torch.ones(self.shape()) * out.grad
        dino = torch.prod(torch.tensor(self.shape())[dim])
        # print('dino', dino)
        self.grad = self.grad / torch.prod(torch.tensor(self.shape())[dim])


      out._backward = _backward
      return out

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + other.__neg__()

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Engine(data={self.data}, grad={self.grad})"

    def zero_grad(self):
        ''' Reset the Tensor's gradients to zero. '''
        self.grad = torch.zeros_like(self.data)