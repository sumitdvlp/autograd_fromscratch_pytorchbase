from autograd_engine import utils as engine
import torch
import unittest
from autograd_engine.nn_modules import layers
import torch.nn as nn

class TestCoreOperations(unittest.TestCase):
    def test_add(self):
        at = torch.rand(2,1,requires_grad=True)
        bt = torch.rand(2,1,requires_grad=True)
        ct = at + bt
        ct.retain_grad()
        dt = torch.rand(2, 1,requires_grad=True)
        et =  ct + dt
        et.retain_grad()
        # et.requires_grad = True
        bct = et.sum()
        bct.retain_grad()
        bct.backward()
        
        
        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data=bt, requires_grad=True,op='b')
        c = a + b
        assert torch.equal(ct,c._data),"addition didn't worked"
        d = engine.tensor(data=dt, requires_grad=True, op='d')
        e = c + d
    
        bc = e.sum()
        bc.backward()
        

        assert torch.equal(c.grad, ct.grad), "addition backward pass didn't worked"
        assert torch.equal(e.grad, et.grad), "addition backward pass didn't worked"
        assert torch.equal(a.grad, at.grad), "addition backward pass didn't worked"
        assert torch.equal(b.grad, bt.grad), "addition backward pass didn't worked"

    def test_mul(self):
        at = torch.rand(2,4,2)
        bt = torch.rand(2,4,2)
        dt = torch.rand(2, 4,2)

        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data = bt, requires_grad=True,op='b')
        c = a * b
        
        d = engine.tensor(data=dt, requires_grad=True, op='d')
        e = c * d


        bc = e.sum()
        bc.backward()  

        at.requires_grad = True    
        bt.requires_grad = True
        dt.requires_grad = True

        ct = at * bt
        
        ct.retain_grad()
        et = ct * dt
        et.retain_grad()
        # et.requires_grad = True
        bct = et.sum()
        bct.retain_grad()
        bct.backward()

        assert torch.equal(bct,bc._data),"multiplication didn't worked"

        assert torch.equal(c.grad, ct.grad), "multiplication backward pass didn't worked"
        assert torch.equal(e.grad, et.grad), "multiplication backward pass didn't worked"
        assert torch.equal(a.grad, at.grad), "multiplication backward pass didn't worked"
        assert torch.equal(b.grad, bt.grad), "multiplication backward pass didn't worked"

    def test_sub(self):
        at = torch.rand(2,4,2)
        bt = torch.rand(2,4,2)
        dt = torch.rand(2, 4,2)

        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data = bt, requires_grad=True,op='b')
        c = a - b
        
        d = engine.tensor(data=dt, requires_grad=True, op='d')
        e = c - d


        bc = e.sum()
        bc.backward()  

        at.requires_grad = True    
        bt.requires_grad = True
        dt.requires_grad = True

        ct = at - bt
        
        ct.retain_grad()
        et = ct - dt
        et.retain_grad()
        # et.requires_grad = True
        bct = et.sum()
        bct.retain_grad()
        bct.backward()

        #forward pass check
        assert torch.equal(bct,bc._data),"subtraction didn't worked"
        # backward pass check
        assert torch.equal(c.grad, ct.grad), "subtraction backward pass didn't worked"
        assert torch.equal(e.grad, et.grad), "subtraction backward pass didn't worked"
        assert torch.equal(a.grad, at.grad), "subtraction backward pass didn't worked"
        assert torch.equal(b.grad, bt.grad), "subtraction backward pass didn't worked"
    
    def test_div(self):
        at = torch.rand(2,4,2)
        bt = torch.rand(2,4,2)
        dt = torch.rand(2, 4,2)

        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data = bt, requires_grad=True,op='b')
        c = a / b
        
        d = engine.tensor(data=dt, requires_grad=True, op='d')
        e = c / d


        bc = e.sum()
        bc.backward()  

        at.requires_grad = True    
        bt.requires_grad = True
        dt.requires_grad = True

        ct = at / bt
        
        ct.retain_grad()
        et = ct / dt
        et.retain_grad()
        # et.requires_grad = True
        bct = et.sum()
        bct.retain_grad()
        bct.backward()


        #forward pass check
        assert torch.equal(bct,bc._data),"division forward didn't worked"
        # backward pass check
        assert torch.equal(c.grad, ct.grad), "division backward pass didn't worked"
        assert torch.equal(e.grad, et.grad), "division backward pass didn't worked"
        assert torch.allclose(a.grad, at.grad), "division backward pass didn't worked"
        assert torch.allclose(b.grad, bt.grad), "division backward pass didn't worked"

    def test_pow(self):
        at = torch.rand(2,4,2)
        bt = torch.rand(2,4,2)
        dt = torch.rand(2, 4,2)

        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data = bt, requires_grad=True,op='b')
        c = a ** b
        
        d = engine.tensor(data=dt, requires_grad=True, op='d')
        e = c ** d


        bc = e.sum()
        bc.backward()  

        at.requires_grad = True    
        bt.requires_grad = True
        dt.requires_grad = True

        ct = at ** bt
        
        ct.retain_grad()
        et = ct ** dt
        et.retain_grad()
        # et.requires_grad = True
        bct = et.sum()
        bct.retain_grad()
        bct.backward()


        #forward pass check
        assert torch.equal(bct,bc._data),"pow forward didn't worked"
        # backward pass check
        assert torch.equal(c.grad, ct.grad), "pow backward pass didn't worked"
        assert torch.equal(e.grad, et.grad), "pow backward pass didn't worked"
        assert torch.allclose(a.grad, at.grad), "pow backward pass didn't worked"
        # We are not calculating backward pass of b in autograd.
        # assert torch.allclose(b.grad, bt.grad), "pow backward pass didn't worked"

    def test_matmul(self):
        at = torch.rand(2,4)
        bt = torch.rand(2,4).T
        dt = torch.rand(2, 4)

        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data = bt, requires_grad=True,op='b')
        c = a @ b
        
        d = engine.tensor(data=dt, requires_grad=True, op='d')
        e = c @ d


        bc = e.sum()
        bc.backward()  

        at.requires_grad = True    
        bt.requires_grad = True
        dt.requires_grad = True

        ct = at @ bt
        
        ct.retain_grad()
        et = ct @ dt
        et.retain_grad()
        # et.requires_grad = True
        bct = et.sum()
        bct.retain_grad()
        bct.backward()


        #forward pass check
        assert torch.equal(bct,bc._data),"pow forward didn't worked"
        # backward pass check
        assert torch.equal(c.grad, ct.grad), "pow backward pass didn't worked"
        assert torch.equal(e.grad, et.grad), "pow backward pass didn't worked"
        assert torch.allclose(a.grad, at.grad), "pow backward pass didn't worked"
        # We are not calculating backward pass of b in autograd.
        # assert torch.allclose(b.grad, bt.grad), "pow backward pass didn't worked"

    def test_exp(self):
        at = torch.rand(2,4,2)
        bt = torch.rand(2,4,2)
        dt = torch.rand(2, 4,2)

        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data = bt, requires_grad=True,op='b')
        
        ca = engine.exp(a)
        cb = engine.exp(b)
        c = ca + cb
        d = engine.tensor(data=dt, requires_grad=True, op='d')
        cd = engine.exp(d)
        e = c + cd


        bc = e.sum()
        bc.backward()  

        # Pytorch tensor
        at.requires_grad = True    
        bt.requires_grad = True
        dt.requires_grad = True

        cat = torch.exp(at)
        cbt = torch.exp(bt)
        ct = cat + cbt
        dt = torch.exp(dt)

        
        dt.retain_grad()
        et = ct + dt
        et.retain_grad()
        # et.requires_grad = True
        bct = et.sum()
        bct.retain_grad()
        bct.backward()

        print('ct ',ct)
        print('c.data',c._data)
        #forward pass check
        assert torch.equal(bct,bc._data),"exp forward didn't worked"
        # backward pass check
        assert torch.equal(c.grad, cd.grad), "exp backward pass didn't worked"
        assert torch.equal(e.grad, et.grad), "exp backward pass didn't worked"
        assert torch.allclose(a.grad, at.grad), "exp backward pass didn't worked"
        # We are not calculating backward pass of b in autograd.
        # assert torch.allclose(b.grad, bt.grad), "pow backward pass didn't worked"

    def test_mean(self):
        at = torch.rand(2,4)
        bt = torch.rand(2,4)
        dt = torch.rand(2, 4)

        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data = bt, requires_grad=True,op='b')
        c = a * b
        
        d = engine.tensor(data=dt, requires_grad=True, op='d')
        e = c * d


        bc = e.mean()
        bc.backward()  

        at.requires_grad = True    
        bt.requires_grad = True
        dt.requires_grad = True

        ct = at * bt
        
        ct.retain_grad()
        et = ct * dt
        et.retain_grad()
        # et.requires_grad = True
        bct = et.mean()
        bct.retain_grad()
        bct.backward()


        #forward pass check
        assert torch.equal(bct,bc._data),"division forward didn't worked"
        # backward pass check
        assert torch.equal(c.grad, ct.grad), "division backward pass didn't worked"
        assert torch.equal(e.grad, et.grad), "division backward pass didn't worked"
        assert torch.allclose(a.grad, at.grad), "division backward pass didn't worked"
        assert torch.allclose(b.grad, bt.grad), "division backward pass didn't worked"

    def test_relu(self):
        at = torch.rand(2,1)
        bt = torch.rand(2,1)
        dt = torch.rand(2, 1)

        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data = bt, requires_grad=True,op='b')
        c = a * b
        
        d = engine.tensor(data=dt, requires_grad=True, op='d')
        e = layers.ReLU()(c)


        bc = e.sum()
        bc.backward()  

        at.requires_grad = True    
        bt.requires_grad = True
        dt.requires_grad = True

        ct = at * bt
        relu = nn.ReLU()
        ct.retain_grad()
        et = relu(ct)
        et.retain_grad()
        # et.requires_grad = True
        
        bct = et.sum()
        bct.retain_grad()
        bct.backward()


        #forward pass check
        assert torch.equal(bct,bc._data),"relu forward didn't worked"
        # backward pass check
        assert torch.equal(c.grad, ct.grad), "relu backward pass didn't worked"
        assert torch.equal(e.grad, et.grad), "relu backward pass didn't worked"
        assert torch.allclose(a.grad, at.grad), "relu backward pass didn't worked"
        assert torch.allclose(b.grad, bt.grad), "relu backward pass didn't worked"

    def test_Softmax(self):
        at = torch.rand(2,1)
        bt = torch.rand(2,1)
        dt = torch.rand(2, 1)

        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data = bt, requires_grad=True,op='b')
        c = a * b
        
        d = engine.tensor(data=dt, requires_grad=True, op='d')
        e = layers.Softmax()(c)


        bc = e.sum()
        bc.backward()  

        at.requires_grad = True    
        bt.requires_grad = True
        dt.requires_grad = True

        ct = at * bt
        Softmax = nn.Softmax()
        ct.retain_grad()
        et = Softmax(ct)
        et.retain_grad()
        # et.requires_grad = True
        
        bct = et.sum()
        bct.retain_grad()
        bct.backward()


        #forward pass check
        assert torch.equal(bct,bc._data),"Softmax forward didn't worked"
        # backward pass check
        assert torch.equal(c.grad, ct.grad), "Softmax backward pass didn't worked"
        assert torch.equal(e.grad, et.grad), "Softmax backward pass didn't worked"
        assert torch.allclose(a.grad, at.grad), "Softmax backward pass didn't worked"
        assert torch.allclose(b.grad, bt.grad), "Softmax backward pass didn't worked"

    def test_MSE(self):
        at = torch.rand(2,1)
        bt = torch.rand(2,1)
        dt = torch.rand(2, 1)

        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data = bt, requires_grad=True,op='b')
        c = a * b
        
        d = engine.tensor(data=dt, requires_grad=True, op='d')
        e = layers.MSE()(c,d)

        bc = e.sum()
        bc.backward()  

        at.requires_grad = True    
        bt.requires_grad = True
        dt.requires_grad = True

        ct = at * bt
        Softmax = nn.MSELoss()
        ct.retain_grad()
        et = Softmax(ct,dt)
        et.retain_grad()
        
        bct = et.sum()
        bct.retain_grad()
        bct.backward()

        #forward pass check
        assert torch.equal(bct,bc._data),"MSE forward didn't worked"
        # backward pass check
        assert torch.equal(c.grad, ct.grad), "MSE backward pass didn't worked"
        assert torch.equal(e.grad, et.grad), "MSE backward pass didn't worked"
        assert torch.allclose(a.grad, at.grad), "MSE backward pass didn't worked"
        assert torch.allclose(b.grad, bt.grad), "MSE backward pass didn't worked"

    def test_Tanh(self):
        at = torch.rand(2,1)
        bt = torch.rand(2,1)
        dt = torch.rand(2, 1)

        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data = bt, requires_grad=True,op='b')
        c = a * b
        
        d = engine.tensor(data=dt, requires_grad=True, op='d')
        e = layers.Tanh()(c)


        bc = e.sum()
        bc.backward()  

        at.requires_grad = True    
        bt.requires_grad = True
        dt.requires_grad = True

        ct = at * bt
        Tanh = nn.Tanh()
        ct.retain_grad()
        et = Tanh(ct)
        et.retain_grad()
        # et.requires_grad = True
        
        bct = et.sum()
        bct.retain_grad()
        bct.backward()

        print('bct',bct.data)
        print('bc',bc._data)

        #forward pass check
        assert torch.allclose(bct,bc._data),"Tanh forward didn't worked"
        # backward pass check
        assert torch.allclose(c.grad, ct.grad), "Tanh backward pass didn't worked"
        assert torch.allclose(e.grad, et.grad), "Tanh backward pass didn't worked"
        assert torch.allclose(a.grad, at.grad), "Tanh backward pass didn't worked"
        assert torch.allclose(b.grad, bt.grad), "Tanh backward pass didn't worked"

    def test_Max(self):
        at = torch.rand(2,1)
        bt = torch.rand(2,1)
        # dt = torch.rand(2, 1)

        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data = bt, requires_grad=True,op='b')
        c = a * b
        
        # d = engine.tensor(data=dt, requires_grad=True, op='d')
        bc = engine.max(c, dim=0)
        bc.backward()  
  

        at.requires_grad = True    
        bt.requires_grad = True
        # dt.requires_grad = True

        ct = at * bt
        
        ct.retain_grad()
        bct = torch.max(ct, dim=0).values
        bct.retain_grad()
        bct.backward()

        print('bct',bct.data)
        print('bc',bc._data)

        #forward pass check
        assert torch.allclose(bct,bc._data),"Max forward didn't worked"
        # backward pass check
        assert torch.allclose(c.grad, ct.grad), "Max backward pass didn't worked"
        # assert torch.allclose(e.grad, et.grad), "Max backward pass didn't worked"
        assert torch.allclose(a.grad, at.grad), "Max backward pass didn't worked"
        assert torch.allclose(b.grad, bt.grad), "Max backward pass didn't worked"

    def test_Slice(self):
        batch_size, n_classes = 2, 4
        at = torch.rand(batch_size,n_classes)
        bt = torch.rand(batch_size,n_classes)
        dt = torch.randint(n_classes, size=(batch_size,))

        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data = bt, requires_grad=True,op='b')
        c = a * b
        
        e = c[range(0,1),[0,2,1]]

        bc = e.sum()
        bc.backward()  

        at.requires_grad = True    
        bt.requires_grad = True    

        ct = at * bt
        ct.retain_grad()
        et = ct[range(0,1),[0,2,1]]
        et.retain_grad()
        
        bct = et.sum()
        bct.retain_grad()
        bct.backward()

        #forward pass check
        assert torch.equal(bct,bc._data),"Slice forward didn't worked"
        # backward pass check
        assert torch.equal(c.grad, ct.grad), "Slice backward pass didn't worked"
        assert torch.equal(e.grad, et.grad), "Slice backward pass didn't worked"
        assert torch.allclose(a.grad, at.grad), "Slice backward pass didn't worked"
        assert torch.allclose(b.grad, bt.grad), "Slice backward pass didn't worked"

    def test_CrossEntropyLoss(self):
        batch_size, n_classes = 2, 4
        at = torch.rand(batch_size,n_classes)
        bt = torch.rand(batch_size,n_classes)
        dt = torch.randint(n_classes, size=(batch_size,))

        a = engine.tensor(data = at, requires_grad=True,op='a')
        b = engine.tensor(data = bt, requires_grad=True,op='b')
        c = a * b
        
        d = engine.tensor(data=dt, requires_grad=False, op='d')
        e = layers.CrossEntropyLoss()(c,d)


        bc = e.sum()
        bc.backward()  

        at.requires_grad = True    
        bt.requires_grad = True    
        # dt.requires_grad = True

        ct = at * bt
        CrossEntropyLoss = nn.CrossEntropyLoss()
        ct.retain_grad()
        et = CrossEntropyLoss(ct,dt)
        
        et.retain_grad()
        
        bct = et.sum()
        bct.retain_grad()
        bct.backward()

        print('E grad',e.grad)
        print('Et grad',et.grad)

        print('C grad',c.grad)
        print('Ct grad',ct.grad)

        print('A grad',a.grad)
        print('At grad',at.grad)

        print('b grad',b.grad)
        print('bt',bt.grad)


        #forward pass check
        assert torch.equal(bct,bc._data),"CrossEntropyLoss forward didn't worked"
        # backward pass check
        # assert torch.equal(c.grad, ct.grad), "Softmax backward pass didn't worked"
        # assert torch.equal(e.grad, et.grad), "CrossEntropyLoss backward pass didn't worked"
        assert torch.allclose(a.grad, at.grad), "Softmax backward pass didn't worked"
        assert torch.allclose(b.grad, bt.grad), "CrossEntropyLoss backward pass didn't worked"

seed_value=42
torch.manual_seed(seed_value)

if __name__ == '__main__':
    unittest.main()