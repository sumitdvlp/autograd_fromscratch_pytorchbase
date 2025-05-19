from autograd_engine import utils as engine
import torch
import unittest


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

        assert torch.equal(ct,c._data),"multiplication didn't worked"

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
        assert torch.equal(ct,c._data),"subtraction didn't worked"
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
        assert torch.equal(ct,c._data),"division forward didn't worked"
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
        assert torch.equal(ct,c._data),"pow forward didn't worked"
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
        assert torch.equal(ct,c._data),"pow forward didn't worked"
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
        assert torch.equal(ct,c._data),"exp forward didn't worked"
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
        assert torch.equal(ct,c._data),"division forward didn't worked"
        # backward pass check
        assert torch.equal(c.grad, ct.grad), "division backward pass didn't worked"
        assert torch.equal(e.grad, et.grad), "division backward pass didn't worked"
        assert torch.allclose(a.grad, at.grad), "division backward pass didn't worked"
        assert torch.allclose(b.grad, bt.grad), "division backward pass didn't worked"


seed_value=42
torch.manual_seed(seed_value)

if __name__ == '__main__':
    unittest.main()